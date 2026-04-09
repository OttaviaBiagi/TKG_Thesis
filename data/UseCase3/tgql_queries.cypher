// ═══════════════════════════════════════════════════════════════════════════
// T-GQL Temporal Path Queries for EPC Project Monitoring
// ═══════════════════════════════════════════════════════════════════════════
// These queries use temporal extensions to standard Cypher.
// SNAPSHOT, BETWEEN, WHEN semantics follow Debrouvier et al. (2021).
// For standard Neo4j, the equivalent temporal filter approach is shown.
// ═══════════════════════════════════════════════════════════════════════════


// ── Q1: SNAPSHOT query ──────────────────────────────────────────────────────
// "What was the project schedule state as of 2024-05-01?"
// (Standard Neo4j temporal filter equivalent)

MATCH (a:Activity)
WHERE a.planned_start <= date('2024-05-01')
  AND a.planned_finish >= date('2024-05-01')
RETURN a.id, a.status, a.pct_complete, a.planned_start, a.planned_finish
ORDER BY a.planned_start;


// ── Q2: BETWEEN query ──────────────────────────────────────────────────────
// "Which activities were active during the interval [2024-03-01, 2024-05-31]?"

MATCH (a:Activity)
WHERE a.planned_start >= date('2024-03-01')
  AND a.planned_start <= date('2024-05-31')
RETURN a.id, a.work_package, a.discipline, a.planned_start, a.planned_finish,
       a.status, a.delayed;


// ── Q3: WHEN clause (parallel-period query) ────────────────────────────────
// "Which activities were delayed WHILE their linked purchase order
//  was in late delivery status?"
// This is the key query not expressible in standard non-temporal Cypher.

MATCH (a:Activity)-[:REQUIRES_MATERIAL]->(po:PurchaseOrder)
WHERE a.delayed = true
  AND po.delivery_status = 'Late'
  AND po.actual_deliv_dt > po.planned_deliv_dt
  AND a.actual_start > a.planned_start
  AND po.planned_deliv_dt >= a.planned_start - duration({days: 60})
  AND po.planned_deliv_dt <= a.planned_start + duration({days: 5})
RETURN a.id             AS delayed_activity,
       a.work_package   AS work_package,
       po.id            AS late_po,
       po.delay_days    AS po_delay,
       duration.between(a.planned_start, a.actual_start).days AS act_delay_days,
       a.delay_reason   AS delay_reason
ORDER BY act_delay_days DESC;


// ── Q4: Multi-hop causal chain path ─────────────────────────────────────────
// "Find the complete causal chain from Event to affected Activity,
//  traversing through intermediate entities."

MATCH path = (e:Event)-[:IMPACTS_ACTIVITY]->(a:Activity)
WHERE e.triggers_delay_days > 0
  AND a.delayed = true
RETURN e.id           AS event,
       e.event_type   AS event_type,
       e.event_date   AS event_date,
       a.id           AS impacted_activity,
       a.work_package AS work_package,
       e.triggers_delay_days AS delay_caused,
       a.delay_reason AS confirmed_reason
ORDER BY e.event_date;


// ── Q5: Cross-source heterogeneous query ────────────────────────────────────
// "Which work packages had BOTH a document approval delay AND a procurement
//  delay within the same 60-day window?"
// This query requires data from Sources A, B, and C simultaneously.

MATCH (wp:WorkPackage)<-[:BELONGS_TO]-(a:Activity)-[:REQUIRES_DOCUMENT]->(doc:Document)
MATCH (a)-[:REQUIRES_MATERIAL]->(po:PurchaseOrder)
WHERE doc.approval_status IN ['Approved_Late', 'Pending']
  AND po.delivery_status = 'Late'
  AND abs(duration.between(doc.planned_appr_date, po.planned_deliv_dt).days) <= 60
RETURN wp.id              AS work_package,
       a.id               AS activity,
       doc.id             AS late_document,
       doc.late_days      AS doc_delay_days,
       po.id              AS late_po,
       po.delay_days      AS po_delay_days,
       doc.planned_appr_date AS doc_appr_date,
       po.planned_deliv_dt   AS po_deliv_date;


// ── Q6: Delay propagation — did A057's delay cascade to A060? ──────────────
// "Trace the dependency chain: was A060's delay caused by A057's delay?"
// (Ground truth chain CC-05)

MATCH (e:Event)-[:IMPACTS_ACTIVITY]->(a1:Activity)
WHERE a1.id = 'A057' AND e.triggers_delay_days > 0
WITH a1, e
MATCH (a2:Activity)
WHERE a2.id = 'A060'
  AND a2.delayed = true
  AND a2.actual_start > a2.planned_start
RETURN e.id           AS root_event,
       e.event_date   AS event_date,
       a1.id          AS upstream_activity,
       a1.delayed     AS upstream_delayed,
       a2.id          AS downstream_activity,
       a2.delayed     AS downstream_delayed,
       duration.between(e.event_date, a2.actual_start).days AS days_to_cascade,
       'Verify: CC-05 ground truth chain' AS note;
