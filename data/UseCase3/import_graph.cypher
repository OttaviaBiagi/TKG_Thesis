// ═══════════════════════════════════════════════════════════════════════════
// EPC Temporal Knowledge Graph: Neo4j Import Script
// ═══════════════════════════════════════════════════════════════════════════
// Run in Neo4j Browser or via cypher-shell:
//   cypher-shell -u neo4j -p <password> -f neo4j/import_graph.cypher
//
// CSV files must be in Neo4j import directory (neo4j/import/ or $NEO4J_HOME/import/)
// Adjust file paths if needed.
// ═══════════════════════════════════════════════════════════════════════════

// ── Constraints (run once) ──────────────────────────────────────────────────
CREATE CONSTRAINT act_id  IF NOT EXISTS FOR (a:Activity)      REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT wp_id   IF NOT EXISTS FOR (w:WorkPackage)   REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT po_id   IF NOT EXISTS FOR (p:PurchaseOrder) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT doc_id  IF NOT EXISTS FOR (d:Document)      REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT evt_id  IF NOT EXISTS FOR (e:Event)         REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT vnd_id  IF NOT EXISTS FOR (v:Vendor)        REQUIRE v.name IS UNIQUE;


// ── SOURCE A: Activities ────────────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///activities.csv' AS row
MERGE (wp:WorkPackage {id: row.work_package})
  SET wp.label = row.work_package
MERGE (a:Activity {id: row.activity_id})
  SET a.description     = row.description,
      a.discipline      = row.discipline,
      a.planned_dur     = toInteger(row.planned_dur),
      a.planned_start   = date(row.planned_start),
      a.planned_finish  = date(row.planned_finish),
      a.actual_start    = CASE WHEN row.actual_start  <> '' THEN date(row.actual_start)  ELSE null END,
      a.actual_finish   = CASE WHEN row.actual_finish <> '' THEN date(row.actual_finish) ELSE null END,
      a.pct_complete    = toInteger(row.pct_complete),
      a.status          = row.status,
      a.delayed         = (row.delayed = 'True'),
      a.delay_reason    = row.delay_reason
MERGE (a)-[:BELONGS_TO {from: date(row.planned_start), to: date(row.planned_finish)}]->(wp);


// ── SOURCE A: Precedence edges (temporal) ──────────────────────────────────
// Predecessors are embedded as a second pass using the known structure.
// In a real system these come from the scheduling tool export.
// Here we reconstruct them from the layer-based structure.
// (Simplified: you can extend this with the full predecessors dict.)


// ── SOURCE B: Procurement ───────────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///procurement.csv' AS row
MERGE (v:Vendor {name: row.vendor})
MERGE (po:PurchaseOrder {id: row.po_id})
  SET po.material_desc     = row.material_desc,
      po.work_package      = row.work_package,
      po.planned_order_dt  = date(row.planned_order_dt),
      po.planned_deliv_dt  = date(row.planned_deliv_dt),
      po.actual_deliv_dt   = CASE WHEN row.actual_deliv_dt <> '' THEN date(row.actual_deliv_dt) ELSE null END,
      po.delivery_status   = row.delivery_status,
      po.delay_days        = toInteger(row.delay_days),
      po.contract_value    = toFloat(row.contract_value)
MERGE (po)-[:SUPPLIED_BY {from: date(row.planned_order_dt)}]->(v)
WITH po, row
MATCH (a:Activity {id: row.linked_activity})
MERGE (a)-[:REQUIRES_MATERIAL {
    planned_by: date(row.planned_deliv_dt)
}]->(po);


// ── SOURCE C: Documents ─────────────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///documents.csv' AS row
MERGE (doc:Document {id: row.doc_id})
  SET doc.doc_type          = row.doc_type,
      doc.discipline        = row.discipline,
      doc.title             = row.title,
      doc.revision          = row.revision,
      doc.issue_date        = date(row.issue_date),
      doc.planned_appr_date = date(row.planned_appr_date),
      doc.actual_appr_date  = CASE WHEN row.actual_appr_date <> '' THEN date(row.actual_appr_date) ELSE null END,
      doc.approval_status   = row.approval_status,
      doc.late_days         = toInteger(row.late_days)
WITH doc, row
MATCH (a:Activity {id: row.linked_activity})
MERGE (a)-[:REQUIRES_DOCUMENT {
    required_by: date(row.planned_appr_date)
}]->(doc);


// ── SOURCE D: Events ────────────────────────────────────────────────────────
LOAD CSV WITH HEADERS FROM 'file:///events.csv' AS row
MERGE (e:Event {id: row.event_id})
  SET e.event_type          = row.event_type,
      e.event_date          = date(row.date),
      e.description         = row.description,
      e.triggers_delay_days = toInteger(row.triggers_delay_days)
WITH e, row
MATCH (a:Activity {id: row.impacts_activity})
MERGE (e)-[:IMPACTS_ACTIVITY {
    at: date(row.date),
    delay_days: toInteger(row.triggers_delay_days)
}]->(a)
WITH e, row
WHERE row.linked_po <> ''
MATCH (po:PurchaseOrder {id: row.linked_po})
MERGE (e)-[:RELATED_TO_PO {at: date(row.date)}]->(po);


// ── Verify load ─────────────────────────────────────────────────────────────
MATCH (n) RETURN labels(n)[0] AS NodeType, count(n) AS Count ORDER BY Count DESC;
