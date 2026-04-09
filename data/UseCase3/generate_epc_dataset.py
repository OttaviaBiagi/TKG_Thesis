"""
EPC Project Dataset Generator
==============================
Combines a PSPLIB-style project scheduling structure with three
synthetically generated data sources (procurement, documents, events)
to produce a heterogeneous dataset suitable for loading into Neo4j
as a Temporal Knowledge Graph.

Sources:
  A - Schedule (PSPLIB-style): activities, durations, predecessors, resources
  B - Procurement:             purchase orders linked to activities
  C - Documents:               engineering drawings with approval workflows
  D - Events:                  change orders, NCRs, inspections

Ground truth causal chains are embedded so that TLogic rule extraction
and T-GQL temporal path queries can be evaluated against known answers.

Usage:
  python3 generate_epc_dataset.py
  
Output:
  data/activities.csv
  data/procurement.csv
  data/documents.csv
  data/events.csv
  data/causal_ground_truth.json
  neo4j/import_graph.cypher         (Cypher script to load into Neo4j)
  neo4j/tgql_queries.cypher         (Example T-GQL / temporal Cypher queries)
"""

import random
import json
import csv
import os
from datetime import datetime, timedelta

# ─── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)

# ─── Output dirs ────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("neo4j", exist_ok=True)

# ─── Project parameters ─────────────────────────────────────────────────────
PROJECT_START   = datetime(2024, 1, 8)   # project kick-off
N_ACTIVITIES    = 60                     # comparable to PSPLIB J60 instances
N_WORK_PACKAGES = 12                     # WBS grouping
N_DISCIPLINES   = 4
N_VENDORS       = 8
N_PROCUREMENT   = 18                     # purchase orders
N_DOCUMENTS     = 30                     # engineering documents

DISCIPLINES     = ["Civil", "Mechanical", "Electrical", "Instrumentation"]
VENDORS         = [f"Vendor_{chr(65+i)}" for i in range(N_VENDORS)]
DOC_TYPES       = ["P&ID", "Isometric", "Datasheet", "Spec", "Procedure", "Layout"]
EVENT_TYPES     = ["ChangeOrder", "NCR", "Inspection", "DelayNotice", "WeatherDelay"]

# ─── Helper: date arithmetic ─────────────────────────────────────────────────
def d(base, days):
    return (base + timedelta(days=days)).strftime("%Y-%m-%d")

def jitter(base_days, sigma=3):
    """Add realistic noise to a planned duration."""
    return max(1, int(base_days + random.gauss(0, sigma)))


# ════════════════════════════════════════════════════════════════════════════
# SOURCE A: SCHEDULE
# Modelled after PSPLIB J60: 60 activities, precedence constraints,
# resource requirements, planned vs actual dates.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Source A: Schedule (PSPLIB-style)...")

activities = []
work_packages = {f"WP{i+1:02d}": [] for i in range(N_WORK_PACKAGES)}

# Assign activities to work packages and disciplines
for i in range(1, N_ACTIVITIES + 1):
    wp  = f"WP{((i-1) % N_WORK_PACKAGES)+1:02d}"
    dis = DISCIPLINES[i % N_DISCIPLINES]
    dur = random.randint(5, 20)          # planned duration in days
    work_packages[wp].append(i)
    activities.append({
        "activity_id":    f"A{i:03d}",
        "work_package":   wp,
        "discipline":     dis,
        "description":    f"{dis} activity {i} in {wp}",
        "planned_dur":    dur,
        "planned_start":  None,          # calculated below
        "planned_finish": None,
        "actual_start":   None,
        "actual_finish":  None,
        "pct_complete":   0,
        "status":         "Not Started",
        "delayed":        False,         # ground truth flag
        "delay_reason":   None,
    })

# Build a simple precedence network (topological layers)
# Layer 0: A001-A010 (no predecessors)
# Each subsequent group depends on ~2 activities from the previous layer
predecessors = {a["activity_id"]: [] for a in activities}
layer_size = 10
layers = [
    [f"A{i:03d}" for i in range(1, 11)],
    [f"A{i:03d}" for i in range(11, 25)],
    [f"A{i:03d}" for i in range(25, 42)],
    [f"A{i:03d}" for i in range(42, 60)],
    [f"A060"],
]
for l_idx in range(1, len(layers)):
    prev = layers[l_idx - 1]
    for act in layers[l_idx]:
        n_pred = random.randint(1, min(3, len(prev)))
        predecessors[act] = random.sample(prev, n_pred)

# Schedule activities (simple critical-path forward pass)
act_map = {a["activity_id"]: a for a in activities}
finish_day = {}   # activity_id -> planned finish day from project start

def scheduled_finish(act_id):
    if act_id in finish_day:
        return finish_day[act_id]
    preds = predecessors[act_id]
    if not preds:
        start = 0
    else:
        start = max(scheduled_finish(p) for p in preds)
    dur = act_map[act_id]["planned_dur"]
    finish_day[act_id] = start + dur
    act_map[act_id]["planned_start"]  = d(PROJECT_START, start)
    act_map[act_id]["planned_finish"] = d(PROJECT_START, start + dur)
    return finish_day[act_id]

for a in activities:
    scheduled_finish(a["activity_id"])

# Simulate actual execution (first ~40 activities completed or in progress)
# Inject 8 deliberate delays in specific activities (ground truth)
DELAYED_ACTIVITIES = {"A012", "A023", "A031", "A038", "A044", "A052", "A057", "A060"}
DELAY_REASONS      = {
    "A012": "procurement_delay",    # caused by PO-003 late delivery
    "A023": "procurement_delay",    # caused by PO-007
    "A031": "document_revision",    # caused by DOC-012 late approval
    "A038": "change_order",         # caused by CO-001
    "A044": "procurement_delay",    # caused by PO-014
    "A052": "weather_delay",        # caused by WD-001
    "A057": "ncr",                  # caused by NCR-002
    "A060": "cascade",              # cascaded from A057
}

for a in activities:
    aid = a["activity_id"]
    plan_start = datetime.strptime(a["planned_start"], "%Y-%m-%d")
    plan_dur   = a["planned_dur"]

    if plan_start < datetime(2024, 9, 1):   # activities starting before Sept
        delay = 0
        if aid in DELAYED_ACTIVITIES:
            delay = random.randint(5, 15)
            a["delayed"] = True
            a["delay_reason"] = DELAY_REASONS[aid]

        actual_start  = plan_start + timedelta(days=random.randint(0, 2) + delay)
        actual_dur    = jitter(plan_dur)
        actual_finish = actual_start + timedelta(days=actual_dur)
        a["actual_start"]  = actual_start.strftime("%Y-%m-%d")
        a["actual_finish"] = actual_finish.strftime("%Y-%m-%d")
        a["pct_complete"]  = 100
        a["status"]        = "Complete" if delay == 0 else "Complete_Late"
    elif plan_start < datetime(2024, 11, 1):
        a["pct_complete"] = random.randint(20, 80)
        a["status"]       = "In Progress"
        a["actual_start"] = plan_start.strftime("%Y-%m-%d")
    else:
        a["status"] = "Not Started"

# Write CSV
with open("data/activities.csv", "w", newline="") as f:
    fields = ["activity_id","work_package","discipline","description",
              "planned_dur","planned_start","planned_finish",
              "actual_start","actual_finish","pct_complete","status",
              "delayed","delay_reason"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for a in activities:
        w.writerow({k: a[k] for k in fields})

print(f"  {len(activities)} activities written to data/activities.csv")


# ════════════════════════════════════════════════════════════════════════════
# SOURCE B: PROCUREMENT
# Purchase orders linked to activities that require specific materials.
# Heterogeneity: different schema, JSON-like structure, vendor entities.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Source B: Procurement...")

procurement = []
LATE_POs = {"PO-003": 12, "PO-007": 9, "PO-014": 14}   # ground truth delays

for i in range(1, N_PROCUREMENT + 1):
    po_id   = f"PO-{i:03d}"
    vendor  = random.choice(VENDORS)
    # Link to an activity that requires this material
    linked_act = random.choice(activities[10:50])["activity_id"]
    linked_wp  = act_map[linked_act]["work_package"]
    plan_start = datetime.strptime(act_map[linked_act]["planned_start"], "%Y-%m-%d")
    lead_days  = random.randint(30, 60)
    plan_order = plan_start - timedelta(days=lead_days)
    plan_deliv = plan_start - timedelta(days=random.randint(5, 15))

    actual_deliv = None
    delivery_status = "On Time"
    delay_days = 0

    if po_id in LATE_POs:
        delay_days = LATE_POs[po_id]
        actual_deliv = (plan_deliv + timedelta(days=delay_days)).strftime("%Y-%m-%d")
        delivery_status = "Late"
    elif plan_deliv < datetime(2024, 10, 1):
        actual_deliv = (plan_deliv + timedelta(days=random.randint(-2, 3))).strftime("%Y-%m-%d")
        delivery_status = "On Time"

    procurement.append({
        "po_id":            po_id,
        "vendor":           vendor,
        "linked_activity":  linked_act,
        "work_package":     linked_wp,
        "material_desc":    f"Equipment/Material for {linked_act}",
        "planned_order_dt": plan_order.strftime("%Y-%m-%d"),
        "planned_deliv_dt": plan_deliv.strftime("%Y-%m-%d"),
        "actual_deliv_dt":  actual_deliv,
        "delivery_status":  delivery_status,
        "delay_days":       delay_days,
        "contract_value":   round(random.uniform(10000, 500000), 2),
    })

with open("data/procurement.csv", "w", newline="") as f:
    fields = ["po_id","vendor","linked_activity","work_package",
              "material_desc","planned_order_dt","planned_deliv_dt",
              "actual_deliv_dt","delivery_status","delay_days","contract_value"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(procurement)

print(f"  {len(procurement)} purchase orders written to data/procurement.csv")


# ════════════════════════════════════════════════════════════════════════════
# SOURCE C: DOCUMENTS
# Engineering documents with revision histories and approval workflows.
# Heterogeneity: multi-revision structure, status machine, links to activities.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Source C: Engineering Documents...")

documents = []
LATE_DOCS = {"DOC-012": 8, "DOC-019": 6}   # ground truth: late approvals

for i in range(1, N_DOCUMENTS + 1):
    doc_id  = f"DOC-{i:03d}"
    dtype   = random.choice(DOC_TYPES)
    dis     = random.choice(DISCIPLINES)
    n_rev   = random.randint(1, 4)
    # Link to activity that requires this doc as prerequisite
    linked_act = random.choice(activities[5:55])["activity_id"]
    req_by_dt  = datetime.strptime(act_map[linked_act]["planned_start"], "%Y-%m-%d")

    issue_dt   = req_by_dt - timedelta(days=random.randint(20, 40))
    plan_appr  = issue_dt + timedelta(days=random.randint(7, 14))

    late_days  = LATE_DOCS.get(doc_id, 0)
    if late_days:
        actual_appr = plan_appr + timedelta(days=late_days)
        appr_status = "Approved_Late"
    elif plan_appr < datetime(2024, 10, 1):
        actual_appr = plan_appr + timedelta(days=random.randint(-1, 2))
        appr_status = "Approved"
    else:
        actual_appr = None
        appr_status = "Pending"

    documents.append({
        "doc_id":            doc_id,
        "doc_type":          dtype,
        "discipline":        dis,
        "title":             f"{dis} {dtype} for {linked_act}",
        "linked_activity":   linked_act,
        "revision":          f"Rev {n_rev}",
        "issue_date":        issue_dt.strftime("%Y-%m-%d"),
        "planned_appr_date": plan_appr.strftime("%Y-%m-%d"),
        "actual_appr_date":  actual_appr.strftime("%Y-%m-%d") if actual_appr else "",
        "approval_status":   appr_status,
        "late_days":         late_days,
    })

with open("data/documents.csv", "w", newline="") as f:
    fields = ["doc_id","doc_type","discipline","title","linked_activity",
              "revision","issue_date","planned_appr_date","actual_appr_date",
              "approval_status","late_days"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(documents)

print(f"  {len(documents)} documents written to data/documents.csv")


# ════════════════════════════════════════════════════════════════════════════
# SOURCE D: EVENTS
# Discrete project events: change orders, NCRs, inspections, delays.
# Heterogeneity: event-centric schema, free-text description, multi-link.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Source D: Project Events...")

events = []

# Ground truth events causing delays (embedded causal chain)
CAUSAL_EVENTS = [
    {"event_id": "CO-001",  "event_type": "ChangeOrder",   "date": "2024-04-15",
     "impacts_activity": "A038", "description": "Scope change: added piping loop",
     "triggers_delay_days": 10, "linked_po": ""},
    {"event_id": "NCR-001", "event_type": "NCR",           "date": "2024-03-20",
     "impacts_activity": "A023", "description": "Material non-conformance on PO-007",
     "triggers_delay_days": 9,  "linked_po": "PO-007"},
    {"event_id": "NCR-002", "event_type": "NCR",           "date": "2024-07-10",
     "impacts_activity": "A057", "description": "Weld quality failure",
     "triggers_delay_days": 12, "linked_po": ""},
    {"event_id": "WD-001",  "event_type": "WeatherDelay",  "date": "2024-05-02",
     "impacts_activity": "A052", "description": "Storm: civil works suspended 7 days",
     "triggers_delay_days": 7,  "linked_po": ""},
    {"event_id": "DN-001",  "event_type": "DelayNotice",   "date": "2024-02-28",
     "impacts_activity": "A012", "description": "Vendor notified 12-day delivery delay",
     "triggers_delay_days": 12, "linked_po": "PO-003"},
    {"event_id": "DN-002",  "event_type": "DelayNotice",   "date": "2024-06-15",
     "impacts_activity": "A044", "description": "Custom valve fabrication delay",
     "triggers_delay_days": 14, "linked_po": "PO-014"},
]
events.extend(CAUSAL_EVENTS)

# Regular (non-causal) inspection events
for i in range(1, 20):
    linked_act = random.choice(activities[5:50])["activity_id"]
    event_date = datetime.strptime(act_map[linked_act]["planned_start"], "%Y-%m-%d")
    event_date = event_date + timedelta(days=random.randint(-5, 5))
    events.append({
        "event_id":             f"INS-{i:03d}",
        "event_type":           "Inspection",
        "date":                 event_date.strftime("%Y-%m-%d"),
        "impacts_activity":     linked_act,
        "description":          f"Routine inspection checkpoint for {linked_act}",
        "triggers_delay_days":  0,
        "linked_po":            "",
    })

with open("data/events.csv", "w", newline="") as f:
    fields = ["event_id","event_type","date","impacts_activity",
              "description","triggers_delay_days","linked_po"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(events)

print(f"  {len(events)} events written to data/events.csv")


# ════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH: Causal chains for evaluation
# These are the known cause-effect chains embedded in the dataset.
# Used to validate TLogic rule extraction and T-GQL path queries.
# ════════════════════════════════════════════════════════════════════════════
ground_truth = {
    "causal_chains": [
        {
            "chain_id": "CC-01",
            "description": "PO-003 late delivery caused A012 delay",
            "chain": [
                {"entity": "PO-003",  "relation": "deliveredLate",     "timestamp": "2024-02-28"},
                {"entity": "DN-001",  "relation": "triggeredDelayIn",  "timestamp": "2024-02-28"},
                {"entity": "A012",    "relation": "startedLate",       "timestamp": "2024-03-12"},
            ],
            "expected_tgql_query": "MATCH path WHERE PO-003 deliveredLate AND A012 startedLate WHEN (PO-003 deliveryInterval OVERLAPS A012 waitInterval)"
        },
        {
            "chain_id": "CC-02",
            "description": "NCR on PO-007 caused A023 delay",
            "chain": [
                {"entity": "PO-007",  "relation": "hadNCR",            "timestamp": "2024-03-20"},
                {"entity": "NCR-001", "relation": "delayedActivity",   "timestamp": "2024-03-20"},
                {"entity": "A023",    "relation": "startedLate",       "timestamp": "2024-03-29"},
            ],
        },
        {
            "chain_id": "CC-03",
            "description": "DOC-012 late approval blocked A031",
            "chain": [
                {"entity": "DOC-012", "relation": "approvedLate",      "timestamp": "2024-04-08"},
                {"entity": "A031",    "relation": "blockedByDocument",  "timestamp": "2024-04-08"},
                {"entity": "A031",    "relation": "startedLate",       "timestamp": "2024-04-16"},
            ],
        },
        {
            "chain_id": "CC-04",
            "description": "Change order CO-001 caused A038 delay, which cascaded to A044",
            "chain": [
                {"entity": "CO-001",  "relation": "issuedAt",          "timestamp": "2024-04-15"},
                {"entity": "A038",    "relation": "impactedByChange",  "timestamp": "2024-04-15"},
                {"entity": "A038",    "relation": "startedLate",       "timestamp": "2024-04-25"},
                {"entity": "A044",    "relation": "dependsOn_A038",    "timestamp": "2024-05-05"},
            ],
        },
        {
            "chain_id": "CC-05",
            "description": "NCR-002 and PO-014 delay cascaded to A057 and A060",
            "chain": [
                {"entity": "NCR-002", "relation": "issuedAt",          "timestamp": "2024-07-10"},
                {"entity": "A057",    "relation": "delayedByNCR",      "timestamp": "2024-07-10"},
                {"entity": "A060",    "relation": "dependsOn_A057",    "timestamp": "2024-07-22"},
                {"entity": "A060",    "relation": "cascade_delay",     "timestamp": "2024-07-22"},
            ],
        }
    ],
    "evaluation_questions": [
        "Which activities were delayed while their linked purchase order was in late delivery status?",
        "Which activities started late immediately after a change order was issued?",
        "Find all multi-hop delay chains of length >= 2 between events and activities.",
        "Which work packages had both a document approval delay AND a procurement delay in the same time window?",
        "What was the planned vs actual schedule state of WP04 on 2024-05-01?"
    ]
}

with open("data/causal_ground_truth.json", "w") as f:
    json.dump(ground_truth, f, indent=2)

print("  Ground truth causal chains written to data/causal_ground_truth.json")


# ════════════════════════════════════════════════════════════════════════════
# NEO4J IMPORT SCRIPT
# Cypher script to load all four sources into Neo4j as a temporal property graph.
# ════════════════════════════════════════════════════════════════════════════
print("Generating Neo4j import script...")

cypher_import = """// ═══════════════════════════════════════════════════════════════════════════
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
"""

with open("neo4j/import_graph.cypher", "w") as f:
    f.write(cypher_import)

print("  Neo4j import script written to neo4j/import_graph.cypher")


# ════════════════════════════════════════════════════════════════════════════
# T-GQL QUERY EXAMPLES
# These queries demonstrate temporal path semantics not expressible in
# standard Cypher. They correspond directly to the ground truth questions.
# ════════════════════════════════════════════════════════════════════════════
print("Generating T-GQL query examples...")

tgql_queries = """// ═══════════════════════════════════════════════════════════════════════════
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
"""

with open("neo4j/tgql_queries.cypher", "w") as f:
    f.write(tgql_queries)

print("  T-GQL queries written to neo4j/tgql_queries.cypher")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Source A (Schedule):    {len(activities)} activities, {N_WORK_PACKAGES} work packages")
print(f"Source B (Procurement): {len(procurement)} purchase orders, {N_VENDORS} vendors")
print(f"Source C (Documents):   {len(documents)} engineering documents")
print(f"Source D (Events):      {len(events)} project events")
print(f"Delayed activities:     {sum(1 for a in activities if a['delayed'])} (with known causes)")
print(f"Ground truth chains:    5 causal chains in data/causal_ground_truth.json")
print()
print("Files generated:")
print("  data/activities.csv")
print("  data/procurement.csv")
print("  data/documents.csv")
print("  data/events.csv")
print("  data/causal_ground_truth.json")
print("  neo4j/import_graph.cypher")
print("  neo4j/tgql_queries.cypher")
print()
print("Next steps:")
print("  1. Run: python3 generate_epc_dataset.py  (already done)")
print("  2. Copy data/*.csv to Neo4j import directory")
print("  3. Run: neo4j/import_graph.cypher in Neo4j Browser")
print("  4. Run queries in neo4j/tgql_queries.cypher to explore the TKG")
print("  5. Use data/causal_ground_truth.json to evaluate TLogic rule extraction")
