"""
populate_onto.py — Load epc_dataset_real.json into an rdflib Graph
following the OWL-2 ontology defined in epc_tkg.ttl.

Produces: ontology/epc_instance_data.ttl  (OWL-2 individuals)

Run from repo root:
    python ontology/populate_onto.py

Dataset top-level keys:
    project, activities, families, steps, activity_steps, step_sequences,
    work_permits, certifications, workers, update_events, metadata

Steps are a flat top-level list (not nested inside activities).
Worker assignments do not exist in the raw data — we generate them
synthetically by assigning hot_work-certified workers to hot_work steps
after the rule-change date, creating realistic compliance violations.
"""
import json
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ── Namespaces ────────────────────────────────────────────────────────────────
EPC  = Namespace("http://tecnicasreunidas.es/ontology/epc#")
BASE = "http://tecnicasreunidas.es/ontology/epc/data#"

DATA_FILE  = Path("data/UseCase4/epc_dataset_real.json")
ONTO_FILE  = Path("ontology/epc_tkg.ttl")
OUT_FILE   = Path("ontology/epc_instance_data.ttl")

FAR_FUTURE    = "9999-12-31T23:59:59+00:00"
PROJECT_START = "2024-01-01T00:00:00+00:00"
RULE_CHANGE   = "2024-06-29T00:00:00+00:00"

# ── Helpers ───────────────────────────────────────────────────────────────────
def uri(local: str) -> URIRef:
    safe = local.replace(" ", "_").replace("/", "_").replace(":", "_")
    return URIRef(BASE + safe)

def dt_lit(iso_str: str) -> Literal:
    return Literal(iso_str or FAR_FUTURE, datatype=XSD.dateTime)

def add_bitemporal(g, node, valid_from, valid_to, tx_time=None):
    g.add((node, EPC.validFrom, dt_lit(valid_from)))
    g.add((node, EPC.validTo,   dt_lit(valid_to or FAR_FUTURE)))
    if tx_time:
        g.add((node, EPC.txTime, dt_lit(tx_time)))

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = json.loads(DATA_FILE.read_text(encoding="utf-8"))

# ── Build RDF graph ───────────────────────────────────────────────────────────
g = Graph()
g.bind("epc",  EPC)
g.bind("owl",  OWL)
g.bind("rdfs", RDFS)
g.bind("xsd",  XSD)

g.parse(str(ONTO_FILE), format="turtle")
print("Schema loaded.")

# ── Project ───────────────────────────────────────────────────────────────────
p = dataset["project"]
proj = uri(f"project_{p['id']}")
g.add((proj, RDF.type,  EPC.Project))
g.add((proj, EPC.id,    Literal(p["id"])))
g.add((proj, EPC.name,  Literal(p["name"])))
print(f"  Project: {p['name']}")

# ── Families ──────────────────────────────────────────────────────────────────
for f in dataset.get("families", []):
    fam = uri(f"family_{f['id']}")
    g.add((fam, RDF.type, EPC.Family))
    g.add((fam, EPC.id,   Literal(f["id"])))
    g.add((fam, EPC.name, Literal(f.get("name", f["id"]))))
print(f"  Families: {len(dataset.get('families', []))}")

# ── Work Permits ──────────────────────────────────────────────────────────────
for wp in dataset.get("work_permits", []):
    wpu = uri(f"permit_{wp['id']}")
    g.add((wpu, RDF.type, EPC.WorkPermit))
    g.add((wpu, EPC.id,   Literal(wp["id"])))
    g.add((wpu, EPC.name, Literal(wp.get("name", wp["id"]))))

# ── Certifications ────────────────────────────────────────────────────────────
for cert in dataset.get("certifications", []):
    certu = uri(f"cert_{cert['id']}")
    g.add((certu, RDF.type, EPC.Certification))
    g.add((certu, EPC.id,   Literal(cert["id"])))
    g.add((certu, EPC.name, Literal(cert.get("name", cert["id"]))))

print(f"  Permits: {len(dataset.get('work_permits', []))}  "
      f"Certifications: {len(dataset.get('certifications', []))}")

# ── Permit-Cert requirements (bitemporal — rule change 2024-06-29) ─────────────
# Before rule change: 3 certs required for hot_work
# After rule change:  4 certs required (Advanced_Fire_Watch added)
CERT_REQS_BEFORE = {
    "hot_work":       ["Hot_Work_Safety", "Fire_Watch", "Welding_Certification"],
    "excavation":     ["Excavation_Safety", "Confined_Space_Entry", "Soil_Assessment"],
    "lifting":        ["Rigging_&_Lifting", "Crane_Operator", "Slinging_Certificate"],
    "electrical":     ["Electrical_Safety", "LOTO_Certification", "HV_Awareness"],
    "confined_space": ["Confined_Space_Entry", "Gas_Testing", "Emergency_Response"],
    "radiography":    ["NDT_Level_II", "Radiation_Safety", "RT_Operator"],
    "work_at_height": ["Working_at_Height", "Scaffold_Inspection", "Fall_Arrest"],
    "general_work":   ["General_Safety_Induction", "Site_Orientation"],
}
CERT_REQS_AFTER = {
    "hot_work": ["Hot_Work_Safety", "Fire_Watch", "Welding_Certification",
                 "Advanced_Fire_Watch"],
}

req_count = 0
for permit_id, certs in CERT_REQS_BEFORE.items():
    for cert_id in certs:
        req_node = uri(f"pcr_{permit_id}_{cert_id}_v1")
        g.add((req_node, RDF.type,                 EPC.PermitCertRequirement))
        g.add((req_node, EPC.forPermit,             uri(f"permit_{permit_id}")))
        g.add((req_node, EPC.requiresCertification, uri(f"cert_{cert_id}")))
        add_bitemporal(g, req_node,
                       valid_from=PROJECT_START,
                       valid_to="2024-06-28T23:59:59+00:00",
                       tx_time=PROJECT_START)
        req_count += 1

for permit_id, certs in CERT_REQS_AFTER.items():
    for cert_id in certs:
        req_node = uri(f"pcr_{permit_id}_{cert_id}_v2")
        g.add((req_node, RDF.type,                 EPC.PermitCertRequirement))
        g.add((req_node, EPC.forPermit,             uri(f"permit_{permit_id}")))
        g.add((req_node, EPC.requiresCertification, uri(f"cert_{cert_id}")))
        add_bitemporal(g, req_node,
                       valid_from=RULE_CHANGE,
                       valid_to=None,
                       tx_time=RULE_CHANGE)
        req_count += 1

print(f"  PermitCertRequirement nodes: {req_count}")

# ── Workers + CertificationHoldings (bitemporal) ──────────────────────────────
# Worker certifications stored as: {"cert": "Hot Work Safety", "valid_from": ..., ...}
workers = dataset.get("workers", [])
holding_count = 0
for w in workers:
    wu = uri(f"worker_{w['id']}")
    g.add((wu, RDF.type,       EPC.Worker))
    g.add((wu, EPC.id,         Literal(w["id"])))
    g.add((wu, EPC.name,       Literal(w.get("name", w["id"]))))
    g.add((wu, EPC.discipline, Literal(w.get("discipline", ""))))

    for cert_info in w.get("certifications", []):
        cert_name = cert_info["cert"]
        cert_id   = cert_name.replace(" ", "_")
        holding   = uri(f"holding_{w['id']}_{cert_id}")
        g.add((holding, RDF.type,                EPC.CertificationHolding))
        g.add((holding, EPC.byWorker,            wu))
        g.add((holding, EPC.holdsCertification,  uri(f"cert_{cert_id}")))
        add_bitemporal(g, holding,
                       valid_from=cert_info.get("valid_from", PROJECT_START),
                       valid_to=cert_info.get("valid_to"),
                       tx_time=cert_info.get("tx_time", PROJECT_START))
        holding_count += 1

print(f"  Workers: {len(workers)}  CertificationHoldings: {holding_count}")

# ── Activities (subset: first 100) ────────────────────────────────────────────
activities = dataset.get("activities", [])[:100]
for act in activities:
    actu = uri(f"activity_{act['id']}")
    g.add((actu, RDF.type,      EPC.Activity))
    g.add((actu, EPC.id,        Literal(act["id"])))
    g.add((actu, EPC.name,      Literal(act.get("name", act["id"]))))
    g.add((actu, EPC.belongsTo, uri(f"family_{act.get('family', 'unknown')}")))
    g.add((proj, EPC.includes,  actu))
print(f"  Activities loaded: {len(activities)}")

# ── Steps (flat top-level list — subset: first 200) ───────────────────────────
# Steps have: id, name, order, weight_pct, permit_type, activity_id, valid_from, valid_to
raw_steps = dataset.get("steps", [])[:200]
step_uris = {}   # step_id -> step_uri  (for precedes chain)
step_by_activity = {}  # activity_id -> [steps in order]

for step in raw_steps:
    su = uri(f"step_{step['id']}")
    step_uris[step["id"]] = su
    g.add((su, RDF.type,      EPC.Step))
    g.add((su, EPC.id,        Literal(step["id"])))
    g.add((su, EPC.name,      Literal(step.get("name", step["id"]))))
    g.add((su, EPC.order,     Literal(step.get("order", 0), datatype=XSD.integer)))
    g.add((su, EPC.weightPct, Literal(step.get("weight_pct", 0.0), datatype=XSD.decimal)))

    permit_type = step.get("permit_type")
    if permit_type:
        g.add((su, EPC.requiresPermit, uri(f"permit_{permit_type}")))
        g.add((su, EPC.permitType,     Literal(permit_type)))

    add_bitemporal(g, su,
                   valid_from=step.get("valid_from", PROJECT_START),
                   valid_to=step.get("valid_to"),
                   tx_time=step.get("tx_time", PROJECT_START))

    act_id = step.get("activity_id")
    if act_id:
        actu = uri(f"activity_{act_id}")
        g.add((actu, EPC.hasStep, su))
        step_by_activity.setdefault(act_id, []).append((step.get("order", 0), su))

# Add epc:precedes chains within each activity (by step order)
prec_count = 0
for act_id, ordered_steps in step_by_activity.items():
    ordered_steps.sort(key=lambda x: x[0])
    for i in range(len(ordered_steps) - 1):
        g.add((ordered_steps[i][1], EPC.precedes, ordered_steps[i+1][1]))
        prec_count += 1

print(f"  Steps: {len(raw_steps)}  Precedes: {prec_count}")

# ── Worker Assignments (synthetic, bitemporal) ────────────────────────────────
# The raw dataset has no explicit worker-step assignments.
# We generate realistic ones: assign each hot_work-certified worker
# to hot_work steps that overlap with the worker's certification period.
# This creates the set of post-rule-change violations detectable by Q6.

hot_steps_post = [s for s in dataset.get("steps", [])
                  if s.get("permit_type") == "hot_work"
                  and (s.get("valid_from") or "") >= RULE_CHANGE][:50]

assign_count = 0
for w in workers:
    certs = {c["cert"].replace(" ", "_") for c in w.get("certifications", [])}
    has_hot_work_base = {"Hot_Work_Safety", "Fire_Watch", "Welding_Certification"}.issubset(certs)
    if not has_hot_work_base:
        continue

    # Assign this worker to the first 5 post-rule-change hot_work steps
    for step in hot_steps_post[:5]:
        step_id    = step["id"]
        valid_from = step.get("valid_from", RULE_CHANGE)
        valid_to   = step.get("valid_to", valid_from)

        assign = uri(f"assign_{w['id']}_{step_id}")
        g.add((assign, RDF.type,           EPC.WorkerAssignment))
        g.add((assign, EPC.assignedWorker, uri(f"worker_{w['id']}")))
        g.add((assign, EPC.assignedToStep, uri(f"step_{step_id}")))
        add_bitemporal(g, assign,
                       valid_from=valid_from,
                       valid_to=valid_to,
                       tx_time=valid_from)

        # Ensure the step node exists (it may be outside the 200-step subset)
        su = uri(f"step_{step_id}")
        if (su, RDF.type, EPC.Step) not in g:
            g.add((su, RDF.type,            EPC.Step))
            g.add((su, EPC.id,              Literal(step_id)))
            g.add((su, EPC.name,            Literal(step.get("name", step_id))))
            g.add((su, EPC.requiresPermit,  uri("permit_hot_work")))
            g.add((su, EPC.permitType,      Literal("hot_work")))
            add_bitemporal(g, su,
                           valid_from=valid_from,
                           valid_to=valid_to,
                           tx_time=step.get("tx_time", valid_from))

        assign_count += 1

print(f"  WorkerAssignments (synthetic): {assign_count}")

# ── Serialize ─────────────────────────────────────────────────────────────────
g.serialize(str(OUT_FILE), format="turtle")
total = len(g)
print(f"\nSaved {OUT_FILE}  ({total:,} triples)")
