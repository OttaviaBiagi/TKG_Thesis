// ═══════════════════════════════════════════════════════════════
// UseCase3 — EPC TKG Import from TR Family Steps real data
// Run after generate_epc_dataset.py
// ═══════════════════════════════════════════════════════════════

// 1. Constraints
CREATE CONSTRAINT activity_id IF NOT EXISTS FOR (a:Activity) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT family_id IF NOT EXISTS FOR (f:Family) REQUIRE f.id IS UNIQUE;
CREATE CONSTRAINT permit_id IF NOT EXISTS FOR (p:WorkPermit) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT cert_id IF NOT EXISTS FOR (c:Certification) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT worker_id IF NOT EXISTS FOR (w:Worker) REQUIRE w.id IS UNIQUE;
CREATE CONSTRAINT project_id IF NOT EXISTS FOR (pr:Project) REQUIRE pr.id IS UNIQUE;

// 2. Load via Python (see import_graph_real.py)
// The Python script reads epc_dataset_real.json and loads all nodes/relations

// ═══════════════════════════════════════════════════════════════
// TEMPORAL QUERIES
// ═══════════════════════════════════════════════════════════════

// Q1: What permits are required for hot work activities in month 6?
// MATCH (s:Step)-[:REQUIRES_PERMIT]->(p:WorkPermit)
// WHERE s.permit_type = 'hot_work'
// AND s.valid_from <= '2024-06-29' AND s.valid_to >= '2024-06-29'
// RETURN s.name, p.name

// Q2: Which workers are certified for hot work BEFORE the rule change?
// MATCH (w:Worker)-[r:HAS_CERT]->(c:Certification)
// WHERE c.name IN ['Hot Work Safety', 'Fire Watch']
// AND r.valid_from <= '2024-06-29'
// RETURN w.id, w.name, collect(c.name)

// Q3: After rule change — which workers NOW lack the new cert?
// MATCH (w:Worker)
// WHERE NOT EXISTS {
//   MATCH (w)-[r:HAS_CERT]->(c:Certification {name: 'Advanced Fire Watch'})
//   WHERE r.valid_from <= '2024-07-01'
// }
// AND EXISTS {
//   MATCH (w)-[:ASSIGNED_TO]->(s:Step {permit_type: 'hot_work'})
// }
// RETURN w.id, w.name AS workers_needing_retraining

// Q4: Step sequence for a specific activity (PRECEDES chain)
// MATCH path = (s1:Step)-[:PRECEDES*]->(sN:Step)
// WHERE s1.activity_id = 'ME.HE1' AND NOT ()-[:PRECEDES]->(s1)
// RETURN [n IN nodes(path) | n.name + ' (' + toString(n.weight_pct) + '%)'] AS sequence

// Q5: Bitemporal audit — what was the hot work rule at month 5 vs month 7?
// MATCH (p:WorkPermit {id: 'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
// WHERE r.valid_from <= $query_date AND (r.valid_to IS NULL OR r.valid_to >= $query_date)
// RETURN $query_date AS as_of, collect(c.name) AS required_certs
