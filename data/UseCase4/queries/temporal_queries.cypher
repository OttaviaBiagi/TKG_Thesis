// UseCase4 — Bitemporal Queries on EPC TKG
// Hot Work Permit Rule Change: 2024-06-29

// ── Q1: Workers qualified for hot_work BEFORE rule change ─────────────────────
MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
WHERE r.valid_from < '2024-06-29T00:00:00+00:00'
WITH collect(c.id) AS required_certs
MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
WHERE c.id IN required_certs
  AND h.valid_from <= '2024-06-28T00:00:00+00:00'
  AND h.valid_to   >= '2024-06-28T00:00:00+00:00'
WITH w, collect(c.id) AS worker_certs, required_certs
WHERE all(rc IN required_certs WHERE rc IN worker_certs)
RETURN w.id, w.name, w.discipline, worker_certs;

// ── Q2: Workers qualified for hot_work AFTER rule change ──────────────────────
MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
WHERE r.valid_from <= '2024-07-01T00:00:00+00:00'
  AND (r.valid_to IS NULL OR r.valid_to > '2024-07-01T00:00:00+00:00')
WITH collect(c.id) AS required_certs
MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
WHERE c.id IN required_certs
  AND h.valid_from <= '2024-07-01T00:00:00+00:00'
  AND h.valid_to   >= '2024-07-01T00:00:00+00:00'
WITH w, collect(c.id) AS worker_certs, required_certs
WHERE all(rc IN required_certs WHERE rc IN worker_certs)
RETURN w.id, w.name, w.discipline, worker_certs;

// ── Q3: Delta — workers who became non-compliant after rule change ─────────────
MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
WHERE r.valid_from < '2024-06-29T00:00:00+00:00'
WITH collect(c.id) AS old_reqs
MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
WHERE c.id IN old_reqs
  AND h.valid_from <= '2024-06-28T00:00:00+00:00'
  AND h.valid_to   >= '2024-06-28T00:00:00+00:00'
WITH w, collect(c.id) AS worker_certs, old_reqs
WHERE all(rc IN old_reqs WHERE rc IN worker_certs)
WITH collect(w.id) AS qualified_before
MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
WITH collect(c.id) AS new_reqs, qualified_before
MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
WHERE w.id IN qualified_before
  AND c.id IN new_reqs
  AND h.valid_from <= '2024-07-01T00:00:00+00:00'
  AND h.valid_to   >= '2024-07-01T00:00:00+00:00'
WITH w, collect(c.id) AS worker_certs, new_reqs
WHERE NOT all(rc IN new_reqs WHERE rc IN worker_certs)
RETURN w.id, w.name, w.discipline AS non_compliant_after_change;

// ── Q4: Audit trail of rule change ────────────────────────────────────────────
MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
RETURN wp.name, c.name, r.valid_from, r.valid_to, r.tx_time
ORDER BY r.valid_from;
