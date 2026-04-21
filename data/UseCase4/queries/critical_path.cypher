// UseCase4 — Critical Path & Bottleneck Queries on EPC TKG

// ── Q5: Critical path (longest chain in PRECEDES, max depth 30) ───────────────
MATCH path = (start:Step)-[:PRECEDES*..30]->(end:Step)
WHERE NOT ()-[:PRECEDES]->(start)
  AND NOT (end)-[:PRECEDES]->()
WITH path, length(path) AS depth
ORDER BY depth DESC
LIMIT 1
RETURN [n IN nodes(path) | n.name] AS critical_path,
       [n IN nodes(path) | n.activity_id] AS activities,
       depth;

// ── Q6: Bottleneck by discipline — permit distribution per discipline ──────────
MATCH (act:Activity)-[:HAS_STEP]->(s:Step)-[:REQUIRES_PERMIT]->(wp:WorkPermit)
RETURN act.discipline AS discipline,
       wp.name        AS permit,
       count(s)       AS n_steps
ORDER BY discipline, n_steps DESC;

// ── Q7: Most blocking steps (steps with most downstream dependencies) ──────────
MATCH (s:Step)-[:PRECEDES*1..30]->(downstream:Step)
WITH s, count(downstream) AS blocks
ORDER BY blocks DESC
LIMIT 10
MATCH (s)-[:REQUIRES_PERMIT]->(wp:WorkPermit)
RETURN s.id, s.name, s.activity_id, wp.id AS permit, blocks;

// ── Q8: Activities on critical path ───────────────────────────────────────────
MATCH path = (start:Step)-[:PRECEDES*..30]->(end:Step)
WHERE NOT ()-[:PRECEDES]->(start)
  AND NOT (end)-[:PRECEDES]->()
WITH path, length(path) AS depth
ORDER BY depth DESC
LIMIT 5
UNWIND nodes(path) AS s
MATCH (act:Activity)-[:HAS_STEP]->(s)
RETURN DISTINCT act.id, act.name, act.discipline, depth
ORDER BY depth DESC;
