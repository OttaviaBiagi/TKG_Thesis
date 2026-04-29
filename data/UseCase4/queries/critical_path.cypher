// UseCase4 — Critical Path & Bottleneck Queries on EPC TKG

// ── Q5: Critical path (max total estimated_hours, depth limit 30) ────────────
// NOTE: hop-count longest path ≠ critical path; must use sum of estimated_hours
MATCH path = (start:Step)-[:PRECEDES*..30]->(end:Step)
WHERE NOT ()-[:PRECEDES]->(start)
  AND NOT (end)-[:PRECEDES]->()
WITH path,
     reduce(total=0, n IN nodes(path) | total + coalesce(n.estimated_hours, 0)) AS total_hours,
     length(path) AS depth
ORDER BY total_hours DESC
LIMIT 1
RETURN [n IN nodes(path) | n.name]        AS critical_path,
       [n IN nodes(path) | n.activity_id] AS activities,
       total_hours,
       depth;

// ── Q6: Bottleneck by discipline — permit distribution per discipline ──────────
MATCH (act:Activity)-[:HAS_STEP]->(s:Step)-[:REQUIRES_PERMIT]->(wp:WorkPermit)
RETURN act.discipline AS discipline,
       wp.name        AS permit,
       count(s)       AS n_steps
ORDER BY discipline, n_steps DESC;

// ── Q7: Most blocking steps (steps with most downstream dependencies) ──────────
// NOTE: OPTIONAL MATCH so steps without a permit are not excluded
MATCH (s:Step)-[:PRECEDES*1..30]->(downstream:Step)
WITH s, count(downstream) AS blocks
ORDER BY blocks DESC
LIMIT 10
OPTIONAL MATCH (s)-[:REQUIRES_PERMIT]->(wp:WorkPermit)
RETURN s.id, s.name, s.activity_id, coalesce(wp.id, 'none') AS permit, blocks;

// ── Q8: Activities on critical path (top-5 by total duration) ─────────────────
MATCH path = (start:Step)-[:PRECEDES*..30]->(end:Step)
WHERE NOT ()-[:PRECEDES]->(start)
  AND NOT (end)-[:PRECEDES]->()
WITH path,
     reduce(total=0, n IN nodes(path) | total + coalesce(n.estimated_hours, 0)) AS total_hours,
     length(path) AS depth
ORDER BY total_hours DESC
LIMIT 5
UNWIND nodes(path) AS s
MATCH (act:Activity)-[:HAS_STEP]->(s)
RETURN DISTINCT act.id, act.name, act.discipline, total_hours, depth
ORDER BY total_hours DESC, discipline;
