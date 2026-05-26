"""
run_cypher_benchmark.py — SO4 / H3: Temporal query overhead measurement.

Compares atemporal vs temporal Cypher queries on the live Neo4j TKG to test:
    H3: "Temporal query overhead < 50% relative to equivalent atemporal queries."

5 query pairs, each run N_RUNS times (after WARMUP warmup runs):
    QA1 / QT1 — 1-hop cert lookup
    QA2 / QT2 — 3-hop compliance chain (Step->Permit->Cert->Worker)
    QA3 / QT3 — 3-hop non-compliance detection (NOT ALL certs held)
    QA5 / QT5 — 4-hop assignment chain via ASSIGNED_TO (Worker->Step->Permit->Cert)
    QA4 / QT4 — bitemporal as-of (valid-time + tx-time axes simultaneously)

Results saved to experiments/UseCase4/results/query_benchmark.json.

Run from repo root on Linux:
    python data/UseCase4/run_cypher_benchmark.py

Requirements:
    pip install neo4j
    Neo4j running at bolt://localhost:7687 with UC4 data loaded.
"""
import json
import time
import statistics
from pathlib import Path
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password"

N_RUNS  = 100    # timed repetitions per query
WARMUP  = 10     # discarded warmup runs

# Date anchors (rule change was 2024-06-29)
DATE_PRE  = "2024-06-01T00:00:00+00:00"   # valid-time before rule change
DATE_POST = "2024-07-01T00:00:00+00:00"   # valid-time after rule change (P1-P4)
DATE_END  = "2025-07-01T00:00:00+00:00"   # project end — covers all ASSIGNED_TO assignments (P5)
TX_SNAP   = "2026-06-01T00:00:00+00:00"   # tx-time snapshot (captures all records)

OUT_FILE  = Path("experiments/UseCase4/results/query_benchmark.json")

# ── Query definitions ─────────────────────────────────────────────────────────
QUERIES = [
    # ── Pair 1: 1-hop cert lookup ──────────────────────────────────────────────
    {
        "id": "QA1",
        "label": "1-hop cert lookup — atemporal",
        "pair": "P1",
        "temporal": False,
        "cypher": """
            MATCH (w:Worker)-[:HAS_CERT]->(c:Certification)
            RETURN w.id AS worker, c.id AS cert
        """,
        "params": {},
    },
    {
        "id": "QT1",
        "label": "1-hop cert lookup — valid-time filter",
        "pair": "P1",
        "temporal": True,
        "cypher": """
            MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
            WHERE h.valid_from <= $check_date AND (h.valid_to IS NULL OR h.valid_to >= $check_date)
            RETURN w.id AS worker, c.id AS cert
        """,
        "params": {"check_date": DATE_POST},
    },

    # ── Pair 2: 3-hop compliance chain ─────────────────────────────────────────
    {
        "id": "QA2",
        "label": "3-hop compliance chain — atemporal",
        "pair": "P2",
        "temporal": False,
        "cypher": """
            MATCH (w:Worker)-[:HAS_CERT]->(c:Certification)
                  <-[:REQUIRES_CERT]-(wp:WorkPermit)
                  <-[:REQUIRES_PERMIT]-(s:Step)
            RETURN w.id AS worker, wp.id AS permit, c.id AS cert, count(s) AS steps
        """,
        "params": {},
    },
    {
        "id": "QT2",
        "label": "3-hop compliance chain — valid-time filter",
        "pair": "P2",
        "temporal": True,
        "cypher": """
            MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
                  <-[r:REQUIRES_CERT]-(wp:WorkPermit)
                  <-[:REQUIRES_PERMIT]-(s:Step)
            WHERE h.valid_from  <= $check_date
              AND (h.valid_to   IS NULL OR h.valid_to >= $check_date)
              AND r.valid_from  <= $check_date
              AND (r.valid_to   IS NULL OR r.valid_to >= $check_date)
            RETURN w.id AS worker, wp.id AS permit, c.id AS cert, count(s) AS steps
        """,
        "params": {"check_date": DATE_POST},
    },

    # ── Pair 3: Non-compliance detection (workers missing at least one required cert) ──
    {
        "id": "QA3",
        "label": "Non-compliance detection — atemporal",
        "pair": "P3",
        "temporal": False,
        "cypher": """
            MATCH (wp:WorkPermit {id:'hot_work'})-[:REQUIRES_CERT]->(req:Certification)
            WITH collect(req.id) AS required
            MATCH (w:Worker)
            WITH w, required,
                 [(w)-[:HAS_CERT]->(c:Certification) | c.id] AS held
            WHERE ANY(r IN required WHERE NOT r IN held)
            RETURN w.id AS worker, w.name AS name
        """,
        "params": {},
    },
    {
        "id": "QT3",
        "label": "Non-compliance detection — valid-time filter",
        "pair": "P3",
        "temporal": True,
        "cypher": """
            MATCH (wp:WorkPermit {id:'hot_work'})-[r:REQUIRES_CERT]->(req:Certification)
            WHERE r.valid_from <= $check_date
              AND (r.valid_to  IS NULL OR r.valid_to >= $check_date)
            WITH collect(req.id) AS required
            MATCH (w:Worker)
            WITH w, required,
                 [(w)-[h:HAS_CERT]->(c:Certification)
                  WHERE h.valid_from <= $check_date
                    AND (h.valid_to IS NULL OR h.valid_to >= $check_date) | c.id] AS held
            WHERE ANY(r IN required WHERE NOT r IN held)
            RETURN w.id AS worker, w.name AS name
        """,
        "params": {"check_date": DATE_POST},
    },

    # ── Pair 5: 4-hop chain via ASSIGNED_TO (Worker→Step→Permit→Cert) ───────────
    {
        "id": "QA5",
        "label": "4-hop assignment chain — atemporal",
        "pair": "P5",
        "temporal": False,
        "cypher": """
            MATCH (w:Worker)-[:ASSIGNED_TO]->(s:Step)
                  -[:REQUIRES_PERMIT]->(wp:WorkPermit)
                  -[:REQUIRES_CERT]->(c:Certification)
            RETURN w.id AS worker, s.id AS step, wp.id AS permit, c.id AS cert
        """,
        "params": {},
    },
    {
        "id": "QT5",
        "label": "4-hop assignment chain — valid-time filter",
        "pair": "P5",
        "temporal": True,
        "cypher": """
            MATCH (w:Worker)-[a:ASSIGNED_TO]->(s:Step)
                  -[:REQUIRES_PERMIT]->(wp:WorkPermit)
                  -[r:REQUIRES_CERT]->(c:Certification)
            WHERE a.valid_from <= $check_date
              AND (a.valid_to  IS NULL OR a.valid_to  >= $check_date)
              AND r.valid_from <= $check_date
              AND (r.valid_to  IS NULL OR r.valid_to  >= $check_date)
            RETURN w.id AS worker, s.id AS step, wp.id AS permit, c.id AS cert
        """,
        "params": {"check_date": DATE_POST},
    },

    # ── Pair 4: Bitemporal as-of (valid-time + tx-time axes) ───────────────────
    {
        "id": "QT4_single",
        "label": "Cert lookup — valid-time only (single axis)",
        "pair": "P4",
        "temporal": True,
        "cypher": """
            MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
            WHERE h.valid_from <= $valid_date
              AND (h.valid_to  IS NULL OR h.valid_to >= $valid_date)
            RETURN w.id, c.id, h.valid_from, h.valid_to
        """,
        "params": {"valid_date": DATE_PRE},
    },
    {
        "id": "QT4_bitemporal",
        "label": "Cert lookup — bitemporal as-of (both axes)",
        "pair": "P4",
        "temporal": True,
        "cypher": """
            MATCH (w:Worker)-[h:HAS_CERT]->(c:Certification)
            WHERE h.valid_from <= $valid_date
              AND (h.valid_to  IS NULL OR h.valid_to >= $valid_date)
              AND h.tx_time   <= $tx_snap
            RETURN w.id, c.id, h.valid_from, h.valid_to, h.tx_time
        """,
        "params": {"valid_date": DATE_PRE, "tx_snap": TX_SNAP},
    },
]


# ── Benchmark runner ──────────────────────────────────────────────────────────
def run_query(session, cypher, params, n_runs, warmup):
    for _ in range(warmup):
        list(session.run(cypher, **params))

    times_ms = []
    row_counts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        results = list(session.run(cypher, **params))
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
        row_counts.append(len(results))

    return {
        "mean_ms":   round(statistics.mean(times_ms), 3),
        "std_ms":    round(statistics.stdev(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "min_ms":    round(min(times_ms), 3),
        "max_ms":    round(max(times_ms), 3),
        "rows":      row_counts[0],
        "n_runs":    n_runs,
        "warmup":    warmup,
    }


def compute_overhead(atemporal, temporal):
    overhead_pct = ((temporal["mean_ms"] - atemporal["mean_ms"])
                    / atemporal["mean_ms"] * 100)
    return round(overhead_pct, 1)


def main():
    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    results = {}
    with driver.session() as session:
        # Verify connection
        rec = session.run("MATCH (n) RETURN count(n) AS total").single()
        total_nodes = rec["total"]
        print(f"Connected. Total nodes: {total_nodes:,}")
        print(f"Running {len(QUERIES)} queries x {N_RUNS} runs each (+ {WARMUP} warmup)...\n")

        for q in QUERIES:
            print(f"  [{q['id']}] {q['label']} ... ", end="", flush=True)
            try:
                metrics = run_query(session, q["cypher"], q["params"], N_RUNS, WARMUP)
                results[q["id"]] = {**q, "metrics": metrics}
                print(f"{metrics['mean_ms']:.1f} ms ± {metrics['std_ms']:.1f}  ({metrics['rows']} rows)")
            except Exception as e:
                results[q["id"]] = {**q, "error": str(e)}
                print(f"ERROR: {e}")

    driver.close()

    # ── Compute overhead per pair ──────────────────────────────────────────────
    pairs = {
        "P1": ("QA1",        "QT1",           "1-hop cert lookup"),
        "P2": ("QA2",        "QT2",           "3-hop compliance chain"),
        "P3": ("QA3",        "QT3",           "Non-compliance detection"),
        "P4": ("QT4_single", "QT4_bitemporal","Single-axis vs bitemporal as-of"),
        "P5": ("QA5",        "QT5",           "4-hop assignment chain (ASSIGNED_TO)"),
    }

    overhead_table = []
    for pair_id, (base_id, temp_id, desc) in pairs.items():
        if "error" in results.get(base_id, {}) or "error" in results.get(temp_id, {}):
            overhead_table.append({"pair": pair_id, "desc": desc, "status": "ERROR"})
            continue
        base = results[base_id]["metrics"]
        temp = results[temp_id]["metrics"]
        pct  = compute_overhead(base, temp)
        overhead_table.append({
            "pair":       pair_id,
            "desc":       desc,
            "base_id":    base_id,
            "temp_id":    temp_id,
            "base_ms":    base["mean_ms"],
            "temp_ms":    temp["mean_ms"],
            "overhead_pct": pct,
            "h3_pass":    pct < 50.0,
        })

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  H3 OVERHEAD BENCHMARK — Temporal vs Atemporal Cypher")
    print("=" * 72)
    print(f"  {'Pair':<5}  {'Base (ms)':>10}  {'Temporal (ms)':>14}  {'Overhead':>9}  {'H3 <50%':>8}")
    print("  " + "-" * 60)
    all_pass = True
    for row in overhead_table:
        if row.get("status") == "ERROR":
            print(f"  {row['pair']:<5}  {'ERROR':>36}")
            all_pass = False
            continue
        symbol = "PASS" if row["h3_pass"] else "FAIL"
        if not row["h3_pass"]:
            all_pass = False
        print(f"  {row['pair']:<5}  {row['base_ms']:>10.1f}  {row['temp_ms']:>14.1f}"
              f"  {row['overhead_pct']:>8.1f}%  {symbol:>8}  {row['desc']}")

    print("=" * 72)
    print(f"  H3 OVERALL: {'SUPPORTED' if all_pass else 'NOT FULLY SUPPORTED'}")
    print("=" * 72)

    # ── Save results ───────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "SO4_H3_temporal_overhead",
        "neo4j_uri": NEO4J_URI,
        "n_runs": N_RUNS,
        "warmup": WARMUP,
        "date_anchors": {
            "date_pre": DATE_PRE,
            "date_post": DATE_POST,
            "tx_snap": TX_SNAP,
        },
        "queries":  {k: v for k, v in results.items()},
        "overhead": overhead_table,
    }
    OUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
