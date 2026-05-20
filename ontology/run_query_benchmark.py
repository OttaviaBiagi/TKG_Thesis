"""
run_query_benchmark.py — SPARQL temporal overhead on the rdflib graph.

Complements run_cypher_benchmark.py (Neo4j). Measures the overhead
of temporal SPARQL filters on the in-memory OWL-2 graph (5,179 triples).

Run from repo root:
    python ontology/run_query_benchmark.py

Requires: rdflib >= 6.0, ontology/epc_instance_data.ttl
"""
import time
import json
import statistics
from pathlib import Path
from rdflib import Graph

DATA_FILE = Path("ontology/epc_instance_data.ttl")
OUT_FILE  = Path("experiments/UseCase4/results/sparql_benchmark.json")

N_RUNS = 200
WARMUP = 20

DATE_POST = "2024-07-01T00:00:00+00:00"
DATE_PRE  = "2024-05-01T00:00:00+00:00"
TX_SNAP   = "2026-06-01T00:00:00+00:00"

PREFIX = """
PREFIX epc: <http://tecnicasreunidas.es/ontology/epc#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

QUERIES = [
    {
        "id": "SA1",
        "label": "1-hop cert holding — atemporal",
        "pair": "S1",
        "sparql": PREFIX + """
            SELECT ?worker ?cert WHERE {
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert .
            }
        """,
    },
    {
        "id": "ST1",
        "label": "1-hop cert holding — valid-time filter",
        "pair": "S1",
        "sparql": PREFIX + f"""
            SELECT ?worker ?cert WHERE {{
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert ;
                   epc:validFrom ?vf ;
                   epc:validTo   ?vt .
                FILTER("{DATE_POST}"^^xsd:dateTime >= ?vf)
                FILTER("{DATE_POST}"^^xsd:dateTime <= ?vt)
            }}
        """,
    },
    {
        "id": "SA2",
        "label": "3-hop compliance chain — atemporal",
        "pair": "S2",
        "sparql": PREFIX + """
            SELECT ?worker ?permit ?cert WHERE {
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert .
                ?req a epc:PermitCertRequirement ;
                     epc:forPermit ?permit ;
                     epc:requiresCertification ?cert .
            }
        """,
    },
    {
        "id": "ST2",
        "label": "3-hop compliance chain — valid-time filter",
        "pair": "S2",
        "sparql": PREFIX + f"""
            SELECT ?worker ?permit ?cert WHERE {{
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert ;
                   epc:validFrom ?hf ; epc:validTo ?ht .
                ?req a epc:PermitCertRequirement ;
                     epc:forPermit ?permit ;
                     epc:requiresCertification ?cert ;
                     epc:validFrom ?rf ; epc:validTo ?rt .
                FILTER("{DATE_POST}"^^xsd:dateTime >= ?hf && "{DATE_POST}"^^xsd:dateTime <= ?ht)
                FILTER("{DATE_POST}"^^xsd:dateTime >= ?rf && "{DATE_POST}"^^xsd:dateTime <= ?rt)
            }}
        """,
    },
    {
        "id": "ST3_single",
        "label": "Cert lookup — valid-time only (single axis)",
        "pair": "S3",
        "sparql": PREFIX + f"""
            SELECT ?worker ?cert ?vf ?vt WHERE {{
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert ;
                   epc:validFrom ?vf ; epc:validTo ?vt .
                FILTER("{DATE_PRE}"^^xsd:dateTime >= ?vf)
                FILTER("{DATE_PRE}"^^xsd:dateTime <= ?vt)
            }}
        """,
    },
    {
        "id": "ST3_bitemporal",
        "label": "Cert lookup — bitemporal as-of (both axes)",
        "pair": "S3",
        "sparql": PREFIX + f"""
            SELECT ?worker ?cert ?vf ?vt ?tx WHERE {{
                ?h a epc:CertificationHolding ;
                   epc:byWorker ?worker ;
                   epc:holdsCertification ?cert ;
                   epc:validFrom ?vf ; epc:validTo ?vt ;
                   epc:txTime    ?tx .
                FILTER("{DATE_PRE}"^^xsd:dateTime >= ?vf)
                FILTER("{DATE_PRE}"^^xsd:dateTime <= ?vt)
                FILTER(?tx <= "{TX_SNAP}"^^xsd:dateTime)
            }}
        """,
    },
]


def run_query(g, sparql, n_runs, warmup):
    for _ in range(warmup):
        list(g.query(sparql))

    times_ms = []
    row_count = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rows = list(g.query(sparql))
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)
        row_count = len(rows)

    return {
        "mean_ms":   round(statistics.mean(times_ms), 3),
        "std_ms":    round(statistics.stdev(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "min_ms":    round(min(times_ms), 3),
        "max_ms":    round(max(times_ms), 3),
        "rows":      row_count,
        "n_runs":    n_runs,
        "warmup":    warmup,
    }


def main():
    if not DATA_FILE.exists():
        print(f"Missing: {DATA_FILE}")
        print("Run: python ontology/populate_onto.py")
        return

    print(f"Loading {DATA_FILE} ...")
    g = Graph()
    g.parse(str(DATA_FILE), format="turtle")
    print(f"Graph: {len(g):,} triples\n")
    print(f"Running {len(QUERIES)} queries x {N_RUNS} runs each (+ {WARMUP} warmup)...\n")

    results = {}
    for q in QUERIES:
        print(f"  [{q['id']}] {q['label']} ... ", end="", flush=True)
        try:
            m = run_query(g, q["sparql"], N_RUNS, WARMUP)
            results[q["id"]] = {**q, "metrics": m}
            print(f"{m['mean_ms']:.1f} ms ± {m['std_ms']:.1f}  ({m['rows']} rows)")
        except Exception as e:
            results[q["id"]] = {**q, "error": str(e)}
            print(f"ERROR: {e}")

    pairs = {
        "S1": ("SA1",          "ST1",           "1-hop cert holding"),
        "S2": ("SA2",          "ST2",           "3-hop compliance chain"),
        "S3": ("ST3_single",   "ST3_bitemporal","Single-axis vs bitemporal as-of"),
    }

    overhead_table = []
    for pair_id, (base_id, temp_id, desc) in pairs.items():
        if "error" in results.get(base_id, {}) or "error" in results.get(temp_id, {}):
            overhead_table.append({"pair": pair_id, "desc": desc, "status": "ERROR"})
            continue
        base = results[base_id]["metrics"]
        temp = results[temp_id]["metrics"]
        pct  = round((temp["mean_ms"] - base["mean_ms"]) / base["mean_ms"] * 100, 1)
        overhead_table.append({
            "pair":         pair_id,
            "desc":         desc,
            "base_ms":      base["mean_ms"],
            "temp_ms":      temp["mean_ms"],
            "overhead_pct": pct,
            "h3_pass":      pct < 50.0,
        })

    print("\n" + "=" * 72)
    print("  H3 OVERHEAD — SPARQL temporal filters (rdflib, 5,179 triples)")
    print("=" * 72)
    print(f"  {'Pair':<5}  {'Base (ms)':>10}  {'Temporal (ms)':>14}  {'Overhead':>9}  {'H3 <50%':>8}")
    print("  " + "-" * 60)
    all_pass = True
    for row in overhead_table:
        if row.get("status") == "ERROR":
            print(f"  {row['pair']:<5}  ERROR")
            all_pass = False
            continue
        symbol = "PASS" if row["h3_pass"] else "FAIL"
        if not row["h3_pass"]:
            all_pass = False
        print(f"  {row['pair']:<5}  {row['base_ms']:>10.1f}  {row['temp_ms']:>14.1f}"
              f"  {row['overhead_pct']:>8.1f}%  {symbol:>8}  {row['desc']}")
    print("=" * 72)
    print(f"  H3 OVERALL (rdflib): {'SUPPORTED' if all_pass else 'NOT FULLY SUPPORTED'}")
    print("=" * 72)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark":   "SO4_H3_sparql_temporal_overhead",
        "substrate":   "rdflib 7.x in-memory",
        "triples":     len(g),
        "n_runs":      N_RUNS,
        "warmup":      WARMUP,
        "date_anchors": {
            "date_post": DATE_POST,
            "date_pre":  DATE_PRE,
            "tx_snap":   TX_SNAP,
        },
        "queries":     {k: v for k, v in results.items()},
        "overhead":    overhead_table,
    }
    OUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
