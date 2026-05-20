"""
run_sparql.py — Execute all SPARQL queries against the populated OWL-2 graph.

Run from repo root:
    python3 ontology/run_sparql.py

Requires:  rdflib >= 6.0   (pip install rdflib)
"""
import sys
from pathlib import Path
from rdflib import Graph

ONTO_DIR  = Path("ontology")
DATA_FILE = ONTO_DIR / "epc_instance_data.ttl"
SPARQL_DIR = ONTO_DIR / "sparql"

QUERIES = [
    ("Q1", "Q1_workers_before_rule_change.sparql",
     "Workers qualified for hot_work BEFORE rule change (2024-06-28)"),
    ("Q2", "Q2_workers_after_rule_change.sparql",
     "Workers qualified for hot_work AFTER rule change (2024-07-01)"),
    ("Q3", "Q3_delta_non_compliant.sparql",
     "Workers who became non-compliant after rule change"),
    ("Q4", "Q4_audit_trail.sparql",
     "Bitemporal audit trail of hot_work requirement changes"),
    ("Q5", "Q5_bitemporal_asof.sparql",
     "Bitemporal 'as-of' query: system knowledge at txTime, for validTime"),
    ("Q6", "Q6_violation_inference.sparql",
     "CONSTRUCT: materialise ComplianceViolation instances via OWL-2 pattern"),
]


def load_graph() -> Graph:
    if not DATA_FILE.exists():
        print(f"Instance data not found: {DATA_FILE}")
        print("Run:  python3 ontology/populate_onto.py")
        sys.exit(1)
    g = Graph()
    g.parse(str(DATA_FILE), format="turtle")
    print(f"Graph loaded: {len(g):,} triples\n")
    return g


def run_select(g, sparql_text, label):
    results = list(g.query(sparql_text))
    if not results:
        print("  (no results)")
        return results
    # Print header
    vars_ = results[0].labels if hasattr(results[0], "labels") else []
    if vars_:
        header = "  " + "  |  ".join(str(v) for v in vars_)
        print(header)
        print("  " + "-" * max(len(header) - 2, 60))
    for row in results[:20]:    # cap output at 20 rows
        cells = [str(v) if v is not None else "NULL" for v in row]
        print("  " + "  |  ".join(cells))
    if len(results) > 20:
        print(f"  ... ({len(results)} total rows)")
    return results


def run_construct(g, sparql_text, label):
    result_graph = g.query(sparql_text).graph
    if result_graph is None:
        result_graph = Graph()
    n = len(result_graph)
    print(f"  CONSTRUCT produced {n} triples")
    if n > 0:
        print("  First 5 triples:")
        for i, (s, p, o) in enumerate(result_graph):
            if i >= 5: break
            print(f"    {s.split('#')[-1]}  {p.split('#')[-1]}  {o.split('#')[-1] if hasattr(o,'split') else o}")
    return result_graph


def main():
    g = load_graph()
    summary = []

    for qid, fname, description in QUERIES:
        path = SPARQL_DIR / fname
        if not path.exists():
            print(f"[{qid}] MISSING: {fname}\n")
            summary.append((qid, "MISSING", 0))
            continue

        sparql = path.read_text(encoding="utf-8")
        print(f"{'='*70}")
        print(f"  [{qid}] {description}")
        print(f"{'='*70}")

        is_construct = sparql.strip().upper().startswith("CONSTRUCT")
        try:
            if is_construct:
                result = run_construct(g, sparql, qid)
                n = len(result)
            else:
                result = run_select(g, sparql, qid)
                n = len(result)
            summary.append((qid, "OK", n))
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append((qid, "ERROR", 0))
        print()

    # Summary
    print("=" * 70)
    print("  SPARQL QUERY SUMMARY")
    print("=" * 70)
    print(f"  {'Query':<6}  {'Status':<8}  {'Rows/Triples':>12}  Description")
    print("  " + "-" * 65)
    for qid, status, n in summary:
        desc = next((d for q, _, d in QUERIES if q == qid), "")[:45]
        print(f"  {qid:<6}  {status:<8}  {n:>12}  {desc}")
    print("=" * 70)


if __name__ == "__main__":
    main()
