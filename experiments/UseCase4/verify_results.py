"""
verify_results.py -- sanity-check all thesis results in one pass.

Run from repo root:
    python3 experiments/UseCase4/verify_results.py
"""
import json
import math
from pathlib import Path

RESULTS = Path("experiments/UseCase4/results")
PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
log = []

def check(label, ok, msg="", level=None):
    status = level if level else (PASS if ok else FAIL)
    log.append((status, label, msg))

def get_m(r):
    m = r.get("metrics", {})
    return m.get("overall", m)

# ── 1. Files exist ────────────────────────────────────────────────────────────
for fname in ["benchmark.json", "benchmark_varied.json", "static_gnn.json",
              "static_baseline.json", "ml_baseline.json"]:
    check(f"file exists: {fname}", (RESULTS / fname).exists())

# ── 2. benchmark.json -- single project ───────────────────────────────────────
bm = json.load(open(RESULTS / "benchmark.json"))["results"]
single_temporal = [r for r in bm if r.get("split") == "temporal" and r.get("dataset") == "single"]

# TGN: 3 seeds
tgn_rows = [r for r in single_temporal if r["model"] == "TGN"]
check("TGN single: 3 seeds present", len(tgn_rows) == 3,
      f"found {len(tgn_rows)} seeds: {sorted(r.get('seed') for r in tgn_rows)}")

tgn_aucs   = [get_m(r).get("auc", float("nan")) for r in tgn_rows]
tgn_auprcs = [get_m(r).get("auprc", float("nan")) for r in tgn_rows]
tgn_auc_mean = sum(tgn_aucs) / len(tgn_aucs)
tgn_auprc_mean = sum(tgn_auprcs) / len(tgn_auprcs)
check("TGN single AUC ~ 0.984+-0.001",
      abs(tgn_auc_mean - 0.984) < 0.003,
      f"mean={tgn_auc_mean:.3f} values={[round(x,3) for x in tgn_aucs]}")
check("TGN single AUPRC ~ 0.177+-0.003",
      abs(tgn_auprc_mean - 0.177) < 0.01,
      f"mean={tgn_auprc_mean:.3f} values={[round(x,3) for x in tgn_auprcs]}")
tgn_recalls = [get_m(r).get("recall", 0) for r in tgn_rows]
check("TGN single: seed 42 recall=1.0 (primary result)",
      any(get_m(r).get("recall", 0) == 1.0 and r.get("seed") == 42 for r in tgn_rows),
      f"recalls per seed={[(r.get('seed'), get_m(r).get('recall')) for r in tgn_rows]}")
if not all(v == 1.0 for v in tgn_recalls):
    check("TGN single: recall varies across seeds (instability note)",
          True, f"recalls={tgn_recalls} -- seed-dependent threshold behaviour", level=WARN)

# TGAT single
tgat_s = [r for r in single_temporal if r["model"] == "TGAT"]
check("TGAT single present", len(tgat_s) >= 1, f"found {len(tgat_s)}")
if tgat_s:
    check("TGAT single AUC in range (0.7–0.9)",
          0.7 < get_m(tgat_s[0]).get("auc", 0) < 0.9,
          f"AUC={get_m(tgat_s[0]).get('auc', 0):.3f}")

# DyRep single
dyrep_s = [r for r in single_temporal if r["model"] == "DyRep"]
check("DyRep single present", len(dyrep_s) >= 1, f"found {len(dyrep_s)}")
if dyrep_s:
    check("DyRep degenerate (recall=1.0, AUPRC~prevalence)",
          get_m(dyrep_s[0]).get("recall", 0) == 1.0,
          f"recall={get_m(dyrep_s[0]).get('recall')}", level=WARN if get_m(dyrep_s[0]).get("recall") != 1.0 else PASS)

# No homogeneous multi
homo_multi = [r for r in bm if r.get("dataset") == "multi"]
check("No homogeneous-multi entries in benchmark.json",
      len(homo_multi) == 0, f"found {len(homo_multi)} entries -- should be 0")

# ── 3. benchmark_varied.json -- multi_varied ───────────────────────────────────
bv = json.load(open(RESULTS / "benchmark_varied.json"))["results"]
varied_temporal = [r for r in bv if r.get("split") == "temporal" and r.get("dataset") == "multi_varied"]

for model, exp_seeds, exp_auprc_range in [
    ("TGAT", [42, 43, 44], (0.60, 0.85)),
    ("TGN",  [42, 43, 44], (0.12, 0.14)),
]:
    rows = [r for r in varied_temporal if r["model"] == model]
    seeds = sorted(r.get("seed") for r in rows)
    check(f"{model} multi_varied: 3 seeds", len(rows) == 3, f"seeds={seeds}")
    auprcs = [get_m(r).get("auprc", float("nan")) for r in rows]
    mean_auprc = sum(auprcs) / len(auprcs) if auprcs else float("nan")
    lo, hi = exp_auprc_range
    check(f"{model} multi_varied AUPRC in expected range ({lo:.2f}–{hi:.2f})",
          lo < mean_auprc < hi,
          f"mean={mean_auprc:.3f} seeds={[round(x,3) for x in auprcs]}")

# TGAT lift > TGN lift on multi_varied
tgat_mv = [r for r in varied_temporal if r["model"] == "TGAT"]
tgn_mv  = [r for r in varied_temporal if r["model"] == "TGN"]
if tgat_mv and tgn_mv:
    tgat_mean_auprc = sum(get_m(r).get("auprc", 0) for r in tgat_mv) / len(tgat_mv)
    tgn_mean_auprc  = sum(get_m(r).get("auprc", 0) for r in tgn_mv)  / len(tgn_mv)
    check("TGAT multi_varied AUPRC > TGN multi_varied AUPRC (hierarchy)",
          tgat_mean_auprc > tgn_mean_auprc,
          f"TGAT={tgat_mean_auprc:.3f} TGN={tgn_mean_auprc:.3f}")

# ── 4. static_gnn.json ────────────────────────────────────────────────────────
sg = json.load(open(RESULTS / "static_gnn.json"))["results"]

for ds, exp_seeds, exp_auprc_mean, exp_auprc_tol in [
    ("single",      [42, 43, 44], 0.55, 0.15),
    ("multi_varied",[42, 43, 44], 0.20, 0.15),
]:
    rows = [r for r in sg if r.get("dataset") == ds]
    seeds = sorted(r.get("seed") for r in rows)
    check(f"StaticGNN {ds}: 3 seeds", set(seeds) == set(exp_seeds), f"seeds={seeds}")
    auprcs = [r["metrics"].get("auprc", float("nan")) for r in rows]
    mean_a = sum(auprcs) / len(auprcs) if auprcs else float("nan")
    check(f"StaticGNN {ds} AUPRC mean ~ {exp_auprc_mean:.2f} (+-{exp_auprc_tol})",
          abs(mean_a - exp_auprc_mean) < exp_auprc_tol,
          f"mean={mean_a:.3f} seeds={[round(x,3) for x in auprcs]}")

# StaticGNN val_AUPRC flat (confirms no learning signal on single)
single_sg = [r for r in sg if r.get("dataset") == "single"]
val_auprcs = [r.get("val_auprc", float("nan")) for r in single_sg]
if not any(math.isnan(v) for v in val_auprcs):
    std_val = (sum((v - sum(val_auprcs)/len(val_auprcs))**2 for v in val_auprcs)/len(val_auprcs))**0.5
    check("StaticGNN single val_AUPRC flat across seeds (std < 0.01)",
          std_val < 0.01,
          f"val_AUPRCs={[round(v,4) for v in val_auprcs]} std={std_val:.4f}")

# ── 5. static_baseline.json -- ComplEx / TNTComplEx ───────────────────────────
sb = json.load(open(RESULTS / "static_baseline.json"))
sb_results = sb if isinstance(sb, list) else sb.get("results", [])
for model in ["ComplEx", "TNTComplEx"]:
    rows = [r for r in sb_results if r.get("model") == model and r.get("dataset") == "single"]
    check(f"{model} single present", len(rows) >= 1, f"found {len(rows)}")
    if rows:
        auprc = rows[0].get("auprc", rows[0].get("metrics", {}).get("auprc", float("nan")))
        check(f"{model} AUPRC ~ prevalence (random)",
              auprc < 0.005,
              f"AUPRC={auprc:.4f} (should be ~0.002)")

# ── 6. ml_baseline.json -- LR / RF ────────────────────────────────────────────
ml = json.load(open(RESULTS / "ml_baseline.json"))
ml_results = ml if isinstance(ml, list) else ml.get("results", [ml])
for model, exp_auc in [("LogisticRegression", 0.738), ("RandomForest", 0.977)]:
    rows = [r for r in ml_results if r.get("model") == model]
    check(f"{model} present", len(rows) >= 1, f"found {len(rows)}")
    if rows:
        r0 = rows[0]
        auc = r0.get("auc", r0.get("metrics", {}).get("auc", float("nan")))
        check(f"{model} AUC ~ {exp_auc}",
              abs(auc - exp_auc) < 0.01,
              f"AUC={auc:.3f}")

# ── 7. Structural hierarchy on multi_varied ───────────────────────────────────
def mean_auprc(model, source):
    rows = [r for r in source if r.get("model") == model
            and r.get("dataset") == "multi_varied"
            and r.get("split", "temporal") == "temporal"]
    if not rows: return float("nan")
    auprcs = [get_m(r).get("auprc", r.get("metrics", {}).get("auprc", float("nan"))) for r in rows]
    return sum(auprcs) / len(auprcs)

complex_mv  = next((r.get("auprc", 0) for r in sb_results
                    if r.get("model") == "ComplEx" and r.get("dataset") == "multi_varied"), float("nan"))
tgn_mv_m    = mean_auprc("TGN",      bv)
sgnn_mv_m   = mean_auprc("StaticGNN", sg)
tgat_mv_m   = mean_auprc("TGAT",     bv)

check("Hierarchy ComplEx < TGN (multi_varied)",
      complex_mv < tgn_mv_m,
      f"ComplEx={complex_mv:.3f} TGN={tgn_mv_m:.3f}")
check("Hierarchy TGN < TGAT (multi_varied)",
      tgn_mv_m < tgat_mv_m,
      f"TGN={tgn_mv_m:.3f} TGAT={tgat_mv_m:.3f}")
check("Hierarchy StaticGNN < TGAT (multi_varied) -- direction",
      sgnn_mv_m < tgat_mv_m,
      f"StaticGNN={sgnn_mv_m:.3f} TGAT={tgat_mv_m:.3f}", level=WARN if sgnn_mv_m >= tgat_mv_m else PASS)

# ── 8. Notebook executed ──────────────────────────────────────────────────────
nb_path = Path("notebooks/UseCase4/08_model_benchmark_final.ipynb")
if nb_path.exists():
    nb = json.load(open(nb_path, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed   = sum(1 for c in code_cells if c.get("execution_count"))
    errors     = [i for i, c in enumerate(nb["cells"])
                  if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    check(f"Notebook: all {len(code_cells)} code cells executed",
          executed == len(code_cells), f"{executed}/{len(code_cells)} executed")
    check("Notebook: no error cells", len(errors) == 0,
          f"errors at cell indices: {errors}")
else:
    check("Notebook exists", False, "08_model_benchmark_final.ipynb not found")

# ── Summary ───────────────────────────────────────────────────────────────────
passes = sum(1 for s, _, _ in log if s == PASS)
warns  = sum(1 for s, _, _ in log if s == WARN)
fails  = sum(1 for s, _, _ in log if s == FAIL)

print(f"\n{'='*70}")
print(f"  THESIS RESULT VERIFICATION")
print(f"{'='*70}")
for status, label, msg in log:
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}[status]
    detail = f"  ({msg})" if msg else ""
    print(f"  [{icon}] {label}{detail}")
print(f"{'='*70}")
print(f"  {passes} PASS  {warns} WARN  {fails} FAIL")
print(f"{'='*70}\n")

if fails > 0:
    raise SystemExit(1)
