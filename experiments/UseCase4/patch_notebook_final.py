"""
Patch notebook 08:
  1. Remove duplicate static-KG cells (21-24, added by earlier patch scripts)
  2. Update §5b baseline table + chart (cells 17-18) with StaticGNN / TNTComplEx / corrected LR
  3. Update §5c to cover ComplEx + TNTComplEx + StaticGNN (cells 19-20)
  4. Update §6c findings from "pending" to actual results (cell 28 after deletion)
  5. Update §8 Key Findings (cell 36 after deletion)
  6. Fix multi-project framing in §6b (cell 23 after deletion)
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB = 'notebooks/UseCase4/08_model_benchmark_final.ipynb'
nb = json.load(open(NB, encoding='utf-8'))

# ── helpers ───────────────────────────────────────────────────────────────────
def _has_outputs(cell):
    return bool(cell.get('outputs')) or bool(cell.get('execution_count'))

def code(src, outputs=None):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': outputs or [], 'source': src}

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

# ── Step 1: delete duplicate cells 21-24 (safe: all have empty outputs) ──────
print('Cells before deletion:', len(nb['cells']))
for i in [24, 23, 22, 21]:   # delete highest index first to keep indices valid
    c = nb['cells'][i]
    if _has_outputs(c):
        print(f'  WARNING: cell {i} has outputs — skipping deletion')
    else:
        nb['cells'].pop(i)
        print(f'  Deleted cell {i} (duplicate static-KG)')
print('Cells after deletion:', len(nb['cells']))

# ── Step 2: cell 17 — §5b baseline table ─────────────────────────────────────
nb['cells'][17] = md(
    '## 5b. Baseline Comparison Analysis\n\n'
    'Academic standard: temporal GNN results must be compared against non-temporal baselines\n'
    '(Xu et al. 2020; Hamilton et al. 2017).\n\n'
    '> **Test-set note**: the single-project temporal split yields **8 test violations / 4,373 events**\n'
    '> (prevalence = 0.18%). With only 8 positives, AUPRC estimates are high-variance.\n'
    '> AUC is more stable here; the multi_varied dataset (201 violations) gives reliable AUPRC.\n\n'
    '> **Lift** uses test-set prevalence (0.18%, 8/4373), not overall dataset prevalence (1.54%).\n\n'
    '### All baselines — single project, temporal split\n\n'
    '| Model | Type | AUC | AUPRC | Lift | Recall | Notes |\n'
    '|-------|------|-----|-------|------|--------| ------|\n'
    '| StaticGNN (d=1) | Structure GNN | 0.759 | 0.498† | ×272† | 0.625 | val_AUPRC=0.068 — unstable |\n'
    '| **TGN** | Temporal GNN | **0.985** | **0.178** | **×98.9** | **1.000** | Best; catches all 8 violations |\n'
    '| Logistic Regression | Feature-only | 0.738 | 0.161 | ×88.4 | 0.625 | |\n'
    '| Random Forest | Feature-only | 0.978 | 0.160 | ×87.8 | 0.125 | |\n'
    '| TGAT | Temporal GNN | 0.822 | 0.046 | ×25.6 | 0.250 | Low AUPRC due to 8-violation noise |\n'
    '| TNTComplEx | Time-aware KG | 0.582 | 0.003 | ×1.6 | — | Time embedding; still ≈ random |\n'
    '| DyRep | Temporal GNN | 0.416 | 0.002 | ×1.1 | 1.000‡ | Degenerate threshold |\n'
    '| ComplEx | Static KG | 0.440 | 0.002 | ×1.0 | — | No temporal information |\n'
    '| Random | Baseline | 0.500 | 0.002 | ×1.0 | — | |\n\n'
    '† StaticGNN single: val_AUPRC=0.068 vs test_AUPRC=0.498 — high-variance artefact of 8 test violations.\n'
    'Multi_varied (201 violations): StaticGNN ×147.6 < TGAT ×309.0 — correct ordering confirmed.\n\n'
    '‡ DyRep recall=1.0 is degenerate: val-tuned threshold ≈ 0, flagging all events as violations (precision=0.002).\n\n'
    '### Structural hierarchy (multi_varied, 201 test violations — reliable)\n\n'
    '| Layer | Model | AUPRC | Lift | What it adds |\n'
    '|-------|-------|-------|------|--------------|\n'
    '| Static KG embedding | ComplEx / TNTComplEx | 0.002 | ×1.0 | Nothing — random |\n'
    '| Structure aggregation | StaticGNN | 0.353 | ×147.6 | Graph neighbourhood context |\n'
    '| Structure + time | TGAT | 0.646 | ×309.0 | Temporal dynamics |\n\n'
    'The ×147.6 → ×309.0 gap quantifies the added value of temporal modelling over structure alone.\n'
)

# ── Step 3: cell 18 — comparison chart code ──────────────────────────────────
nb['cells'][18] = code(
    'import json\n'
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'from pathlib import Path\n'
    'from matplotlib.patches import Patch\n'
    '\n'
    'RESULTS = Path("../../experiments/UseCase4/results")\n'
    '\n'
    'ml_raw  = json.load(open(RESULTS / "ml_baseline.json"))["results"]\n'
    'sb_raw  = json.load(open(RESULTS / "static_baseline.json"))["results"]\n'
    'gnn_raw = json.load(open(RESULTS / "static_gnn.json"))["results"]\n'
    '\n'
    'TEST_PREV = 8 / 4373   # single-project temporal split\n'
    '\n'
    '# ── build comparison table ──────────────────────────────────────────────\n'
    'comparison = [\n'
    '    {"Model": "TGN",   "Type": "Temporal GNN",   "AUC": 0.985, "AUPRC": 0.178, "F1": 0.084},\n'
    '    {"Model": "TGAT",  "Type": "Temporal GNN",   "AUC": 0.822, "AUPRC": 0.046, "F1": 0.129},\n'
    '    {"Model": "DyRep", "Type": "Temporal GNN",   "AUC": 0.416, "AUPRC": 0.002, "F1": 0.004},\n'
    ']\n'
    'for r in ml_raw:\n'
    '    m = r["metrics"]\n'
    '    comparison.append({"Model": r["model"], "Type": "Feature-only ML",\n'
    '                        "AUC": round(m["auc"],3), "AUPRC": round(m["auprc"],3), "F1": round(m["f1"],3)})\n'
    'for r in sb_raw:\n'
    '    if r["dataset"] == "single":\n'
    '        m = r["metrics"]\n'
    '        comparison.append({"Model": r["model"], "Type": "Static KG",\n'
    '                            "AUC": round(m["auc"],3), "AUPRC": round(m["auprc"],3), "F1": round(m["f1"],3)})\n'
    'for r in gnn_raw:\n'
    '    if r["dataset"] == "single":\n'
    '        m = r["metrics"]\n'
    '        comparison.append({"Model": f\'StaticGNN(d={r["best_depth"]})\', "Type": "Static GNN",\n'
    '                            "AUC": round(m["auc"],3), "AUPRC": round(m["auprc"],3), "F1": round(m["f1"],3)})\n'
    'comparison.append({"Model": "Random", "Type": "Baseline", "AUC": 0.500, "AUPRC": round(TEST_PREV,4), "F1": 0.0})\n'
    '\n'
    'df_cmp = pd.DataFrame(comparison).sort_values("AUPRC", ascending=False)\n'
    'df_cmp["Lift"] = (df_cmp["AUPRC"] / TEST_PREV).round(1)\n'
    'print("=== Single-project Baseline Comparison (temporal split, 8 test violations) ===")\n'
    'print(f"Test-set prevalence: {TEST_PREV*100:.3f}%  (note: AUPRC estimates high-variance with 8 violations)")\n'
    'print()\n'
    'print(df_cmp[["Model","Type","AUC","AUPRC","Lift","F1"]].to_string(index=False))\n'
    '\n'
    '# ── bar chart ─────────────────────────────────────────────────────────────\n'
    'palette = {"Temporal GNN":"#2196F3","Feature-only ML":"#FF9800",\n'
    '           "Static KG":"#9C27B0","Static GNN":"#00BCD4","Baseline":"#9E9E9E"}\n'
    'colors = [palette[t] for t in df_cmp["Type"]]\n'
    '\n'
    'fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n'
    'fig.suptitle("All Baselines — Single Project, Temporal Split (8 test violations)",\n'
    '             fontsize=11, fontweight="bold")\n'
    'for ax, (col, ylabel, ymin, ymax) in zip(axes, [\n'
    '    ("AUC",  "AUC",        0.35, 1.05),\n'
    '    ("AUPRC","AUPRC",      0.00, 0.60),\n'
    '    ("Lift", "AUPRC Lift", 0.0,  340.),\n'
    ']):\n'
    '    bars = ax.bar(df_cmp["Model"], df_cmp[col], color=colors, edgecolor="white", linewidth=1.2)\n'
    '    for bar, val in zip(bars, df_cmp[col]):\n'
    '        label = f"x{val:.0f}" if col=="Lift" else f"{val:.3f}"\n'
    '        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ymax*0.02,\n'
    '                label, ha="center", va="bottom", fontsize=7, fontweight="bold")\n'
    '    ax.set_ylim(ymin, ymax); ax.set_title(ylabel); ax.set_ylabel(ylabel)\n'
    '    ax.tick_params(axis="x", rotation=40)\n'
    '    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)\n'
    'fig.legend(handles=[Patch(facecolor=c,label=l) for l,c in palette.items()],\n'
    '           loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5,-0.04))\n'
    'plt.tight_layout(rect=[0,0.05,1,1])\n'
    'fig.savefig(RESULTS/"benchmark_baseline_comparison.png", dpi=150, bbox_inches="tight")\n'
    'plt.show()\n'
    'print("Saved -> results/benchmark_baseline_comparison.png")\n'
)
nb['cells'][18]['outputs'] = []

# ── Step 4: cells 19-20 — §5c all static baselines ───────────────────────────
nb['cells'][19] = md(
    '### §5c. All Static Baselines — ComplEx, TNTComplEx, StaticGNN\n\n'
    '**ComplEx** (Trouillon et al. 2016): static score = Re(⟨e_h, w_r, conj(e_t)⟩). '
    'Trained on compliant triples only. No timestamp → cannot distinguish violation from compliance.\n\n'
    '**TNTComplEx** (Lacroix et al. 2020): adds time embedding e_τ. '
    'Marginal improvement (AUC=0.582 on single) but AUPRC≈0.002 = random. '
    'Time bins too coarse; no persistent memory.\n\n'
    '**StaticGNN** (Kipf & Welling 2017): GCN + MLP edge classifier. '
    'Graph structure aggregation without temporal order. '
    'Single-project result (AUPRC=0.498) is high-variance (8 test violations; val_AUPRC=0.068). '
    'Multi_varied result (AUPRC=0.353, ×147.6) is the reliable estimate.\n\n'
    '| Model | Dataset | AUC | AUPRC | Lift | n_pos_test |\n'
    '|-------|---------|-----|-------|------|------------|\n'
    '| ComplEx | single | 0.440 | 0.002 | ×1.0 | 8 |\n'
    '| ComplEx | multi | 0.503 | 0.002 | ×1.0 | 933 |\n'
    '| ComplEx | multi_varied | 0.521 | 0.002 | ×1.0 | 201 |\n'
    '| TNTComplEx | single | 0.582 | 0.003 | ×1.6 | 8 |\n'
    '| TNTComplEx | multi | 0.507 | 0.002 | ×1.0 | 933 |\n'
    '| TNTComplEx | multi_varied | 0.516 | 0.002 | ×1.0 | 201 |\n'
    '| StaticGNN (d=1) | single | 0.759 | 0.498† | ×272† | 8 |\n'
    '| StaticGNN (d=2) | multi_varied | 0.930 | **0.353** | **×147.6** | 201 |\n\n'
    '† Unreliable: val_AUPRC=0.068 vs test_AUPRC=0.498. Multi_varied is the correct estimate.\n'
)

nb['cells'][20] = code(
    'import json, numpy as np, pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'from pathlib import Path\n'
    '\n'
    'RESULTS = Path("../../experiments/UseCase4/results")\n'
    'sb_all  = json.load(open(RESULTS / "static_baseline.json"))["results"]\n'
    'gnn_all = json.load(open(RESULTS / "static_gnn.json"))["results"]\n'
    '\n'
    'prev_map = {"single": 8/4373, "multi": 933/437250, "multi_varied": 201/83982}\n'
    '\n'
    'rows = []\n'
    'for r in sb_all:\n'
    '    m = r["metrics"]\n'
    '    prev = prev_map.get(r["dataset"], 0.015)\n'
    '    rows.append({"Model": r["model"], "Dataset": r["dataset"],\n'
    '                 "n_pos": r["n_pos_test"], "AUC": round(m["auc"],3),\n'
    '                 "AUPRC": round(m["auprc"],3), "Lift": round(m["auprc"]/prev,1),\n'
    '                 "F1": round(m["f1"],3)})\n'
    'for r in gnn_all:\n'
    '    m = r["metrics"]\n'
    '    prev = prev_map.get(r["dataset"], 0.015)\n'
    '    note = " (val=0.068)" if r["dataset"]=="single" else ""\n'
    '    rows.append({"Model": f\'StaticGNN(d={r["best_depth"]})\', "Dataset": r["dataset"],\n'
    '                 "n_pos": r["n_pos_test"], "AUC": round(m["auc"],3),\n'
    '                 "AUPRC": round(m["auprc"],3), "Lift": round(m["auprc"]/prev,1),\n'
    '                 "F1": round(m["f1"],3)})\n'
    '\n'
    'df_sb = pd.DataFrame(rows)\n'
    'print("All static baselines — temporal split, seed=42")\n'
    'print(df_sb[["Model","Dataset","n_pos","AUC","AUPRC","Lift","F1"]].to_string(index=False))\n'
    'print()\n'
    'print("Conclusion:")\n'
    'print("  ComplEx / TNTComplEx: AUPRC = prevalence at ALL scales => random")\n'
    'print("  StaticGNN single:     AUPRC=0.498 (unreliable, 8 violations)")\n'
    'print("  StaticGNN multi_v:    AUPRC=0.353 (x147.6) — structure adds signal")\n'
    'print("  TGAT multi_v:         AUPRC=0.646 (x309.0) — time doubles the lift")\n'
)
nb['cells'][20]['outputs'] = []

# ── Step 5: after deletion, cell 32 (§6c findings) is now at index 28 ────────
# New indices after deleting cells 21-24:
#   old 25 → new 21  (inductive markdown)
#   old 26 → new 22  (inductive code)
#   old 27 → new 23  (§6b markdown)
#   old 28 → new 24  (multi code)
#   old 29 → new 25  (§6c motivation markdown)
#   old 30 → new 26  (§6c benchmark code)
#   old 31 → new 27  (comparison chart)
#   old 32 → new 28  (§6c findings — currently "pending")
#   old 33 → new 29  (§7 temporal stability)
#   old 40 → new 36  (§8 key findings)

# Fix §6b multi framing (new index 23)
c23 = nb['cells'][23]
if 'source' in c23:
    src = ''.join(c23['source']) if isinstance(c23['source'], list) else c23['source']
    src = src.replace(
        'sharing the same EPC graph topology (identical step node IDs)',
        'using the same EPC process structure (node IDs scoped per project: no cross-project sharing at model level)'
    ).replace(
        'All 100 projects share the same step node IDs',
        'All 100 projects use the same EPC activity codes, but node IDs are scoped per project (P{proj}:step)'
    )
    nb['cells'][23]['source'] = src

# Update §6c findings (new index 28)
nb['cells'][28] = md(
    '### §6c Findings — Varied Multi-Project Results\n\n'
    '**TGAT** generalises best across 30 structurally diverse EPC families '
    '(AUPRC=0.646, lift=×309.0, F1=0.603).\n\n'
    '**StaticGNN** (depth=2) provides a meaningful structure-only baseline: '
    'AUPRC=0.353, lift=×147.6. The gap from ×147.6 to ×309.0 **quantifies the value of temporal modelling** '
    'over graph structure alone.\n\n'
    '**ComplEx / TNTComplEx** remain random (AUPRC=0.002) — confirmed at all scales.\n\n'
    '| Model | Type | AUC | AUPRC | Lift | F1 |\n'
    '|-------|------|-----|-------|------|----|\n'
    '| **TGAT** | Temporal GNN | **0.992** | **0.646** | **×309.0** | 0.603 |\n'
    '| StaticGNN (d=2) | Structure GNN | 0.930 | 0.353 | ×147.6 | 0.091 |\n'
    '| TGN | Temporal GNN | — | — | — | — |\n'
    '| ComplEx | Static KG | 0.521 | 0.002 | ×1.0 | 0.005 |\n'
    '| TNTComplEx | Time-aware KG | 0.516 | 0.002 | ×1.0 | 0.005 |\n\n'
    '*TGN multi_varied: run `python experiments/UseCase4/run_benchmark.py --model TGN --dataset multi_varied` to add.*\n\n'
    '**Design note**: node IDs are scoped per project (`V{proj}:step`, `V{proj}:worker`); '
    'no cross-project node sharing occurs. Results reflect genuine generalisation across different '
    'EPC structural families.\n'
)

# Update §8 Key Findings (new index 36)
nb['cells'][36] = md(
    '## 8. Key Findings\n\n'
    '### 8.1 Model Ranking\n\n'
    'Primary benchmark: single-project temporal split (AUC is the reliable metric here; '
    'AUPRC with 8 test violations is high-variance).\n\n'
    '1. **TGN — best overall** (AUC=0.985, AUPRC=0.178, lift=×98.9, recall=1.0)\n'
    '   Persistent memory captures worker certificate history. Only model with recall=1.0.\n\n'
    '2. **Random Forest** (AUC=0.978, AUPRC=0.160, lift=×87.8) — nearly matches TGN\n'
    '   6 edge features already encode most violation signal. TGN adds +10% AUPRC and recall=1.0.\n\n'
    '3. **TGAT** (AUC=0.822, AUPRC=0.046 on single — unreliable with 8 violations)\n'
    '   True performance visible on multi_varied: AUPRC=0.646 (×309.0) — best at scale.\n\n'
    '4. **DyRep** — degenerate failure at 1.5% imbalance. Retained as negative result.\n\n'
    '### 8.2 Structural Hierarchy (multi_varied, 201 violations — reliable)\n\n'
    '```\n'
    'ComplEx / TNTComplEx  AUPRC=0.002  ×  1.0  (static KG = random)\n'
    'StaticGNN (d=2)       AUPRC=0.353  ×147.6  (structure alone)\n'
    'TGAT                  AUPRC=0.646  ×309.0  (structure + temporal)\n'
    '```\n\n'
    'The ×147.6 → ×309.0 gap is the empirical contribution of temporal modelling.\n\n'
    '### 8.3 Multi-Project Evaluation\n\n'
    '- **Scalability (100 identical instances)**: TGAT ×454.8 / TGN ×44.8. '
    'TGAT attention accumulates per-step risk across repeated structures; '
    'TGN memory degrades when repeated event patterns write conflicting state.\n'
    '- **Generalisation (30 diverse families)**: TGAT ×309.0 — temporal attention '
    'generalises to structurally new EPC families. StaticGNN ×147.6 confirms that '
    'graph structure alone captures half the signal without any temporal information.\n\n'
    '### 8.4 Static Baseline Verdict\n\n'
    'ComplEx and TNTComplEx are random (AUPRC = prevalence) at every scale. '
    'The same (worker, step, relation) triple can be compliant or a violation '
    'depending only on the timestamp. Static and time-binned embeddings '
    'cannot capture this without persistent memory — confirming the core thesis '
    'that temporal context is the essential differentiator.\n'
)

# ── Step 6: clean all cells ────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell.setdefault('execution_count', None)
        cell.setdefault('outputs', [])
    elif cell['cell_type'] == 'markdown':
        cell.pop('outputs', None)
        cell.pop('execution_count', None)

json.dump(nb, open(NB, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'\nDone. Notebook has {len(nb["cells"])} cells.')
print('Updated: cells 17-20 (§5b-§5c baselines), 28 (§6c findings), 36 (§8 key findings)')
print('Deleted: old duplicate cells 21-24')
