import json, sys, statistics as st
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/UseCase4/07_tlogic_symbolic_reasoning.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Load hybrid results
hybrid = json.load(open('experiments/UseCase4/results/hybrid_ensemble.json'))
agg = hybrid['aggregate']

def _s(name, key):
    return f"{agg[name][key+'_mean']:.3f}±{agg[name][key+'_std']:.3f}"

md_cell = {
    "cell_type": "markdown",
    "id": "hybrid-ens-md",
    "metadata": {},
    "source": [
        "---\n",
        "## 16. Hybrid Ensemble — T-Logic R1 + TGAT (multi_varied)\n",
        "\n",
        "Four strategies evaluated on multi_varied temporal split (3 seeds: 42/43/44).\n",
        "201 test violations / 83,982 test events (base rate 0.24%).\n",
        "\n",
        "**Logic:**\n",
        "- *T-Logic R1*: deterministic rule, no training\n",
        "- *TGAT*: temporal attention GNN, trained on 70% of events\n",
        "- *Hybrid OR*: `R1 fires OR TGAT > threshold` — maximise recall\n",
        "- *Hybrid AND*: `R1 fires AND TGAT > threshold` — maximise precision\n",
        "\n",
        f"| Strategy | P | R | F1 | Note |\n",
        "|---|---|---|---|---|\n",
        f"| **T-Logic R1** | **{_s('T-Logic R1','precision')}** | **{_s('T-Logic R1','recall')}** | **{_s('T-Logic R1','f1')}** | No training — performance ceiling |\n",
        f"| TGAT alone | {_s('TGAT','precision')} | {_s('TGAT','recall')} | {_s('TGAT','f1')} | High variance; AUC=0.979±0.025 |\n",
        f"| Hybrid OR | {_s('Hybrid OR','precision')} | {_s('Hybrid OR','recall')} | {_s('Hybrid OR','f1')} | R=1.0 guaranteed; P < T-Logic |\n",
        f"| Hybrid AND | {_s('Hybrid AND','precision')} | {_s('Hybrid AND','recall')} | {_s('Hybrid AND','f1')} | P=1.0; R drops (TGAT misses ~33%) |\n",
        "\n",
        "**Conclusions:**\n",
        "1. T-Logic R1 is the **hard ceiling** — both hybrid variants are strictly worse on ≥1 metric.\n",
        "2. **Hybrid OR** guarantees R=1.0 (catches every violation T-Logic or TGAT sees) but inherits TGAT's false positives — useful only when T-Logic has FP (e.g., real-world with manager overrides).\n",
        "3. **Hybrid AND** achieves P=1.0 but TGAT misses ~33% of violations on average — unacceptable for safety-critical monitoring.\n",
        "4. **When is the hybrid valuable?** On real EPC data with managerial override exceptions (T-Logic FP ≈ 1–4%), Hybrid OR reduces false alarms while preserving recall. On the synthetic benchmark (T-Logic FP=0), it adds no value.\n",
        "5. **TGAT alone** shows high seed variance (P: 0.528–0.932) due to threshold sensitivity at 0.24% class imbalance. AUC=0.979±0.025 confirms good ranking; threshold calibration is the bottleneck.\n",
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "hybrid-ens-code",
    "metadata": {},
    "outputs": [],
    "source": """\

# ── 16.1 Load and display hybrid ensemble results ─────────────────────────────
import json as _json, statistics as _st
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

RESULTS = Path('../../experiments/UseCase4/results')
hybrid  = _json.load(open(RESULTS / 'hybrid_ensemble.json'))
agg     = hybrid['aggregate']

def _fmt(name, k):
    return f"{agg[name][k+'_mean']:.3f}±{agg[name][k+'_std']:.3f}"

print("Hybrid Ensemble — T-Logic R1 + TGAT  (multi_varied, temporal split, 3 seeds)")
print(f"Test: {hybrid['per_seed'][0]['n_test']:,} events / {hybrid['per_seed'][0]['n_pos_test']} violations")
print()
print(f"  {'Strategy':22s} {'P':>12} {'R':>12} {'F1':>12}")
print("  " + "─" * 62)
for name in ['T-Logic R1', 'TGAT', 'Hybrid OR', 'Hybrid AND']:
    print(f"  {name:22s} {_fmt(name,'precision'):>12} {_fmt(name,'recall'):>12} {_fmt(name,'f1'):>12}")
print()

# Per-seed detail
print("Per-seed breakdown:")
for r in hybrid['per_seed']:
    s = r['seed']
    tl = r['strategies']['T-Logic R1']
    tg = r['strategies']['TGAT']
    ho = r['strategies']['Hybrid OR']
    ha = r['strategies']['Hybrid AND']
    print(f"  seed={s}  thr={r['threshold']:.4f}")
    print(f"    T-Logic: P={tl['precision']:.3f} R={tl['recall']:.3f} F1={tl['f1']:.3f}  TP={tl['tp']}/FP={tl['fp']}/FN={tl['fn']}")
    print(f"    TGAT:    P={tg['precision']:.3f} R={tg['recall']:.3f} F1={tg['f1']:.3f}  TP={tg['tp']}/FP={tg['fp']}/FN={tg['fn']}  AUC={tg['auc']:.3f}")
    print(f"    OR:      P={ho['precision']:.3f} R={ho['recall']:.3f} F1={ho['f1']:.3f}")
    print(f"    AND:     P={ha['precision']:.3f} R={ha['recall']:.3f} F1={ha['f1']:.3f}")
print()
print("Conclusion: T-Logic R1 is the performance ceiling (P=R=F1=1.0).")
print("Hybrid OR guarantees R=1.0 but inherits TGAT FP — useful when T-Logic has overrides.")
print("Hybrid AND achieves P=1.0 but misses ~33% violations — unsafe for compliance monitoring.")
""".splitlines(keepends=True)
}

nb['cells'].append(md_cell)
nb['cells'].append(code_cell)

with open('notebooks/UseCase4/07_tlogic_symbolic_reasoning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Added §16 (2 cells). Total cells: {len(nb["cells"])}')
