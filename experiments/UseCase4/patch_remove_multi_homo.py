"""
Patch 08_model_benchmark_final.ipynb:
  - Cell 6 (& duplicate): replace single-vs-multi analysis with single-only
  - Cell 29: update setup — remove multi_results (omogeneo rimosso)
  - Cell 30: update chart  — single vs multi_varied only
"""
import json
from pathlib import Path

NB = Path("notebooks/UseCase4/08_model_benchmark_final.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


CELL_SINGLE_SUMMARY = [
    "import json, numpy as np, pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    'RESULTS = Path("../../experiments/UseCase4/results")\n',
    'raw = json.load(open(RESULTS / "benchmark.json"))["results"]\n',
    "\n",
    "def _get(r, key):\n",
    '    m = r["metrics"]\n',
    '    return m.get("overall", m).get(key, float("nan"))\n',
    "\n",
    "# Single-project results — temporal split only\n",
    "rows = []\n",
    "for r in raw:\n",
    '    if r.get("skipped") or r["split"] != "temporal" or r["dataset"] != "single": continue\n',
    "    rows.append({\n",
    '        "Model":   r["model"],\n',
    '        "AUC":     round(_get(r, "auc"),   3),\n',
    '        "AUPRC":   round(_get(r, "auprc"), 3),\n',
    '        "F1":      round(_get(r, "f1"),    3),\n',
    '        "n_pos":   r["n_pos_test"],\n',
    '        "n_test":  r["n_test"],\n',
    '        "seed":    r.get("seed", "\\u2014"),\n',
    "    })\n",
    "\n",
    "df_single = pd.DataFrame(rows)\n",
    'df_single["Prevalence"] = (df_single["n_pos"] / df_single["n_test"]).round(4)\n',
    'df_single["Lift"]       = (df_single["AUPRC"] / df_single["Prevalence"]).round(1)\n',
    "\n",
    'print("=== Single-Project Benchmark \\u2014 Temporal Split ===")\n',
    "print(f\"  {'Model':8s}  {'Seed':>4}  {'AUC':>6}  {'AUPRC':>6}  {'Lift':>6}  {'F1':>6}  {'n_pos':>6}\")\n",
    'print("  " + "-" * 55)\n',
    'for _, r in df_single.sort_values("AUPRC", ascending=False).iterrows():\n',
    "    lift_str = f\"\\u00d7{r['Lift']:.1f}\"\n",
    "    print(f\"  {r['Model']:8s}  {str(r['seed']):>4}  {r['AUC']:>6.3f}  \"\n",
    "          f\"{r['AUPRC']:>6.3f}  {lift_str:>6}  {r['F1']:>6.3f}  {r['n_pos']:>6}\")\n",
    "\n",
    "# Bar chart — AUC and AUPRC, mean across seeds\n",
    'models_order = ["TGN", "TGAT", "DyRep"]\n',
    'agg = df_single.groupby("Model")[["AUC","AUPRC"]].mean()\n',
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    'fig.suptitle("Single-Project Benchmark \\u2014 Neural Models, Temporal Split",\n',
    '             fontsize=11, fontweight="bold")\n',
    'colors_bar = ["#1565C0", "#FF9800", "#4CAF50"]\n',
    "\n",
    'for ax, metric in zip(axes, ["AUC", "AUPRC"]):\n',
    "    vals = [agg.loc[m, metric] if m in agg.index else float(\"nan\") for m in models_order]\n",
    "    bars = ax.bar(models_order, vals, color=colors_bar, edgecolor=\"white\",\n",
    "                  linewidth=1.2, alpha=0.88)\n",
    "    for bar, v in zip(bars, vals):\n",
    "        if not np.isnan(v):\n",
    "            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
    "                    f\"{v:.3f}\", ha=\"center\", va=\"bottom\", fontsize=9, fontweight=\"bold\")\n",
    "    ax.set_title(metric, fontsize=11)\n",
    "    ax.set_ylabel(metric)\n",
    "    ax.set_ylim(0, 1.15)\n",
    "    ax.axhline(0.5, color=\"gray\", linestyle=\"--\", alpha=0.4, linewidth=0.8)\n",
    "    ax.spines[[\"top\", \"right\"]].set_visible(False)\n",
    "\n",
    "plt.tight_layout()\n",
    'fig.savefig(RESULTS / "benchmark_single_neural.png", dpi=150, bbox_inches="tight")\n',
    "plt.show()\n",
    'print(f"\\nSaved \\u2192 results/benchmark_single_neural.png")\n',
]


CELL_29_SETUP = [
    "# Build per-dataset result dicts for chart cells (\\u00a76c: single vs multi_varied)\n",
    'raw_bm  = json.load(open(RESULTS / "benchmark.json"))["results"]\n',
    'raw_var = json.load(open(RESULTS / "benchmark_varied.json"))["results"]\n',
    "\n",
    "def get_temporal(result_dict, model):\n",
    '    """Return [AUC, AUPRC] for model, temporal split, seed 42."""\n',
    "    if model not in result_dict:\n",
    "        return [float('nan'), float('nan')]\n",
    "    r = result_dict[model]\n",
    '    m = r["metrics"]\n',
    "    return [m.get(\"auc\", float('nan')), m.get(\"auprc\", float('nan'))]\n",
    "\n",
    "def make_result_dict(dataset, raw):\n",
    "    d = {}\n",
    "    for r in raw:\n",
    '        if r.get("skipped") or r.get("split") != "temporal": continue\n',
    '        if r["dataset"] != dataset: continue\n',
    "        if r.get(\"seed\", 42) != 42: continue\n",
    '        m = r["metrics"].get("overall", r["metrics"])\n',
    '        d[r["model"]] = {"metrics": m}\n',
    "    return d\n",
    "\n",
    'single_results = make_result_dict("single",      raw_bm)\n',
    'varied_results = make_result_dict("multi_varied", raw_var)\n',
    "\n",
    "print(\"single_results models:\", list(single_results.keys()))\n",
    "print(\"varied_results models:\", list(varied_results.keys()))\n",
]


CELL_30_CHART = [
    "fig, axes = plt.subplots(1, 2, figsize=(10, 5))\n",
    "fig.suptitle(\n",
    "    'Single-Project vs Cross-Project Generalisation (\\u00a76c) \\u2014 temporal split, seed=42',\n",
    "    fontsize=11, fontweight='bold')\n",
    "\n",
    "dataset_labels = ['Single\\n(real TR)', 'Multi varied\\n(\\u00a76c)']\n",
    "colors = {'TGN': '#2196F3', 'TGAT': '#FF9800'}\n",
    "x, w  = np.arange(2), 0.32\n",
    "\n",
    "def vals_for(model, metric_idx):\n",
    "    return [get_temporal(r, model)[metric_idx]\n",
    "            for r in [single_results, varied_results]]\n",
    "\n",
    "for ax, mi, ylabel, title in [\n",
    "    (axes[0], 0, 'AUC',   'ROC-AUC'),\n",
    "    (axes[1], 1, 'AUPRC', 'AUPRC'),\n",
    "]:\n",
    "    for model, offset in zip(['TGN', 'TGAT'], [-w/2, w/2]):\n",
    "        vs   = vals_for(model, mi)\n",
    "        bars = ax.bar(x + offset, vs, w, label=model,\n",
    "                      color=colors[model], alpha=0.85,\n",
    "                      edgecolor='white', linewidth=0.5)\n",
    "        for bar, v in zip(bars, vs):\n",
    "            if not np.isnan(v):\n",
    "                ax.text(bar.get_x() + bar.get_width()/2,\n",
    "                        bar.get_height() + 0.01,\n",
    "                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)\n",
    "    ax.set_ylabel(ylabel, fontsize=10)\n",
    "    ax.set_xticks(x)\n",
    "    ax.set_xticklabels(dataset_labels, fontsize=9)\n",
    "    ax.set_ylim(0, 1.15)\n",
    "    ax.legend(fontsize=9)\n",
    "    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)\n",
    "    ax.set_title(title, fontsize=11)\n",
    "    ax.spines[['top', 'right']].set_visible(False)\n",
    "\n",
    "plt.tight_layout()\n",
    "out_fig = RESULTS / 'benchmark_varied_comparison.png'\n",
    "fig.savefig(out_fig, dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Saved -> {out_fig}')\n",
]


def replace_cell(idx, new_src_lines):
    cells[idx]["source"] = new_src_lines
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    print(f"  Patched cell [{idx}]")


fingerprints = {
    "single_summary_1": "Architectural Analysis: Why TGAT >> TGN on multi-project",
    "single_summary_2": "Architectural Analysis: Why TGAT >> TGN on multi-project",
    "setup_29":         "Build per-dataset result dicts for chart cells",
    "chart_30":         "Single vs Same-topology Multi",
}

found = {}
for i, c in enumerate(cells):
    src = "".join(c["source"])
    for name, fp in fingerprints.items():
        if fp in src and name not in found:
            found[name] = i

print("Found cells:", found)

seen = set()
for name, new_src in [
    ("single_summary_1", CELL_SINGLE_SUMMARY),
    ("single_summary_2", CELL_SINGLE_SUMMARY),
    ("setup_29",         CELL_29_SETUP),
    ("chart_30",         CELL_30_CHART),
]:
    if name in found and found[name] not in seen:
        replace_cell(found[name], new_src)
        seen.add(found[name])

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Notebook saved.")
