"""
Insert missing setup cell before the benchmark chart cell in notebook 08.
The chart uses single_results / multi_results / varied_results / get_temporal
but those were never defined. This patch inserts a setup code cell immediately
before the chart cell.
"""
import json, pathlib

NB = pathlib.Path(__file__).parents[2] / "notebooks/UseCase4/08_model_benchmark_final.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

CHART_MARKER = "dataset_labels = ['Single"

SETUP_SOURCE = """\
# Build per-dataset result dicts for chart cells (§6b/§6c comparison)
raw_bm = json.load(open(RESULTS / "benchmark.json"))["results"]

def get_temporal(result_dict, model):
    \"\"\"Return [AUC, AUPRC] for model from a per-dataset result dict (temporal split).\"\"\"
    if model not in result_dict:
        return [float('nan'), float('nan')]
    r = result_dict[model]
    return [r["metrics"]["auc"], r["metrics"]["auprc"]]

def make_result_dict(dataset):
    d = {}
    for r in raw_bm:
        if r["dataset"] == dataset and r["split"] == "temporal":
            d[r["model"]] = r
    return d

single_results  = make_result_dict("single")
multi_results   = make_result_dict("multi")
varied_results  = make_result_dict("multi_varied")
print("Loaded:", {k: list(v.keys()) for k, v in [
    ("single", single_results), ("multi", multi_results), ("varied", varied_results)]})
"""

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": SETUP_SOURCE.splitlines(keepends=True),
}

chart_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if CHART_MARKER in src:
        chart_idx = i
        break

if chart_idx is None:
    print("ERROR: chart cell not found — check CHART_MARKER string")
else:
    nb["cells"].insert(chart_idx, setup_cell)
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted setup cell at index {chart_idx} (before chart cell now at {chart_idx+1})")
    print("Done.")
