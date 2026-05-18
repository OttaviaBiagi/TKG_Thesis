"""Fix TypeError: train_sec None / 60 in notebook 08 static-baseline cells."""
import json, pathlib, re

NB = pathlib.Path(__file__).parents[2] / "notebooks/UseCase4/08_model_benchmark_final.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))

OLD = '        "train_min":  round(r["train_sec"] / 60, 1),\n'
NEW = '        "train_min":  round(r["train_sec"] / 60, 1) if r["train_sec"] is not None else None,\n'

patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    new_src = [NEW if line == OLD else line for line in src]
    if new_src != src:
        cell["source"] = new_src
        cell["outputs"] = []
        cell["execution_count"] = None
        patched += 1

print(f"Patched {patched} cell(s)")
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved.")
