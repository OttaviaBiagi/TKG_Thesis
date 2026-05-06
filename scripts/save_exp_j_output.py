"""Save Experiment J captured output into nb06 cell 44."""
import json, base64
from pathlib import Path

NB  = Path('notebooks/UseCase4/06_tkg_models.ipynb')
nb  = json.load(NB.open(encoding='utf-8'))
IMG = Path('experiments/UseCase4/exp_j_delay_analysis.png')

OUT_J = """\
J1 — PRECEDES DAG integrity
  Cross-activity edges : 0   Self-loops: 0   Backward: 0   Cycles: False
  Steps/activity: min=1 max=14 mean=5.2 median=4

J2 — Accumulated delay
  Delayed steps: 16042/29150 (55%)  Mean: 14.8d  Max: 231d

J3 — Rule-change impact
  Total violations: 449  |  pre-RC: 175  RC-caused: 33  coincident: 241

Saved: experiments/UseCase4/exp_j_delay_analysis.png
Experiment J complete.
"""

img_data = base64.b64encode(IMG.read_bytes()).decode('ascii')

outputs = [
    {"output_type": "stream", "name": "stdout", "text": [OUT_J]},
    {
        "output_type": "display_data",
        "data": {"image/png": img_data, "text/plain": ["<Figure>"]},
        "metadata": {"image/png": {"width": 1600, "height": 1200}},
    },
]

# Cell 44 = Experiment J code (0-indexed)
nb['cells'][44]['outputs'] = outputs
nb['cells'][44]['execution_count'] = 1

with NB.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Output saved to cell 44 (Experiment J)')
print(f'Total cells: {len(nb["cells"])}')
