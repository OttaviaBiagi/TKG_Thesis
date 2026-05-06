"""
Save captured outputs from Experiments H and I into nb06 cells 40 and 42.
"""
import json, subprocess, sys
from pathlib import Path

NB = Path('notebooks/UseCase4/06_tkg_models.ipynb')
nb = json.load(NB.open(encoding='utf-8'))

# Exp H output (captured manually from run)
OUT_H = """\
Test set: 8880 events | 51 violations (0.57%)
AUC-ROC : 0.933
AUC-PR  : 0.0627  (random baseline ~= 0.0057)

Optimal threshold (F1-beta=2): 0.449
  -> Precision=0.162  Recall=0.235  F1-beta=0.216

 Threshold  Precision   Recall       F1  F1-beta   #flagged
------------------------------------------------------------
      0.05      0.040    0.941    0.076    0.170       1205
      0.10      0.047    0.922    0.089    0.194       1009
      0.15      0.048    0.745    0.090    0.190        795
      0.20      0.050    0.667    0.092    0.191        686
      0.25      0.049    0.647    0.091    0.188        673
      0.30      0.050    0.529    0.092    0.183        535
      0.40      0.090    0.275    0.135    0.194        156
      0.50      0.143    0.196    0.165    0.182         70
      0.60      0.038    0.020    0.026    0.022         26

Saved: experiments/UseCase4/exp_h_rf_pr_threshold.png
Experiment H complete -- AUC-ROC=0.933  AUC-PR=0.0627  Optimal th=0.449 -> P=0.162 R=0.235
"""

# Exp I output (captured manually from run)
OUT_I = """\
Total workers : 50
Test ASSIGNED_TO events (tau>=12): 5845

Discipline-match rate (all ASSIGNED_TO): 1485/29150 = 5.1%
-> Workers assigned cross-discipline: no structural constraint in dataset.

       Strategy  Avg cands   Median  Recall@filter   MRR_random
-----------------------------------------------------------------
     Unfiltered         50       50          1.000       0.0200
     discipline        2.2        2          0.041       0.0185  (4% of 50)
           cert        4.8        5          0.185       0.0388  (10% of 50)
         consec       50.0       50          1.000       0.0200  (100% of 50)
            all        0.1        0          0.004       0.0364  (0% of 50)

Key finding: filtering reduces candidates by 90-95% but recall is low.
The correct worker is typically NOT in the filtered set.
Cause: ASSIGNED_TO is stochastic -- workers assigned regardless of discipline/cert.

Contrast with REQUIRES_PERMIT:
  step -> permit_type is deterministic (8 candidates, static mapping).
  TNTComplEx: MRR=0.401 >> random (0.125), H@10=1.000.

T-GQL consecutive-path filtering improves evaluation when candidate reduction
preserves recall (deterministic structural relations). For stochastic many-to-many
ASSIGNED_TO, no graph-structural model can significantly outperform random.

Experiment I (T-GQL consecutive-path analysis) complete
"""

def stream_output(text, exec_count):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [text]
    }

# Cell 40 = Exp H code (index in 0-based: cells 39=md_h, 40=code_h, 41=md_i, 42=code_i)
nb['cells'][40]['outputs'] = [stream_output(OUT_H, 1)]
nb['cells'][40]['execution_count'] = 1

nb['cells'][42]['outputs'] = [stream_output(OUT_I, 2)]
nb['cells'][42]['execution_count'] = 2

with NB.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Outputs saved to cells 40 (Exp H) and 42 (Exp I)')
print(f'Total cells: {len(nb["cells"])}')
