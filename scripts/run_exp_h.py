"""Run Experiment H code extracted from nb06 cell 40."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path as P
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                             average_precision_score, f1_score,
                             precision_score, recall_score)

EXP_DIR = P('experiments/UseCase4')
data    = np.load(str(EXP_DIR / 'exp_d_rf_probs.npy'))
y_true  = data[:, 0].astype(int)
y_prob  = data[:, 1]

print(f'Test set: {len(y_true)} events | {y_true.sum()} violations ({y_true.mean()*100:.2f}%)')

prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
pr_auc  = average_precision_score(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)
print(f'AUC-ROC : {roc_auc:.3f}')
print(f'AUC-PR  : {pr_auc:.4f}  (random baseline ~= {y_true.mean():.4f})')
print()

beta = 2
f1b_arr = []
for th in thresholds:
    yp = (y_prob >= th).astype(int)
    p  = precision_score(y_true, yp, zero_division=0)
    r  = recall_score(y_true, yp, zero_division=0)
    f1b_arr.append((1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0.0)
f1b_arr  = np.array(f1b_arr)
best_idx = int(np.argmax(f1b_arr))
best_th  = float(thresholds[best_idx])
yp_opt   = (y_prob >= best_th).astype(int)
best_p   = precision_score(y_true, yp_opt, zero_division=0)
best_r   = recall_score(y_true, yp_opt, zero_division=0)
print(f'Optimal threshold (F1-beta={beta}): {best_th:.3f}')
print(f'  -> Precision={best_p:.3f}  Recall={best_r:.3f}  F1-beta={f1b_arr[best_idx]:.3f}')
print()

print(f'{"Threshold":>10} {"Precision":>10} {"Recall":>8} {"F1":>8} {"F1-beta":>8} {"#flagged":>10}')
print('-' * 60)
for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]:
    yp  = (y_prob >= th).astype(int)
    p   = precision_score(y_true, yp, zero_division=0)
    r   = recall_score(y_true, yp, zero_division=0)
    f1  = f1_score(y_true, yp, zero_division=0)
    f1b = (1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0
    mark = '  <- optimal' if abs(th - round(best_th, 2)) < 0.015 else ''
    print(f'{th:>10.2f} {p:>10.3f} {r:>8.3f} {f1:>8.3f} {f1b:>8.3f} {yp.sum():>10}{mark}')
print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1e1e2e')
for ax in [ax1, ax2]:
    ax.set_facecolor('#181825')
    ax.tick_params(colors='#cdd6f4')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color('#313244')
TX = '#cdd6f4'

ax1.plot(rec, prec, color='#fab387', lw=2, label=f'RF (AUC-PR={pr_auc:.3f})')
ax1.axhline(y_true.mean(), color='#6c7086', linestyle=':', lw=1,
            label=f'Random baseline ({y_true.mean():.3f})')
ax1.scatter([best_r], [best_p], color='#a6e3a1', s=80, zorder=5,
            label=f'Optimal th={best_th:.2f}\nP={best_p:.2f} R={best_r:.2f}')
ax1.set_xlabel('Recall', color=TX); ax1.set_ylabel('Precision', color=TX)
ax1.set_title('Precision-Recall -- Random Forest\n(temporal split, 51 violations/8,880)', color=TX)
ax1.legend(facecolor='#313244', labelcolor=TX, fontsize=9)

all_ths = np.linspace(0.01, 0.99, 300)
f1b_curve = []
for th in all_ths:
    yp  = (y_prob >= th).astype(int)
    p   = precision_score(y_true, yp, zero_division=0)
    r   = recall_score(y_true, yp, zero_division=0)
    f1b_curve.append((1 + beta**2) * p * r / (beta**2 * p + r) if (p + r) > 0 else 0)
ax2.plot(all_ths, f1b_curve, color='#cba6f7', lw=2)
ax2.axvline(best_th, color='#a6e3a1', linestyle='--', lw=1.5, label=f'Optimal th={best_th:.2f}')
ax2.axvline(0.50, color='#6c7086', linestyle=':', lw=1, label='Default th=0.50')
ax2.set_xlabel('Threshold', color=TX); ax2.set_ylabel(f'F1-beta (beta={beta})', color=TX)
ax2.set_title(f'F1-beta vs Threshold -- RF\n(beta={beta}: recall 2x precision)', color=TX)
ax2.legend(facecolor='#313244', labelcolor=TX, fontsize=9)

plt.tight_layout()
out = EXP_DIR / 'exp_h_rf_pr_threshold.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
print(f'Experiment H complete -- AUC-ROC={roc_auc:.3f}  AUC-PR={pr_auc:.4f}  '
      f'Optimal th={best_th:.3f} -> P={best_p:.3f} R={best_r:.3f}')
