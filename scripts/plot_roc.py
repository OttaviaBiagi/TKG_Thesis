"""
Standalone ROC plot — loads saved .npy probability files and saves the figure.
Run from repo root:  python scripts/plot_roc.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

EXP_DIR = Path('experiments/UseCase4')

lr_data  = np.load(EXP_DIR / 'exp_d_lr_probs.npy')
rf_data  = np.load(EXP_DIR / 'exp_d_rf_probs.npy')
tgn_data = np.load(EXP_DIR / 'exp_d_tgn_probs.npy')

y_true   = lr_data[:, 0].astype(int)
lr_prob  = lr_data[:, 1]
rf_prob  = rf_data[:, 1]
tgn_prob = tgn_data[:, 1]

lr_auc  = roc_auc_score(y_true, lr_prob)
rf_auc  = roc_auc_score(y_true, rf_prob)
tgn_auc = roc_auc_score(y_true, tgn_prob)

fpr_lr,  tpr_lr,  _ = roc_curve(y_true, lr_prob)
fpr_rf,  tpr_rf,  _ = roc_curve(y_true, rf_prob)
fpr_tgn, tpr_tgn, _ = roc_curve(y_true, tgn_prob)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#181825')

ax.plot(fpr_tgn, tpr_tgn, color='#cba6f7', lw=2,
        label=f'TGN cert-aware  (AUC={tgn_auc:.3f})')
ax.plot(fpr_lr,  tpr_lr,  color='#a6e3a1', lw=1.5, linestyle='--',
        label=f'Logistic Regression  (AUC={lr_auc:.3f})')
ax.plot(fpr_rf,  tpr_rf,  color='#fab387', lw=1.5, linestyle='--',
        label=f'Random Forest  (AUC={rf_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#6c7086', linestyle=':', lw=1, label='Random')

ax.set_xlabel('False Positive Rate', color='#cdd6f4')
ax.set_ylabel('True Positive Rate',  color='#cdd6f4')
ax.set_title('Violation Detection — ROC Curves\n(temporal split, cert-aware features)',
             color='#cdd6f4')
ax.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=9)
ax.tick_params(colors='#cdd6f4')
for sp in ['top', 'right']:   ax.spines[sp].set_visible(False)
for sp in ['bottom', 'left']: ax.spines[sp].set_color('#313244')

out = EXP_DIR / 'roc_all_models.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
print(f'  TGN  AUC = {tgn_auc:.4f}')
print(f'  LR   AUC = {lr_auc:.4f}')
print(f'  RF   AUC = {rf_auc:.4f}')
