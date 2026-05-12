"""
ML Baseline runner — Logistic Regression and Random Forest on FEAT_COLS.

Usage (from project root):
    python3 experiments/UseCase4/run_ml_baseline.py

Compares non-temporal feature-only baselines against TGN on the same
temporal 70/15/15 split and evaluation protocol used in run_benchmark.py.
Results saved to experiments/UseCase4/results/ml_baseline.json

Why this matters: if LR/RF ≈ TGN then graph structure adds little.
If TGN >> LR/RF then temporal graph modelling is the key contribution.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_single_project, FEAT_COLS
from eval_framework import split_dataset, find_best_threshold, compute_metrics

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42


def run_baseline(model_name: str, clf, df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = split_dataset(
        df, method='temporal', label_col='label', time_col='tau')

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[FEAT_COLS].fillna(0).values)
    X_val   = scaler.transform(val_df[FEAT_COLS].fillna(0).values)
    X_test  = scaler.transform(test_df[FEAT_COLS].fillna(0).values)

    y_train = train_df['label'].values
    y_val   = val_df['label'].values
    y_test  = test_df['label'].values

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_sec = time.time() - t0

    # Find optimal threshold on val set
    val_scores  = clf.predict_proba(X_val)[:, 1]
    t_star      = find_best_threshold(y_val, val_scores)

    # Evaluate on test
    test_scores = clf.predict_proba(X_test)[:, 1]
    metrics     = compute_metrics(y_test, test_scores, threshold=t_star)

    prev = y_test.mean()
    lift = metrics['auprc'] / prev if prev > 0 else float('nan')

    print(f"  {model_name:<20} AUC={metrics['auc']:.3f}  "
          f"AUPRC={metrics['auprc']:.3f}  lift=x{lift:.1f}  "
          f"F1={metrics['f1']:.3f}  thr={t_star:.3f}  ({train_sec:.1f}s)")

    return {
        'model':        model_name,
        'split':        'temporal',
        'dataset':      'single',
        'type':         'ml_baseline',
        'train_sec':    round(train_sec, 2),
        'n_train':      len(train_df),
        'n_val':        len(val_df),
        'n_test':       len(test_df),
        'n_pos_test':   int(y_test.sum()),
        'prevalence':   round(float(prev), 4),
        'threshold':    round(t_star, 4),
        'auprc_lift':   round(lift, 2),
        'metrics':      metrics,
    }


def run(data_dir: str = 'data/UseCase4'):
    print('ML Baseline Runner')
    print(f'  Split: temporal 70/15/15  (same as TGN benchmark)')
    print(f'  Features: {FEAT_COLS}')
    print(f'  Threshold: optimised on val set (same protocol as TGN)')
    print()

    np.random.seed(SEED)
    df = load_single_project(data_dir)

    models = [
        ('LogisticRegression',
         LogisticRegression(class_weight='balanced', max_iter=1000,
                            random_state=SEED, C=1.0)),
        ('RandomForest',
         RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                max_depth=10, random_state=SEED, n_jobs=-1)),
    ]

    results = []
    for name, clf in models:
        r = run_baseline(name, clf, df)
        results.append(r)

    print()
    # Lift = AUPRC / test-set prevalence. Temporal split: n_pos=8, n_test=4373 → prev=0.0018
    print('=== Summary vs TGN temporal (AUC=0.985, AUPRC=0.178, lift=x98.9) ===')
    print(f"  {'Model':<22} {'AUC':>6}  {'AUPRC':>6}  {'Lift':>6}  {'F1':>6}")
    print('  ' + '-' * 55)
    tgn_ref = {'model': 'TGN (temporal GNN)', 'auc': 0.985,
               'auprc': 0.178, 'lift': 98.9, 'f1': 0.084}
    for r in [tgn_ref] + [{'model': r['model'],
                            'auc':   r['metrics']['auc'],
                            'auprc': r['metrics']['auprc'],
                            'lift':  r['auprc_lift'],
                            'f1':    r['metrics']['f1']} for r in results]:
        print(f"  {r['model']:<22} {r['auc']:>6.3f}  {r['auprc']:>6.3f}  "
              f"x{r['lift']:>5.1f}  {r['f1']:>6.3f}")

    out = RESULTS_DIR / 'ml_baseline.json'
    with open(out, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f'\nSaved -> {out}')
    return results


if __name__ == '__main__':
    run()
