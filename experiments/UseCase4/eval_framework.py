"""
Unified evaluation framework for the TKG thesis benchmark.

Four split methods, identical interface:
    split_dataset(df, method) -> (train_df, val_df, test_df)

Methods
-------
stratified  : sklearn stratified 70/15/15, class-balanced across all splits
temporal    : chronological 70/15/15, no future leakage
6slot       : temporal + test divided into n_slots time windows (test_df gets 'slot' col)
inductive   : temporal + inductive_frac of worker nodes withheld from train
              (test_df gets 'is_new_node' col, mimics TGAT inductive protocol)

Usage example
-------------
    from experiments.UseCase4.eval_framework import split_dataset, compute_metrics

    train_df, val_df, test_df = split_dataset(df, method='temporal')
    metrics = compute_metrics(y_true, y_score)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

SPLIT_METHODS = ('stratified', 'temporal', '6slot', 'inductive')


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    method: str = 'temporal',
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    inductive_frac: float = 0.10,
    n_slots: int = 6,
    random_state: int = 42,
    label_col: str = 'label',
    time_col: str = 'tau',
    worker_col: str = 'src',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into (train_df, val_df, test_df).

    All splits are returned as independent copies (no shared index with df).
    The DataFrame is sorted by time_col before splitting.

    Special columns added:
      '6slot'     -> test_df gains 'slot' (int, 0-indexed)
      'inductive' -> train_df has holdout workers removed;
                     val_df and test_df gain 'is_new_node' (int 0/1)

    Parameters
    ----------
    df            : event DataFrame
    method        : 'stratified' | 'temporal' | '6slot' | 'inductive'
    train_frac    : fraction for training (default 0.70)
    val_frac      : fraction for validation (default 0.15)
    inductive_frac: share of unique worker nodes to hold out (inductive only)
    n_slots       : number of test time windows (6slot only)
    random_state  : reproducibility seed
    label_col     : binary violation label column name
    time_col      : temporal ordering column (numeric or datetime)
    worker_col    : column containing worker/source node IDs (inductive only)
    """
    if method not in SPLIT_METHODS:
        raise ValueError(f"method must be one of {SPLIT_METHODS}, got '{method}'")

    df = df.sort_values(time_col).reset_index(drop=True)

    if method == 'stratified':
        return _stratified_split(df, train_frac, val_frac, label_col, random_state)
    if method == 'temporal':
        return _temporal_split(df, train_frac, val_frac)
    if method == '6slot':
        return _sixslot_split(df, train_frac, val_frac, n_slots)
    # inductive
    return _inductive_split(df, train_frac, val_frac, inductive_frac,
                            random_state, worker_col)


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Find the decision threshold that maximises F1 on a *validation* set.
    Apply the returned threshold to the test set (never fit threshold on test).

    Why: at 1.5% positive rate, threshold=0.5 almost never fires, yielding F1≈0.
    The optimal threshold sits near the score percentile matching the positive rate.
    """
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    # prec/rec arrays have len = len(thresholds)+1; drop the last sentinel value
    denom = prec[:-1] + rec[:-1]
    with np.errstate(invalid='ignore'):
        f1s = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / denom, 0.0)
    return float(thresholds[np.argmax(f1s)]) if len(thresholds) > 0 else 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Standard metrics for imbalanced binary classification.

    Returns AUC, AUPRC, F1, precision, recall.
    Returns NaN for AUC/AUPRC when only one class is present in y_true.
    """
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred  = (y_score >= threshold).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1

    return {
        'auc':       float(roc_auc_score(y_true, y_score))          if has_both_classes else float('nan'),
        'auprc':     float(average_precision_score(y_true, y_score)) if has_both_classes else float('nan'),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'n_pos':     int(y_true.sum()),
        'n_total':   int(len(y_true)),
    }


def compute_slot_metrics(
    test_df: pd.DataFrame,
    y_score: np.ndarray,
    label_col: str = 'label',
    threshold: float = 0.5,
) -> dict:
    """
    Compute per-slot and overall metrics for the '6slot' split.

    test_df must have a 'slot' column (produced by split_dataset with method='6slot').
    y_score must be aligned with test_df's index.
    threshold should be the optimal value found on the validation set.
    """
    if 'slot' not in test_df.columns:
        raise ValueError("test_df must have a 'slot' column (use method='6slot')")

    results = {'per_slot': {}, 'overall': None}
    test_df = test_df.reset_index(drop=True)

    for slot in sorted(test_df['slot'].unique()):
        mask = (test_df['slot'] == slot).values
        slot_labels = test_df.loc[mask, label_col].values
        slot_scores = y_score[mask]
        results['per_slot'][int(slot)] = compute_metrics(slot_labels, slot_scores,
                                                          threshold=threshold)

    results['overall'] = compute_metrics(
        test_df[label_col].values, y_score, threshold=threshold)
    return results


def split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = 'label',
) -> None:
    """Print a one-line summary for each split partition."""
    for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
        n     = len(part)
        n_pos = int(part[label_col].sum()) if label_col in part.columns else -1
        rate  = 100 * n_pos / n if n > 0 else 0.0

        extras = []
        if 'slot' in part.columns:
            extras.append(f"slots={part['slot'].nunique()}")
        if 'is_new_node' in part.columns:
            extras.append(f"new_nodes={int(part['is_new_node'].sum())}")
        extra_str = '  ' + '  '.join(extras) if extras else ''

        print(f"  {name:5s}: {n:7,d} events  "
              f"violations={n_pos:4d} ({rate:.1f}%){extra_str}")


def save_results(results: dict, path: str | Path) -> None:
    """Persist benchmark results to JSON (NaN -> null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_clean(results), f, indent=2)
    print(f"Results saved -> {path}")


def load_results(path: str | Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Internal split implementations
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_split(df, train_frac, val_frac, label_col, random_state):
    test_frac = 1.0 - train_frac - val_frac
    idx       = np.arange(len(df))
    labels    = df[label_col].values

    tr_idx, tmp_idx = train_test_split(
        idx, test_size=(val_frac + test_frac),
        stratify=labels, random_state=random_state)

    val_rel = val_frac / (val_frac + test_frac)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=(1.0 - val_rel),
        stratify=labels[tmp_idx], random_state=random_state)

    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy(), df.iloc[te_idx].copy()


def _temporal_split(df, train_frac, val_frac):
    n      = len(df)
    tr_end = int(n * train_frac)
    va_end = int(n * (train_frac + val_frac))

    return (df.iloc[:tr_end].copy(),
            df.iloc[tr_end:va_end].copy(),
            df.iloc[va_end:].copy())


def _sixslot_split(df, train_frac, val_frac, n_slots):
    train_df, val_df, test_df = _temporal_split(df, train_frac, val_frac)
    test_df = test_df.copy()

    n_test = len(test_df)
    # Assign slots so each slot has equal number of events (quantile-based)
    slot_size  = n_test // n_slots
    remainder  = n_test - slot_size * n_slots
    slot_arr   = np.repeat(np.arange(n_slots), slot_size)
    if remainder > 0:
        slot_arr = np.concatenate([slot_arr, np.full(remainder, n_slots - 1)])
    test_df['slot'] = slot_arr

    return train_df, val_df, test_df


def _inductive_split(df, train_frac, val_frac, inductive_frac, random_state, worker_col):
    train_df, val_df, test_df = _temporal_split(df, train_frac, val_frac)

    rng            = np.random.RandomState(random_state)
    train_workers  = train_df[worker_col].unique()
    n_holdout      = max(1, int(len(train_workers) * inductive_frac))
    holdout_set    = set(rng.choice(train_workers, size=n_holdout, replace=False).tolist())

    # Remove holdout workers from training — they become "new" at inference
    train_df = train_df[~train_df[worker_col].isin(holdout_set)].copy()

    val_df   = val_df.copy()
    test_df  = test_df.copy()
    val_df['is_new_node']  = val_df[worker_col].isin(holdout_set).astype(int)
    test_df['is_new_node'] = test_df[worker_col].isin(holdout_set).astype(int)

    return train_df, val_df, test_df
