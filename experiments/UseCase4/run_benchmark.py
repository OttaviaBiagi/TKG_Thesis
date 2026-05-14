"""
Full benchmark runner — fills the models × splits × datasets matrix.

Usage (from project root):
    python experiments/UseCase4/run_benchmark.py
    python experiments/UseCase4/run_benchmark.py --dataset single
    python experiments/UseCase4/run_benchmark.py --model TGN --split temporal
    python experiments/UseCase4/run_benchmark.py --skip-tuning  (use defaults if no best_params.json)

Output:
    experiments/UseCase4/results/benchmark.json   — full results with per-slot detail
    experiments/UseCase4/results/benchmark.csv    — summary matrix (easy to paste into thesis)

The benchmark runs:
    3 models  x  4 splits  x  2 datasets  =  24 experiments

Split methods
-------------
stratified  : sklearn stratified 70/15/15 (class-balanced)
temporal    : chronological 70/15/15 (no future leakage)
6slot       : temporal + test split into 6 time windows
inductive   : temporal + 10% worker nodes withheld from training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from data_loader    import load_single_project, load_multi_project, load_multi_varied, FEAT_COLS
from eval_framework import (split_dataset, compute_metrics, find_best_threshold,
                             compute_slot_metrics, split_info, save_results)
from models import MODEL_REGISTRY

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
PARAMS_FILE = RESULTS_DIR / 'best_params.json'

SEED = 42

def _set_seed(seed: int = SEED):
    """Fix all random seeds before each experiment for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _predict(model_name: str, model, df: pd.DataFrame,
             feat_cols, scaler, device) -> tuple:
    """
    Return (y_true: np.ndarray, y_scores: np.ndarray) for df.
    Does NOT update model state (update=False / memory reset before scoring).
    """
    src = torch.tensor(df['src'].values, dtype=torch.long, device=device)
    dst = torch.tensor(df['dst'].values, dtype=torch.long, device=device)
    ef  = torch.tensor(
        np.nan_to_num(scaler.transform(df[feat_cols].fillna(0).values)).astype(np.float32),
        dtype=torch.float32, device=device)
    y   = df['label'].values.astype(int)

    if model_name == 'TGAT':
        tau_min   = getattr(model, '_tau_min',   0.0)
        tau_range = getattr(model, '_tau_range', 1.0)
        t = torch.tensor(
            ((df['tau'].values.astype(np.float64) - tau_min) / tau_range).astype(np.float32),
            dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(src[i:i+256], dst[i:i+256], ef[i:i+256], t[i:i+256],
                      update=False).cpu().numpy()
                for i in range(0, len(src), 256)])
    else:
        dt_max = getattr(model, '_dt_max', 1.0)
        ts     = df['tau'].values.astype(np.float64)
        dt_np  = (np.diff(ts, prepend=ts[0]) / dt_max).astype(np.float32)
        dt     = torch.tensor(dt_np, dtype=torch.float32, device=device).unsqueeze(1)
        if model_name == 'DyRep':
            model.eval(); model.reset()
            with torch.no_grad():
                scores = np.concatenate([
                    model(src[i:i+256], dst[i:i+256], ef[i:i+256], dt[i:i+256],
                          update=False)[0].cpu().numpy()
                    for i in range(0, len(src), 256)])
        else:  # TGN
            model.eval(); model.tgn.memory.reset()
            with torch.no_grad():
                scores = np.concatenate([
                    model(src[i:i+256], dst[i:i+256], ef[i:i+256], dt[i:i+256],
                          update=False).cpu().numpy()
                    for i in range(0, len(src), 256)])
    return y, scores


MODELS   = ('TGN', 'DyRep', 'TGAT')
SPLITS   = ('stratified', 'temporal', '6slot', 'inductive')
DATASETS = ('single', 'multi', 'multi_varied')

# ── Default hyperparameters (used when no tuning has been run) ────────────────

DEFAULTS = {
    'TGN':   {'embed_dim': 32, 'memory_dim': 32, 'message_dim': 32,
               'lr': 1e-3, 'pos_weight_factor': 10.0,
               'batch_size': 256, 'n_epochs': 30},
    'DyRep': {'embed_dim': 32, 'n_neighbors': 10,
               'lr': 1e-4, 'intensity_reg': 0.001, 'pos_weight_factor': 10.0,
               'batch_size': 256, 'n_epochs': 30},
    'TGAT':  {'embed_dim': 64, 'n_heads': 4, 'n_neighbors': 20,
               'lr': 1e-3, 'pos_weight_factor': 15.0,
               'batch_size': 512, 'n_epochs': 30},
}


def _load_best_params(model_name: str, dataset: str) -> dict:
    key = f'{model_name}_{dataset}'
    if PARAMS_FILE.exists():
        stored = json.load(open(PARAMS_FILE))
        if key in stored:
            p = stored[key]['params'].copy()
            p['n_epochs'] = 30   # tuning used 20 epochs; final run uses 30
            return p
    print(f'  [warn] No tuned params for {key}, using defaults')
    return DEFAULTS[model_name].copy()


# ── Single experiment ─────────────────────────────────────────────────────────

def run_one(model_name: str, split_method: str, df: pd.DataFrame,
            dataset_tag: str, seed: int = SEED) -> dict:
    """Run one (model, split, dataset, seed) cell and return metrics dict."""
    registry  = MODEL_REGISTRY[model_name]
    hparams   = _load_best_params(model_name, dataset_tag)
    num_nodes = df.attrs['num_nodes']
    edge_dim  = df.attrs['edge_dim']
    feat_cols = df.attrs.get('feat_cols', FEAT_COLS)

    _set_seed(seed)
    print(f'\n  [{model_name:6s}] [{split_method:10s}] [{dataset_tag:6s}] seed={seed}', end='  ')

    # ── Split ────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_dataset(
        df, method=split_method, label_col='label', time_col='tau')

    # Sanity check: abort if no positives in test
    if test_df['label'].sum() == 0:
        print('SKIP (0 violations in test)')
        return {'skipped': True, 'reason': 'no_positives_in_test'}

    # ── Scaler (fit on train only) ────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaler.fit(train_df[feat_cols].fillna(0).values)

    # ── Build + train ─────────────────────────────────────────────────────────
    model = registry['make'](num_nodes, edge_dim, **hparams).to(DEVICE)
    t0    = time.time()
    model, _ = registry['train'](model, train_df, val_df, feat_cols, scaler,
                                  device=DEVICE, **hparams)
    train_sec = time.time() - t0

    # ── Find optimal threshold on val set (avoids data leakage into test) ────────
    val_y, val_scores = _predict(model_name, model, val_df, feat_cols, scaler, DEVICE)
    t_star = find_best_threshold(val_y, val_scores)

    # ── Evaluate on test with optimal threshold ────────────────────────────────
    test_y, test_scores = _predict(model_name, model, test_df, feat_cols, scaler, DEVICE)

    if split_method == '6slot':
        metrics = compute_slot_metrics(test_df.reset_index(drop=True),
                                       test_scores, label_col='label',
                                       threshold=t_star)
    elif split_method == 'inductive':
        mask_new  = (test_df['is_new_node'] == 1).values
        mask_seen = (test_df['is_new_node'] == 0).values
        metrics_new  = (compute_metrics(test_y[mask_new],  test_scores[mask_new],
                                        threshold=t_star)
                        if mask_new.sum() > 0 and test_y[mask_new].sum() > 0
                        else {'skipped': True})
        metrics_seen = (compute_metrics(test_y[mask_seen], test_scores[mask_seen],
                                        threshold=t_star)
                        if mask_seen.sum() > 0 and test_y[mask_seen].sum() > 0
                        else {'skipped': True})
        metrics = {
            'overall':    compute_metrics(test_y, test_scores, threshold=t_star),
            'new_nodes':  metrics_new,
            'seen_nodes': metrics_seen,
        }
    else:
        metrics = compute_metrics(test_y, test_scores, threshold=t_star)

    # Remove raw scores from saved results (large arrays)
    def _drop_scores(m):
        if isinstance(m, dict):
            return {k: _drop_scores(v) for k, v in m.items() if k != 'scores'}
        return m

    result = {
        'model':        model_name,
        'split':        split_method,
        'dataset':      dataset_tag,
        'seed':         seed,
        'train_sec':    round(train_sec, 1),
        'n_train':      len(train_df),
        'n_val':        len(val_df),
        'n_test':       len(test_df),
        'n_pos_test':   int(test_df['label'].sum()),
        'threshold':    round(t_star, 4),
        'metrics':      _drop_scores(metrics),
        'hparams':      {k: v for k, v in hparams.items() if k != 'n_epochs'},
    }

    # Print summary line
    if split_method in ('6slot', 'inductive'):
        ov = metrics.get('overall', {})
        print(f"AUC={ov.get('auc', float('nan')):.3f}  "
              f"AUPRC={ov.get('auprc', float('nan')):.3f}  "
              f"F1={ov.get('f1', float('nan')):.3f}  "
              f"thr={t_star:.3f}  ({train_sec:.0f}s)")
    else:
        print(f"AUC={metrics.get('auc', float('nan')):.3f}  "
              f"AUPRC={metrics.get('auprc', float('nan')):.3f}  "
              f"F1={metrics.get('f1', float('nan')):.3f}  "
              f"thr={t_star:.3f}  ({train_sec:.0f}s)")

    return result


# ── Summary table helpers ─────────────────────────────────────────────────────

def _extract_auc(r: dict) -> float:
    m = r.get('metrics', {})
    if 'overall' in m:          return m['overall'].get('auc', float('nan'))
    if 'per_slot' in m:         return m.get('overall', {}).get('auc', float('nan'))
    return m.get('auc', float('nan'))

def _extract_auprc(r: dict) -> float:
    m = r.get('metrics', {})
    if 'overall' in m:          return m['overall'].get('auprc', float('nan'))
    return m.get('auprc', float('nan'))

def _extract_f1(r: dict) -> float:
    m = r.get('metrics', {})
    if 'overall' in m:          return m['overall'].get('f1', float('nan'))
    return m.get('f1', float('nan'))


def _print_matrix(results: list[dict]):
    """Pretty-print the benchmark matrix."""
    print('\n' + '=' * 90)
    print(f'{"Model":8s}  {"Split":12s}  {"Dataset":8s}  {"Seed":5s}  '
          f'{"AUC":6s}  {"AUPRC":6s}  {"F1":6s}  {"Thr":5s}  {"sec":5s}')
    print('-' * 90)
    for r in results:
        if r.get('skipped'):
            continue
        print(f'{r["model"]:8s}  {r["split"]:12s}  {r["dataset"]:8s}  '
              f'{r.get("seed", 42):5d}  '
              f'{_extract_auc(r):6.3f}  {_extract_auprc(r):6.3f}  '
              f'{_extract_f1(r):6.3f}  {r.get("threshold", 0.5):5.3f}  '
              f'{r["train_sec"]:5.0f}')
    print('=' * 90)


def _to_csv(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.get('skipped'):
            continue
        rows.append({
            'model':      r['model'],
            'split':      r['split'],
            'dataset':    r['dataset'],
            'seed':       r.get('seed', 42),
            'auc':        round(_extract_auc(r),   4),
            'auprc':      round(_extract_auprc(r), 4),
            'f1':         round(_extract_f1(r),    4),
            'threshold':  r.get('threshold', 0.5),
            'train_sec':  r['train_sec'],
            'n_train':    r['n_train'],
            'n_test':     r['n_test'],
            'n_pos_test': r['n_pos_test'],
        })
    return pd.DataFrame(rows)


# ── Main runner ───────────────────────────────────────────────────────────────

def run(models=MODELS, splits=SPLITS, datasets=DATASETS,
        data_dir: str = 'data/UseCase4',
        seeds: tuple = (SEED,)):
    seeds = list(seeds)
    print(f'TKG Benchmark Runner')
    print(f'  Device:   {DEVICE}')
    print(f'  Models:   {list(models)}')
    print(f'  Splits:   {list(splits)}')
    print(f'  Datasets: {list(datasets)}')
    print(f'  Seeds:    {seeds}')
    total = len(models) * len(splits) * len(datasets) * len(seeds)
    print(f'  Total experiments: {total}')

    # multi_varied results go to their own file to avoid overwriting single/multi results
    datasets_list = list(datasets)
    only_varied   = datasets_list == ['multi_varied'] or datasets_list == ['multi_varied']
    out_path      = (RESULTS_DIR / 'benchmark_varied.json'
                     if set(datasets_list) == {'multi_varied'}
                     else RESULTS_DIR / 'benchmark.json')
    print(f'  Output:   {out_path.name}')

    # Pre-load datasets
    dfs = {}
    for ds_name in datasets:
        print(f'\nLoading {ds_name} dataset...')
        if ds_name == 'single':
            dfs[ds_name] = load_single_project(data_dir)
        elif ds_name == 'multi':
            dfs[ds_name] = load_multi_project(data_dir)
        else:
            dfs[ds_name] = load_multi_varied(data_dir)

    results = []
    done    = 0
    t_start = time.time()

    for ds_name in datasets:
        df = dfs[ds_name]
        for model_name in models:
            for split_method in splits:
                for seed in seeds:
                    done += 1
                    print(f'\n[{done}/{total}]', end='')
                    r = run_one(model_name, split_method, df, ds_name, seed=seed)
                    results.append(r)

                    # Save incrementally
                    save_results({'results': results}, out_path)

    # Final save
    _print_matrix(results)
    csv_df   = _to_csv(results)
    csv_path = out_path.with_suffix('.csv')
    csv_df.to_csv(csv_path, index=False)

    elapsed = time.time() - t_start
    print(f'\nTotal time: {elapsed/60:.1f} min')
    print(f'Results -> {out_path}')
    print(f'CSV     -> {csv_path}')
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='TKG benchmark runner')
    ap.add_argument('--model',   nargs='+', choices=list(MODELS),
                    default=list(MODELS))
    ap.add_argument('--split',   nargs='+', choices=list(SPLITS),
                    default=list(SPLITS))
    ap.add_argument('--dataset', nargs='+', choices=list(DATASETS),
                    default=list(DATASETS))
    ap.add_argument('--data-dir', default='data/UseCase4')
    ap.add_argument('--seeds',   nargs='+', type=int, default=[SEED],
                    help='Random seeds to run (e.g. --seeds 42 43 44 for 3-seed stats)')
    args = ap.parse_args()

    run(models=args.model, splits=args.split,
        datasets=args.dataset, data_dir=args.data_dir,
        seeds=tuple(args.seeds))
