"""
Hyperparameter tuning for TGN, DyRep, TGAT using Optuna.

Usage (from project root):
    python experiments/UseCase4/tune_models.py --model TGN --dataset single --n-trials 50
    python experiments/UseCase4/tune_models.py --model DyRep --dataset single --n-trials 50
    python experiments/UseCase4/tune_models.py --model TGAT --dataset single --n-trials 50
    python experiments/UseCase4/tune_models.py --all --n-trials 50

Saves best hyperparameters to:
    experiments/UseCase4/results/best_params.json

Tuning protocol:
  - Split: stratified 70/15/15 (fixed for all tuning runs, not a split variable)
  - Objective: maximize val AUPRC (robust for class imbalance at ~1%)
  - scaler: MinMaxScaler fitted on train partition
  - n_epochs: 20 (faster tuning); final training uses 30 in run_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from data_loader  import load_single_project, load_multi_project, FEAT_COLS
from eval_framework import split_dataset
from models import MODEL_REGISTRY

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
PARAMS_FILE = RESULTS_DIR / 'best_params.json'

# ── Per-model search spaces ───────────────────────────────────────────────────

def _suggest_tgn(trial):
    return {
        'embed_dim':          trial.suggest_categorical('embed_dim',  [32, 64, 128]),
        'memory_dim':         trial.suggest_categorical('memory_dim', [32, 64]),
        'message_dim':        trial.suggest_categorical('message_dim',[32, 64]),
        'lr':                 trial.suggest_float('lr',  1e-4, 1e-2, log=True),
        'pos_weight_factor':  trial.suggest_float('pos_weight_factor', 5.0, 30.0),
        'batch_size':         trial.suggest_categorical('batch_size', [128, 256, 512]),
        'n_epochs':           20,
    }

def _suggest_dyrep(trial):
    return {
        'embed_dim':          trial.suggest_categorical('embed_dim',    [16, 32, 64]),
        'n_neighbors':        trial.suggest_categorical('n_neighbors',  [5, 10, 20]),
        'lr':                 trial.suggest_float('lr',   1e-5, 1e-2, log=True),
        'intensity_reg':      trial.suggest_float('intensity_reg', 0.001, 0.1, log=True),
        'pos_weight_factor':  trial.suggest_float('pos_weight_factor', 5.0, 30.0),
        'batch_size':         trial.suggest_categorical('batch_size', [128, 256, 512]),
        'n_epochs':           20,
    }

def _suggest_tgat(trial):
    return {
        'embed_dim':          trial.suggest_categorical('embed_dim',   [32, 64, 128]),
        'n_heads':            trial.suggest_categorical('n_heads',     [2, 4, 8]),
        'n_neighbors':        trial.suggest_categorical('n_neighbors', [10, 20]),
        'lr':                 trial.suggest_float('lr',  1e-4, 1e-2, log=True),
        'pos_weight_factor':  trial.suggest_float('pos_weight_factor', 5.0, 30.0),
        'batch_size':         trial.suggest_categorical('batch_size', [256, 512]),
        'n_epochs':           20,
    }

SUGGEST = {'TGN': _suggest_tgn, 'DyRep': _suggest_dyrep, 'TGAT': _suggest_tgat}


# ── Objective factory ─────────────────────────────────────────────────────────

def _make_objective(model_name: str, train_df, val_df,
                    feat_cols: list[str], scaler, num_nodes: int, edge_dim: int):
    registry = MODEL_REGISTRY[model_name]

    def objective(trial):
        hparams = SUGGEST[model_name](trial)
        model   = registry['make'](num_nodes, edge_dim, **hparams).to(DEVICE)
        _, hist = registry['train'](model, train_df, val_df, feat_cols, scaler,
                                    device=DEVICE, **hparams)
        # Return best val AUPRC achieved during training
        valid = [v for v in hist if not (isinstance(v, float) and v != v)]
        return max(valid) if valid else 0.0

    return objective


# ── Main tuning function ──────────────────────────────────────────────────────

def tune(model_name: str, dataset: str, n_trials: int = 50,
         data_dir: str = 'data/UseCase4') -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print('Optuna not found — install with: pip install optuna')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'Tuning {model_name} on {dataset} dataset  ({n_trials} trials)')
    print(f'Device: {DEVICE}')
    print('='*60)

    # Load data
    df = (load_single_project(data_dir) if dataset == 'single'
          else load_multi_project(data_dir))
    num_nodes = df.attrs['num_nodes']
    edge_dim  = df.attrs['edge_dim']
    feat_cols = df.attrs.get('feat_cols', FEAT_COLS)

    # Fixed stratified split for tuning
    train_df, val_df, _ = split_dataset(df, method='stratified',
                                         label_col='label', time_col='tau')

    scaler = MinMaxScaler()
    scaler.fit(train_df[feat_cols].fillna(0).values)

    # Load existing partial results so callback can update incrementally
    existing_all = {}
    if PARAMS_FILE.exists():
        existing_all = json.load(open(PARAMS_FILE))
    key = f'{model_name}_{dataset}'

    def _save_callback(study, trial):
        """Save best params to disk after every completed trial."""
        if trial.state.name != 'COMPLETE':
            return
        try:
            best_t = study.best_trial
        except ValueError:
            return
        existing_all[key] = {
            'model':              model_name,
            'dataset':            dataset,
            'auprc':              best_t.value,
            'params':             best_t.params,
            'n_trials_completed': len([t for t in study.trials
                                       if t.state.name == 'COMPLETE']),
        }
        with open(PARAMS_FILE, 'w') as f:
            json.dump(existing_all, f, indent=2)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    obj   = _make_objective(model_name, train_df, val_df,
                             feat_cols, scaler, num_nodes, edge_dim)
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True,
                   catch=(Exception,), callbacks=[_save_callback])

    best = study.best_trial
    print(f'\nBest trial #{best.number}  val_AUPRC={best.value:.4f}')
    print(f'  Params: {best.params}')

    result = {
        'model':    model_name,
        'dataset':  dataset,
        'auprc':    best.value,
        'params':   best.params,
        'n_trials_completed': len([t for t in study.trials
                                   if t.state.name == 'COMPLETE']),
    }
    return result


def run_all(n_trials: int = 50, data_dir: str = 'data/UseCase4'):
    """Tune all three models on both datasets and save best_params.json."""
    existing = {}
    if PARAMS_FILE.exists():
        existing = json.load(open(PARAMS_FILE))

    for model_name in ('TGN', 'DyRep', 'TGAT'):
        for dataset in ('single', 'multi'):
            key = f'{model_name}_{dataset}'
            if key in existing:
                print(f'  Skipping {key} (already in best_params.json)')
                continue
            result = tune(model_name, dataset, n_trials, data_dir)
            existing[key] = result
            # Save incrementally after each study
            with open(PARAMS_FILE, 'w') as f:
                json.dump(existing, f, indent=2)
            print(f'  Saved {key} -> {PARAMS_FILE}')

    print(f'\nAll done. Best params -> {PARAMS_FILE}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Hyperparameter tuning for TKG benchmark')
    ap.add_argument('--model',    choices=['TGN', 'DyRep', 'TGAT'])
    ap.add_argument('--dataset',  choices=['single', 'multi'], default='single')
    ap.add_argument('--n-trials', type=int, default=50)
    ap.add_argument('--data-dir', default='data/UseCase4')
    ap.add_argument('--all',      action='store_true',
                    help='Tune all models on both datasets (ignores --model/--dataset)')
    args = ap.parse_args()

    if args.all:
        run_all(n_trials=args.n_trials, data_dir=args.data_dir)
    elif args.model:
        result = tune(args.model, args.dataset, args.n_trials, args.data_dir)
        existing = {}
        if PARAMS_FILE.exists():
            existing = json.load(open(PARAMS_FILE))
        key = f'{args.model}_{args.dataset}'
        existing[key] = result
        with open(PARAMS_FILE, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f'Saved {key} -> {PARAMS_FILE}')
    else:
        ap.print_help()
