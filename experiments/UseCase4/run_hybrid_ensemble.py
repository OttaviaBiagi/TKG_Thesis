"""
Hybrid Ensemble Experiment — T-Logic R1 + TGAT on multi_varied dataset.

Trains TGAT (temporal split, seeds 42/43/44), saves per-event test probabilities,
then evaluates four strategies:
  1. T-Logic R1 alone          (deterministic rule, no training)
  2. TGAT alone                (learned, optimal val threshold)
  3. Hybrid OR:  R1 fires OR TGAT > th     (maximize recall)
  4. Hybrid AND: R1 fires AND TGAT > th    (maximize precision)

Output:
    experiments/UseCase4/results/hybrid_ensemble.json
    experiments/UseCase4/results/hybrid_tgat_probs_mv_s{seed}.npy  (test scores)
    experiments/UseCase4/results/hybrid_r1_mv_s{seed}.npy          (R1 flags)
    experiments/UseCase4/results/hybrid_labels_mv_s{seed}.npy      (true labels)

Usage (from project root, tkg-env):
    python experiments/UseCase4/run_hybrid_ensemble.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import torch
from data_loader    import load_multi_varied, FEAT_COLS
from eval_framework import split_dataset, find_best_threshold
from models         import MODEL_REGISTRY

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
PARAMS_FILE = RESULTS_DIR / 'best_params.json'
SEEDS       = [42, 43, 44]


# ── Hyperparameters (use best_params if available) ────────────────────────────
TGAT_DEFAULTS = {'embed_dim': 64, 'n_heads': 4, 'n_neighbors': 20,
                 'lr': 1e-3, 'pos_weight_factor': 15.0,
                 'batch_size': 512, 'n_epochs': 30}

def _load_params() -> dict:
    if PARAMS_FILE.exists():
        stored = json.load(open(PARAMS_FILE))
        key = 'TGAT_multi_varied'
        if key in stored:
            p = stored[key]['params'].copy()
            p['n_epochs'] = 30
            print(f'  Using tuned params for {key}')
            return p
    print('  Using default TGAT params')
    return TGAT_DEFAULTS.copy()


# ── Load multi_varied with R1 flags ──────────────────────────────────────────
def load_multi_varied_with_r1(data_dir='data/UseCase4') -> pd.DataFrame:
    """
    Load multi_varied and add 'r1_fires' column.
    R1 fires = worker is missing at least one required cert at event time.
    This is computed in _events_to_df but not stored — we add it here.
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz
    from data_loader import (CERT_REQS, PERMIT_ENCODE,
                              _build_disc_encode, _build_worker_certs,
                              _add_node_indices)

    data_dir = Path(data_dir)
    proj_dir = data_dir / 'projects_varied'
    index    = _json.load(open(proj_dir / 'index.json'))

    all_dfs = []
    for entry in index:
        p_dir = proj_dir / entry['path']
        try:
            ds_p = _json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
            ev_p = _json.load(open(p_dir / 'events.json',  encoding='utf-8'))
        except Exception as exc:
            print(f'  Warning: skipping {entry["path"]} — {exc}')
            continue

        pid           = ds_p['project']['project_id']
        disc_encode   = _build_disc_encode(ds_p['steps'])
        worker_certs  = _build_worker_certs(ds_p['workers'])
        step_info     = {s['id']: s for s in ds_p['steps']}
        denied_set    = {(v['worker_id'], v['step_id']) for v in ev_p['permit_denied']}
        completed_map = {c['step_id']: c for c in ev_p['completed']}

        rc_str = ds_p['update_events'][0]['valid_from']
        rc_dt  = _dt.fromisoformat(rc_str)
        if rc_dt.tzinfo is None:
            rc_dt = rc_dt.replace(tzinfo=_tz.utc)

        records = []
        for e in ev_p['assigned_to']:
            wid = e['worker_id']; sid = e['step_id']
            d   = _dt.fromisoformat(e['date'])
            if d.tzinfo is None: d = d.replace(tzinfo=_tz.utc)

            step     = step_info.get(sid, {})
            permit   = step.get('permit_type', 'general_work')
            disc     = step.get('discipline', 'XX')
            after_rc = int(d >= rc_dt)

            req  = set(CERT_REQS.get(permit, set()))
            if after_rc and permit == 'hot_work':
                req.add('Advanced Fire Watch')

            wc      = worker_certs.get(wid, {})
            missing = [c for c in req if c not in wc or not (wc[c][0] <= d <= wc[c][1])]
            exp_soon = int(any((wc[c][1] - d).days < 30 for c in req if c in wc))
            comp  = completed_map.get(sid, {})

            records.append({
                'worker_id':         pid + ':' + wid,
                'step_id':           pid + ':' + sid,
                'tau':               float(d.timestamp()),
                'permit_enc':        PERMIT_ENCODE.get(permit, 0),
                'disc_enc':          disc_encode.get(disc, 0),
                'after_rc':          after_rc,
                'on_critical_path':  int(comp.get('on_critical_path', False)),
                'weight_pct':        float(comp.get('weight_pct', 0.0) or 0.0),
                'cert_expires_soon': exp_soon,
                'r1_fires':          int(len(missing) > 0),
                'label_viol':        int((wid, sid) in denied_set),
            })

        part = pd.DataFrame(records).sort_values('tau').reset_index(drop=True)
        part['label'] = part['label_viol']
        all_dfs.append(part)

    df = pd.concat(all_dfs, ignore_index=True).sort_values('tau').reset_index(drop=True)
    df, num_nodes = _add_node_indices(df)
    df.attrs['num_nodes'] = num_nodes
    df.attrs['edge_dim']  = 6   # len(FEAT_COLS)
    df.attrs['feat_cols'] = FEAT_COLS
    print(f'[varied+r1] {len(df):,} events  violations={int(df["label"].sum()):,}  '
          f'r1_fires={int(df["r1_fires"].sum()):,}  num_nodes={num_nodes}')
    return df


# ── TGAT predict ──────────────────────────────────────────────────────────────
def _predict_tgat(model, df, feat_cols, scaler):
    src = torch.tensor(df['src'].values, dtype=torch.long, device=DEVICE)
    dst = torch.tensor(df['dst'].values, dtype=torch.long, device=DEVICE)
    ef  = torch.tensor(
        np.nan_to_num(scaler.transform(df[feat_cols].fillna(0).values)).astype(np.float32),
        dtype=torch.float32, device=DEVICE)
    tau_min   = getattr(model, '_tau_min',   0.0)
    tau_range = getattr(model, '_tau_range', 1.0)
    t = torch.tensor(
        ((df['tau'].values.astype(np.float64) - tau_min) / tau_range).astype(np.float32),
        dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        scores = np.concatenate([
            model(src[i:i+512], dst[i:i+512], ef[i:i+512], t[i:i+512],
                  update=False).cpu().numpy()
            for i in range(0, len(src), 512)])
    return scores


# ── Metrics helper ────────────────────────────────────────────────────────────
def _metrics(y_true, y_pred, scores=None):
    tp = int(np.logical_and(y_pred == 1, y_true == 1).sum())
    fp = int(np.logical_and(y_pred == 1, y_true == 0).sum())
    fn = int(np.logical_and(y_pred == 0, y_true == 1).sum())
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = float(roc_auc_score(y_true, scores)) if scores is not None and len(np.unique(y_true)) > 1 else float('nan')
    return {'precision': round(p,4), 'recall': round(r,4), 'f1': round(f1,4),
            'auc': round(auc,4), 'tp': tp, 'fp': fp, 'fn': fn}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Device: {DEVICE}')
    print('Loading multi_varied with R1 flags...')
    df = load_multi_varied_with_r1('data/UseCase4')

    hparams = _load_params()
    registry = MODEL_REGISTRY['TGAT']
    results  = []

    # R1 alone on full dataset (reference)
    r1_all   = df['r1_fires'].values.astype(int)
    y_all    = df['label'].values.astype(int)

    def _set_seed(s):
        import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

    for seed in SEEDS:
        print(f'\n=== Seed {seed} ===')
        _set_seed(seed)

        train_df, val_df, test_df = split_dataset(
            df, method='temporal', label_col='label', time_col='tau')
        print(f'  Train={len(train_df):,}  Val={len(val_df):,}  Test={len(test_df):,}')
        print(f'  Test violations={int(test_df["label"].sum())}  '
              f'R1 test fires={int(test_df["r1_fires"].sum())}')

        scaler = MinMaxScaler()
        scaler.fit(train_df[FEAT_COLS].fillna(0).values)

        num_nodes = df.attrs['num_nodes']
        edge_dim  = df.attrs['edge_dim']

        print(f'  Training TGAT ({hparams["n_epochs"]} epochs)...')
        t0    = time.time()
        model = registry['make'](num_nodes, edge_dim, **hparams).to(DEVICE)
        model, _ = registry['train'](model, train_df, val_df, FEAT_COLS, scaler,
                                      device=DEVICE, **hparams)
        print(f'  Train time: {time.time()-t0:.0f}s')

        # Optimal threshold from val set
        val_y  = val_df['label'].values.astype(int)
        val_sc = _predict_tgat(model, val_df, FEAT_COLS, scaler)
        t_star = find_best_threshold(val_y, val_sc)
        print(f'  Optimal threshold: {t_star:.4f}')

        # Test predictions
        test_y   = test_df['label'].values.astype(int)
        test_r1  = test_df['r1_fires'].values.astype(int)
        test_sc  = _predict_tgat(model, test_df, FEAT_COLS, scaler)
        test_tgat = (test_sc >= t_star).astype(int)

        # Save raw arrays
        np.save(str(RESULTS_DIR / f'hybrid_tgat_probs_mv_s{seed}.npy'), test_sc)
        np.save(str(RESULTS_DIR / f'hybrid_r1_mv_s{seed}.npy'),         test_r1)
        np.save(str(RESULTS_DIR / f'hybrid_labels_mv_s{seed}.npy'),     test_y)

        # ── Evaluate 4 strategies ─────────────────────────────────────────────
        hybrid_or  = np.clip(test_r1 + test_tgat, 0, 1)
        hybrid_and = (test_r1.astype(bool) & test_tgat.astype(bool)).astype(int)

        strategies = {
            'T-Logic R1':   (test_r1,   None),
            'TGAT':         (test_tgat, test_sc),
            'Hybrid OR':    (hybrid_or, None),
            'Hybrid AND':   (hybrid_and, None),
        }

        print()
        print(f'  {"Strategy":20s} {"P":>7} {"R":>7} {"F1":>7} {"AUC":>7}  TP/FP/FN')
        print('  ' + '─' * 70)
        seed_results = {}
        for name, (y_pred, scores) in strategies.items():
            m = _metrics(test_y, y_pred, scores)
            print(f'  {name:20s} {m["precision"]:>7.3f} {m["recall"]:>7.3f} '
                  f'{m["f1"]:>7.3f} {m["auc"]:>7.3f}  '
                  f'{m["tp"]}/{m["fp"]}/{m["fn"]}')
            seed_results[name] = m

        # Also try AND with lower thresholds (sweep)
        print()
        print('  Hybrid AND threshold sweep:')
        for th in [0.1, 0.2, 0.3, 0.5]:
            tgat_th = (test_sc >= th).astype(int)
            h_and   = (test_r1.astype(bool) & tgat_th.astype(bool)).astype(int)
            m = _metrics(test_y, h_and)
            print(f'    AND th={th:.1f}: P={m["precision"]:.3f}  R={m["recall"]:.3f}  '
                  f'F1={m["f1"]:.3f}  TP={m["tp"]}/FP={m["fp"]}/FN={m["fn"]}')

        results.append({'seed': seed, 'n_test': len(test_df),
                         'n_pos_test': int(test_y.sum()),
                         'threshold': round(t_star, 4),
                         'strategies': seed_results})

    # ── Aggregate across seeds ────────────────────────────────────────────────
    import statistics as st
    strategy_names = list(results[0]['strategies'].keys())
    print('\n' + '═' * 72)
    print('AGGREGATE (3 seeds)')
    print('═' * 72)
    print(f'{"Strategy":20s} {"P mean±std":>14} {"R mean±std":>14} {"F1 mean±std":>14}')
    print('─' * 72)
    agg = {}
    for name in strategy_names:
        ps  = [r['strategies'][name]['precision'] for r in results]
        rs  = [r['strategies'][name]['recall']    for r in results]
        fs  = [r['strategies'][name]['f1']        for r in results]
        mp, sp = st.mean(ps), st.stdev(ps) if len(ps)>1 else 0
        mr, sr = st.mean(rs), st.stdev(rs) if len(rs)>1 else 0
        mf, sf = st.mean(fs), st.stdev(fs) if len(fs)>1 else 0
        print(f'{name:20s} {mp:>6.3f}±{sp:.3f}  {mr:>6.3f}±{sr:.3f}  {mf:>6.3f}±{sf:.3f}')
        agg[name] = {'precision_mean': round(mp,4), 'precision_std': round(sp,4),
                     'recall_mean':    round(mr,4), 'recall_std':    round(sr,4),
                     'f1_mean':        round(mf,4), 'f1_std':        round(sf,4)}

    out = {'experiment': 'hybrid_ensemble_tlogic_tgat_multi_varied',
           'dataset':    'multi_varied', 'split': 'temporal',
           'seeds': SEEDS, 'per_seed': results, 'aggregate': agg}
    out_path = RESULTS_DIR / 'hybrid_ensemble.json'
    json.dump(out, open(out_path, 'w'), indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
