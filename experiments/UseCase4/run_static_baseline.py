"""
Static KG Baselines — ComplEx and TNTComplEx for EPC violation detection.

Protocol (identical to TGN/TGAT/DyRep benchmark):
    - Temporal 70/15/15 split via split_dataset(method='temporal')
    - Train on COMPLIANT triples only (label=0)
    - Threshold tuned on validation set (same as neural models)
    - Metrics: AUC, AUPRC, F1 on test set

ComplEx (Trouillon et al. 2016):
    Static embedding — no timestamp. Score(h,r,t) = Re(<e_h, w_r, conj(e_t)>).
    Expected result: random (AUPRC ≈ prevalence) because violations are caused
    by temporal dynamics invisible to structure-only models.

TNTComplEx (Lacroix et al. NeurIPS 2020):
    Time-aware embedding. Score(h,r,t,τ) = Re(<e_h, w_r, conj(e_t), e_τ>).
    Adds a learned time embedding — can distinguish (worker, step, t=early)
    from (worker, step, t=after_rule_change). Expected to outperform ComplEx
    but fall short of TGN (no persistent memory, no message passing).

Usage:
    python experiments/UseCase4/run_static_baseline.py [--dataset single|multi|multi_varied|all]
    python experiments/UseCase4/run_static_baseline.py --model complex
    python experiments/UseCase4/run_static_baseline.py --model tntcomplex
    python experiments/UseCase4/run_static_baseline.py --model all

Output:
    experiments/UseCase4/results/static_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_single_project, load_multi_project, load_multi_varied
from eval_framework import split_dataset, find_best_threshold, compute_metrics

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED              = 42
EPOCHS            = 50
NEG_SAMPLES       = 5
BATCH_SIZE        = 1024
MAX_TRAIN_TRIPLES = 200_000   # cap for large datasets

# ComplEx hyperparams
DIM          = 64
LR_COMPLEX   = 0.1
WD_COMPLEX   = 1e-4

# TNTComplEx hyperparams
RANK_TNT     = 64             # same as ComplEx DIM for fair comparison
N_TIME_BINS  = 32             # quantile bins for tau discretisation
LR_TNT       = 5e-3           # from notebook 06
LAM_TNT      = 1e-3           # N3 regularisation weight


# ─── ComplEx ──────────────────────────────────────────────────────────────────

class ComplEx(nn.Module):
    """ComplEx (Trouillon et al. 2016). Score: Re(<e_h, w_r, conj(e_t)>)."""
    def __init__(self, n_entities: int, n_relations: int, dim: int = 64):
        super().__init__()
        self.e_re = nn.Embedding(n_entities, dim)
        self.e_im = nn.Embedding(n_entities, dim)
        self.r_re = nn.Embedding(n_relations, dim)
        self.r_im = nn.Embedding(n_relations, dim)
        for emb in [self.e_re, self.e_im, self.r_re, self.r_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h, r, t):
        return (
            self.e_re(h) * self.r_re(r) * self.e_re(t)
          + self.e_re(h) * self.r_im(r) * self.e_im(t)
          + self.e_im(h) * self.r_re(r) * self.e_im(t)
          - self.e_im(h) * self.r_im(r) * self.e_re(t)
        ).sum(-1)


# ─── TNTComplEx ───────────────────────────────────────────────────────────────

class TNTComplEx(nn.Module):
    """
    TNTComplEx (Lacroix et al. NeurIPS 2020).
    Score: Re(<e_h, w_r, conj(e_t), e_tau>)
    Time embedding e_tau modulates the score per timestamp bin.
    """
    def __init__(self, n_entities: int, n_relations: int, n_times: int, rank: int = 64):
        super().__init__()
        self.E_re  = nn.Embedding(n_entities,  rank)
        self.E_im  = nn.Embedding(n_entities,  rank)
        self.R_re  = nn.Embedding(n_relations, rank)
        self.R_im  = nn.Embedding(n_relations, rank)
        self.T_re  = nn.Embedding(n_times,     rank)
        self.T_im  = nn.Embedding(n_times,     rank)
        self.drop  = nn.Dropout(0.1)
        for emb in [self.E_re, self.E_im, self.R_re, self.R_im, self.T_re, self.T_im]:
            nn.init.normal_(emb.weight, std=0.1)

    def score(self, h, r, t, tau):
        """Score for a batch of (h, r, t, tau) index tensors. Returns (batch,)."""
        h_re, h_im = self.drop(self.E_re(h)), self.drop(self.E_im(h))
        r_re, r_im = self.R_re(r), self.R_im(r)
        t_re, t_im = self.drop(self.E_re(t)), self.drop(self.E_im(t))
        tau_re, tau_im = self.T_re(tau), self.T_im(tau)
        # h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # (h*r) * conj(t)  [conj flips im sign]
        hrt_re =  hr_re * t_re + hr_im * t_im
        hrt_im =  hr_im * t_re - hr_re * t_im
        # Re( hrt * tau )
        return (hrt_re * tau_re - hrt_im * tau_im).sum(-1)

    def regularisation(self, h, r, t, tau):
        """N3 regularisation (Lacroix et al. 2018)."""
        def cubic(x): return x.abs().pow(3).sum(-1).mean()
        return (cubic(self.E_re(h)) + cubic(self.E_im(h)) +
                cubic(self.R_re(r)) + cubic(self.R_im(r)) +
                cubic(self.E_re(t)) + cubic(self.E_im(t)) +
                cubic(self.T_re(tau)) + cubic(self.T_im(tau))) / 8


# ─── Shared data helpers ──────────────────────────────────────────────────────

def build_vocab(df):
    workers = sorted(df['worker_id'].unique())
    steps   = sorted(df['step_id'].unique())
    w2idx   = {w: i for i, w in enumerate(workers)}
    s2idx   = {s: i + len(workers) for i, s in enumerate(steps)}
    n_ent   = len(workers) + len(steps)
    n_rel   = int(df['permit_enc'].max()) + 1
    return w2idx, s2idx, n_ent, n_rel


def df_to_triples(df, w2idx, s2idx):
    h = np.array([w2idx.get(w, 0) for w in df['worker_id']], dtype=np.int64)
    r = df['permit_enc'].values.astype(np.int64)
    t = np.array([s2idx.get(s, 0) for s in df['step_id']],   dtype=np.int64)
    return h, r, t


def discretize_tau(tau_full: np.ndarray, n_bins: int = N_TIME_BINS) -> np.ndarray:
    """Map continuous tau values to integer bin indices [0, n_bins-1]."""
    edges = np.percentile(tau_full, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    bins  = np.digitize(tau_full, edges[:-1]) - 1
    return np.clip(bins, 0, len(edges) - 2).astype(np.int64)


# ─── ComplEx training & inference ─────────────────────────────────────────────

def train_complex(pos_h, pos_r, pos_t, n_entities, n_relations, device):
    model = ComplEx(n_entities, n_relations, DIM).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR_COMPLEX, weight_decay=WD_COMPLEX)
    rng   = np.random.RandomState(SEED)
    n_pos = len(pos_h)

    ph = torch.tensor(pos_h, dtype=torch.long, device=device)
    pr = torch.tensor(pos_r, dtype=torch.long, device=device)
    pt = torch.tensor(pos_t, dtype=torch.long, device=device)

    for epoch in range(EPOCHS):
        model.train()
        perm  = rng.permutation(n_pos)
        eloss = 0.0; nb = 0
        for start in range(0, min(n_pos, MAX_TRAIN_TRIPLES), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            bh, br, bt = ph[idx], pr[idx], pt[idx]
            bsz = len(idx)
            pos_s = model.score(bh, br, bt)
            neg_s = torch.stack([
                model.score(torch.randint(0, n_entities, (bsz,), device=device), br, bt)
                if rng.rand() < 0.5 else
                model.score(bh, br, torch.randint(0, n_entities, (bsz,), device=device))
                for _ in range(NEG_SAMPLES)
            ], dim=1)
            loss = (-torch.log(torch.sigmoid(pos_s) + 1e-8).mean()
                    -torch.log(1 - torch.sigmoid(neg_s) + 1e-8).mean())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1:3d}/{EPOCHS}  loss={eloss/nb:.4f}')
    return model


@torch.no_grad()
def score_triples_complex(model, h_arr, r_arr, t_arr, device, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(h_arr), bs):
        bh = torch.tensor(h_arr[s:s+bs], dtype=torch.long, device=device)
        br = torch.tensor(r_arr[s:s+bs], dtype=torch.long, device=device)
        bt = torch.tensor(t_arr[s:s+bs], dtype=torch.long, device=device)
        out.append(model.score(bh, br, bt).cpu().numpy())
    scores = np.concatenate(out)
    return torch.sigmoid(torch.tensor(-scores)).numpy()


# ─── TNTComplEx training & inference ──────────────────────────────────────────

def train_tntcomplex(pos_h, pos_r, pos_t, pos_tau, n_entities, n_relations, device):
    n_times = N_TIME_BINS
    model   = TNTComplEx(n_entities, n_relations, n_times, RANK_TNT).to(device)
    opt     = optim.Adam(model.parameters(), lr=LR_TNT)
    rng     = np.random.RandomState(SEED)
    n_pos   = len(pos_h)

    ph   = torch.tensor(pos_h,   dtype=torch.long, device=device)
    pr   = torch.tensor(pos_r,   dtype=torch.long, device=device)
    pt   = torch.tensor(pos_t,   dtype=torch.long, device=device)
    ptau = torch.tensor(pos_tau, dtype=torch.long, device=device)

    for epoch in range(EPOCHS):
        model.train()
        perm  = rng.permutation(n_pos)
        eloss = 0.0; nb = 0
        for start in range(0, min(n_pos, MAX_TRAIN_TRIPLES), BATCH_SIZE):
            idx  = perm[start:start + BATCH_SIZE]
            bh, br, bt, btau = ph[idx], pr[idx], pt[idx], ptau[idx]
            bsz  = len(idx)
            pos_s = model.score(bh, br, bt, btau)
            neg_s = torch.stack([
                model.score(torch.randint(0, n_entities, (bsz,), device=device), br, bt, btau)
                if rng.rand() < 0.5 else
                model.score(bh, br, torch.randint(0, n_entities, (bsz,), device=device), btau)
                for _ in range(NEG_SAMPLES)
            ], dim=1)
            loss = (-torch.log(torch.sigmoid(pos_s) + 1e-8).mean()
                    -torch.log(1 - torch.sigmoid(neg_s) + 1e-8).mean())
            reg  = model.regularisation(bh, br, bt, btau)
            loss = loss + LAM_TNT * reg
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1:3d}/{EPOCHS}  loss={eloss/nb:.4f}')
    return model


@torch.no_grad()
def score_triples_tnt(model, h_arr, r_arr, t_arr, tau_arr, device, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(h_arr), bs):
        bh   = torch.tensor(h_arr[s:s+bs],   dtype=torch.long, device=device)
        br   = torch.tensor(r_arr[s:s+bs],   dtype=torch.long, device=device)
        bt   = torch.tensor(t_arr[s:s+bs],   dtype=torch.long, device=device)
        btau = torch.tensor(tau_arr[s:s+bs], dtype=torch.long, device=device)
        out.append(model.score(bh, br, bt, btau).cpu().numpy())
    scores = np.concatenate(out)
    return torch.sigmoid(torch.tensor(-scores)).numpy()


# ─── Experiment runners ───────────────────────────────────────────────────────

def run_complex_baseline(df, dataset_name: str, device: torch.device) -> dict:
    print(f'\n  [ComplEx] [{dataset_name}] building vocab + splitting...')
    train_df, val_df, test_df = split_dataset(df, method='temporal', label_col='label', time_col='tau')
    w2idx, s2idx, n_ent, n_rel = build_vocab(df)
    train_c = train_df[train_df['label'] == 0]
    pos_h, pos_r, pos_t = df_to_triples(train_c, w2idx, s2idx)
    print(f'    n_entities={n_ent:,}  n_relations={n_rel}')
    print(f'    train_compliant={len(train_c):,}  training ComplEx (dim={DIM}, epochs={EPOCHS})...')
    t0 = time.time()
    model = train_complex(pos_h, pos_r, pos_t, n_ent, n_rel, device)
    sec = time.time() - t0
    val_h, val_r, val_t = df_to_triples(val_df, w2idx, s2idx)
    t_star = find_best_threshold(val_df['label'].values,
                                 score_triples_complex(model, val_h, val_r, val_t, device))
    test_h, test_r, test_t = df_to_triples(test_df, w2idx, s2idx)
    probs   = score_triples_complex(model, test_h, test_r, test_t, device)
    metrics = compute_metrics(test_df['label'].values, probs, threshold=t_star)
    prev    = float(test_df['label'].mean())
    lift    = metrics['auprc'] / prev if prev > 0 else float('nan')
    print(f'    AUC={metrics["auc"]:.3f}  AUPRC={metrics["auprc"]:.3f}  '
          f'lift=\xd7{lift:.1f}  F1={metrics["f1"]:.3f}  thr={t_star:.3f}  ({sec:.0f}s)')
    return {'model': 'ComplEx', 'split': 'temporal', 'dataset': dataset_name,
            'type': 'static_kg', 'seed': SEED, 'dim': DIM, 'epochs': EPOCHS,
            'n_entities': n_ent, 'n_relations': n_rel,
            'n_train_compliant': int(len(train_c)),
            'n_val': len(val_df), 'n_test': len(test_df),
            'n_pos_test': int(test_df['label'].sum()),
            'prevalence': round(prev, 4), 'threshold': round(float(t_star), 4),
            'train_sec': round(sec, 1), 'auprc_lift': round(float(lift), 2),
            'metrics': metrics}


def run_tntcomplex_baseline(df, dataset_name: str, device: torch.device) -> dict:
    print(f'\n  [TNTComplEx] [{dataset_name}] building vocab + splitting...')
    train_df, val_df, test_df = split_dataset(df, method='temporal', label_col='label', time_col='tau')
    w2idx, s2idx, n_ent, n_rel = build_vocab(df)

    # Discretise tau on full df, then index by position
    tau_full = df['tau'].values.astype(float)
    tau_bins = discretize_tau(tau_full, N_TIME_BINS)
    # Build position-to-bin mapping via dataframe reset index
    df_reset = df.reset_index(drop=True)
    tau_map   = dict(zip(df_reset.index, tau_bins))

    def get_tau(sub_df):
        sub_reset = sub_df.reset_index(drop=True)
        # find original positions in df
        return np.array([tau_bins[df_reset.index.get_loc(i) if i in df_reset.index else 0]
                         for i in sub_df.index], dtype=np.int64)

    # Simpler: reindex tau_bins by the integer positions in df
    df_pos = {orig_idx: pos for pos, orig_idx in enumerate(df.index)}

    def tau_for(sub_df):
        return np.array([tau_bins[df_pos.get(i, 0)] for i in sub_df.index], dtype=np.int64)

    train_c   = train_df[train_df['label'] == 0]
    pos_h, pos_r, pos_t = df_to_triples(train_c, w2idx, s2idx)
    pos_tau   = tau_for(train_c)

    print(f'    n_entities={n_ent:,}  n_relations={n_rel}  n_time_bins={N_TIME_BINS}')
    print(f'    train_compliant={len(train_c):,}  training TNTComplEx (rank={RANK_TNT}, epochs={EPOCHS})...')
    t0 = time.time()
    try:
        model = train_tntcomplex(pos_h, pos_r, pos_t, pos_tau, n_ent, n_rel, device)
    except RuntimeError as e:
        if 'memory' in str(e).lower() or 'alloc' in str(e).lower():
            print(f'    [OOM] n_entities={n_ent:,} too large for TNTComplEx on this device — skipping')
            return None
        raise
    sec = time.time() - t0

    val_h, val_r, val_t = df_to_triples(val_df, w2idx, s2idx)
    val_tau  = tau_for(val_df)
    t_star   = find_best_threshold(val_df['label'].values,
                                   score_triples_tnt(model, val_h, val_r, val_t, val_tau, device))

    test_h, test_r, test_t = df_to_triples(test_df, w2idx, s2idx)
    test_tau = tau_for(test_df)
    probs    = score_triples_tnt(model, test_h, test_r, test_t, test_tau, device)
    metrics  = compute_metrics(test_df['label'].values, probs, threshold=t_star)
    prev     = float(test_df['label'].mean())
    lift     = metrics['auprc'] / prev if prev > 0 else float('nan')
    print(f'    AUC={metrics["auc"]:.3f}  AUPRC={metrics["auprc"]:.3f}  '
          f'lift=\xd7{lift:.1f}  F1={metrics["f1"]:.3f}  thr={t_star:.3f}  ({sec:.0f}s)')
    return {'model': 'TNTComplEx', 'split': 'temporal', 'dataset': dataset_name,
            'type': 'temporal_kg', 'seed': SEED, 'rank': RANK_TNT,
            'n_time_bins': N_TIME_BINS, 'epochs': EPOCHS,
            'n_entities': n_ent, 'n_relations': n_rel,
            'n_train_compliant': int(len(train_c)),
            'n_val': len(val_df), 'n_test': len(test_df),
            'n_pos_test': int(test_df['label'].sum()),
            'prevalence': round(prev, 4), 'threshold': round(float(t_star), 4),
            'train_sec': round(sec, 1), 'auprc_lift': round(float(lift), 2),
            'metrics': metrics}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['single', 'multi', 'multi_varied', 'all'],
                        default='all')
    parser.add_argument('--model',   choices=['complex', 'tntcomplex', 'all'],
                        default='all')
    parser.add_argument('--data_dir', default='data/UseCase4')
    args = parser.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_all_ds  = args.dataset == 'all'
    run_complex = args.model in ('complex', 'all')
    run_tnt     = args.model in ('tntcomplex', 'all')

    print('Static KG Baselines (ComplEx + TNTComplEx)')
    print(f'  Device:  {device}')
    print(f'  Models:  {"ComplEx " if run_complex else ""}{"TNTComplEx" if run_tnt else ""}')
    print(f'  Dataset: {args.dataset}')

    loaders = []
    if args.dataset in ('single',) or run_all_ds:
        loaders.append(('single', lambda: load_single_project(args.data_dir)))
    if args.dataset in ('multi',) or run_all_ds:
        loaders.append(('multi', lambda: load_multi_project(args.data_dir)))
    if args.dataset in ('multi_varied',) or run_all_ds:
        loaders.append(('multi_varied', lambda: load_multi_varied(args.data_dir)))

    new_results = []
    for ds_name, loader in loaders:
        print(f'\nLoading {ds_name}...')
        df = loader()
        if run_complex:
            new_results.append(run_complex_baseline(df, ds_name, device))
        if run_tnt:
            r = run_tntcomplex_baseline(df, ds_name, device)
            if r is not None:
                new_results.append(r)

    # Merge with existing results
    out_path = RESULTS_DIR / 'static_baseline.json'
    if out_path.exists():
        existing = json.load(open(out_path, encoding='utf-8')).get('results', [])
    else:
        existing = []

    existing_keys = {(r['model'], r['dataset']) for r in existing}
    for r in new_results:
        key = (r['model'], r['dataset'])
        if key in existing_keys:
            existing = [r if (e['model'], e['dataset']) == key else e for e in existing]
        else:
            existing.append(r)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': existing}, f, indent=2)
    print(f'\nSaved -> {out_path}')

    # ── Summary table ──────────────────────────────────────────────────────────
    ref_rows = [
        ('Random',    'single',       0.500, 0.002,   1.0, 0.000),
        ('ComplEx',   'single',       0.440, 0.002,   1.0, 0.004),
        ('ComplEx',   'multi',        0.503, 0.002,   1.0, 0.005),
        ('ComplEx',   'multi_varied', 0.521, 0.002,   1.0, 0.005),
        ('LR',        'single',       0.738, 0.161,  87.8, 0.024),
        ('RF',        'single',       0.978, 0.160,  87.0, 0.071),
        ('TGN',       'single',       0.985, 0.178,  98.9, 0.084),
        ('TGAT',      'multi',        1.000, 0.955, 454.8, 0.905),
        ('TGAT',      'multi_varied', 0.992, 0.646, 309.0, 0.603),
    ]
    print('\n' + '=' * 85)
    print(f"  {'Model':<14} {'Dataset':<14} {'AUC':>6}  {'AUPRC':>6}  {'Lift':>7}  {'F1':>6}")
    print('  ' + '-' * 70)
    for model, ds, auc, auprc, lift, f1 in ref_rows:
        print(f"  {model:<14} {ds:<14} {auc:>6.3f}  {auprc:>6.3f}  \xd7{lift:>6.1f}  {f1:>6.3f}")
    print('  ' + '-' * 70)
    for r in [x for x in existing if x['model'] in ('TNTComplEx',)]:
        m = r['metrics']
        print(f"  {'TNTComplEx':<14} {r['dataset']:<14} "
              f"{m['auc']:>6.3f}  {m['auprc']:>6.3f}  \xd7{r['auprc_lift']:>6.1f}  {m['f1']:>6.3f}")
    print('=' * 85)


if __name__ == '__main__':
    main()
