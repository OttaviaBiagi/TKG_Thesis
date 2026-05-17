"""
Static KG Baseline — ComplEx embedding for EPC violation detection.

Why ComplEx:
    Handles both symmetric (worker-worker substitutability) and asymmetric
    (worker assigned-to step) relations. TransE would require strict translational
    structure (h+r≈t) which breaks for many-to-many EPC assignments.

Training protocol:
    Train on COMPLIANT training triples only (label=0). The model learns what
    triples are "normal". Violations should receive low plausibility scores.
    violation_prob = sigmoid(-score)  [high score = plausible = compliant]

Expected result:
    Poor AUPRC despite potentially high AUC, because violations are caused by
    temporal dynamics (certificate expiry, regulation changes after rule_change_date)
    that a static embedding cannot see — a worker compliant before the regulation
    change looks structurally identical to a non-compliant worker afterwards.
    This is the thesis argument: temporal graph context is necessary.

Usage (from project root):
    python experiments/UseCase4/run_static_baseline.py [--dataset single|multi|multi_varied|all]

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
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_single_project, load_multi_project, load_multi_varied
from eval_framework import split_dataset, find_best_threshold, compute_metrics

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED             = 42
DIM              = 64       # embedding dimension
EPOCHS           = 50       # training epochs
LR               = 0.1      # stable LR with gradient clipping
NEG_SAMPLES      = 5        # negative triples per positive
BATCH_SIZE       = 1024
WEIGHT_DECAY     = 1e-4     # light L2 regularization
MAX_TRAIN_TRIPLES = 200_000  # cap for multi/multi_varied to keep runtime ~10 min


# ─── Model ────────────────────────────────────────────────────────────────────

class ComplEx(nn.Module):
    """
    ComplEx (Trouillon et al., 2016) — complex-valued bilinear scoring.

    Score: Re(<h, r, conj(t)>) = sum of four element-wise products.
    High score = plausible (compliant) triple.
    Low score  = implausible (anomalous) triple → potential violation.
    """
    def __init__(self, n_entities: int, n_relations: int, dim: int = 64):
        super().__init__()
        self.e_re = nn.Embedding(n_entities, dim)
        self.e_im = nn.Embedding(n_entities, dim)
        self.r_re = nn.Embedding(n_relations, dim)
        self.r_im = nn.Embedding(n_relations, dim)
        for emb in [self.e_re, self.e_im, self.r_re, self.r_im]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute Re(<e_h, w_r, conj(e_t)>) per triple. Returns shape (batch,)."""
        return (
            self.e_re(h) * self.r_re(r) * self.e_re(t)
          + self.e_re(h) * self.r_im(r) * self.e_im(t)
          + self.e_im(h) * self.r_re(r) * self.e_im(t)
          - self.e_im(h) * self.r_im(r) * self.e_re(t)
        ).sum(-1)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def build_vocab(df):
    """
    Build entity and relation vocabularies from the full DataFrame.

    Transductive setting: all entities known at train time (standard for KG).
    Relations = permit types (7 permit categories in CERT_REQS).
    """
    workers = sorted(df['worker_id'].unique())
    steps   = sorted(df['step_id'].unique())
    w2idx   = {w: i for i, w in enumerate(workers)}
    s2idx   = {s: i + len(workers) for i, s in enumerate(steps)}
    n_ent   = len(workers) + len(steps)
    n_rel   = int(df['permit_enc'].max()) + 1
    return w2idx, s2idx, n_ent, n_rel


def df_to_triples(df, w2idx: dict, s2idx: dict):
    """Convert DataFrame rows to (h_idx, r_idx, t_idx) integer arrays."""
    h = np.array([w2idx.get(w, 0) for w in df['worker_id']], dtype=np.int64)
    r = df['permit_enc'].values.astype(np.int64)
    t = np.array([s2idx.get(s, 0) for s in df['step_id']],   dtype=np.int64)
    return h, r, t


# ─── Training ─────────────────────────────────────────────────────────────────

def train_complex(
    pos_h: np.ndarray,
    pos_r: np.ndarray,
    pos_t: np.ndarray,
    n_entities: int,
    n_relations: int,
    device: torch.device,
) -> ComplEx:
    """Train ComplEx on positive (compliant) triples with random negative sampling."""
    model = ComplEx(n_entities, n_relations, DIM).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng   = np.random.RandomState(SEED)
    n_pos = len(pos_h)

    ph = torch.tensor(pos_h, dtype=torch.long, device=device)
    pr = torch.tensor(pos_r, dtype=torch.long, device=device)
    pt = torch.tensor(pos_t, dtype=torch.long, device=device)

    for epoch in range(EPOCHS):
        model.train()
        perm       = rng.permutation(n_pos)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, min(n_pos, MAX_TRAIN_TRIPLES), BATCH_SIZE):
            idx   = perm[start:start + BATCH_SIZE]
            bh, br, bt = ph[idx], pr[idx], pt[idx]
            bsz   = len(idx)

            # Positive scores
            pos_scores = model.score(bh, br, bt)

            # Negative scores — corrupt head or tail uniformly at random
            neg_scores_list = []
            for _ in range(NEG_SAMPLES):
                if rng.rand() < 0.5:
                    nh = torch.randint(0, n_entities, (bsz,), device=device)
                    neg_scores_list.append(model.score(nh, br, bt))
                else:
                    nt = torch.randint(0, n_entities, (bsz,), device=device)
                    neg_scores_list.append(model.score(bh, br, nt))

            neg_scores = torch.stack(neg_scores_list, dim=1)  # (bsz, NEG_SAMPLES)

            # Binary cross-entropy: positives=1, negatives=0
            pos_loss = -torch.log(torch.sigmoid(pos_scores)      + 1e-8).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores)  + 1e-8).mean()
            loss     = pos_loss + neg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}")

    return model


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_triples(
    model: ComplEx,
    h_arr: np.ndarray,
    r_arr: np.ndarray,
    t_arr: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Compute violation probabilities for all triples.

    violation_prob = sigmoid(-score):
        high ComplEx score → plausible (compliant) → low violation prob
        low  ComplEx score → implausible (anomalous) → high violation prob
    """
    model.eval()
    scores = []
    for start in range(0, len(h_arr), batch_size):
        bh = torch.tensor(h_arr[start:start + batch_size], dtype=torch.long, device=device)
        br = torch.tensor(r_arr[start:start + batch_size], dtype=torch.long, device=device)
        bt = torch.tensor(t_arr[start:start + batch_size], dtype=torch.long, device=device)
        scores.append(model.score(bh, br, bt).cpu().numpy())
    scores = np.concatenate(scores)
    return torch.sigmoid(torch.tensor(-scores)).numpy()


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_complex_baseline(df, dataset_name: str, device: torch.device) -> dict:
    print(f"\n  [ComplEx] [{dataset_name}] building vocab + splitting...")

    train_df, val_df, test_df = split_dataset(
        df, method='temporal', label_col='label', time_col='tau')

    w2idx, s2idx, n_ent, n_rel = build_vocab(df)

    train_compliant = train_df[train_df['label'] == 0]
    pos_h, pos_r, pos_t = df_to_triples(train_compliant, w2idx, s2idx)

    print(f"    n_entities={n_ent:,}  n_relations={n_rel}")
    print(f"    train_compliant={len(train_compliant):,}  violations excluded from KG")
    print(f"    training ComplEx (dim={DIM}, epochs={EPOCHS}, neg={NEG_SAMPLES})...")

    t0 = time.time()
    model = train_complex(pos_h, pos_r, pos_t, n_ent, n_rel, device)
    train_sec = time.time() - t0

    val_h, val_r, val_t = df_to_triples(val_df, w2idx, s2idx)
    val_probs = score_triples(model, val_h, val_r, val_t, device)
    t_star = find_best_threshold(val_df['label'].values, val_probs)

    test_h, test_r, test_t = df_to_triples(test_df, w2idx, s2idx)
    test_probs = score_triples(model, test_h, test_r, test_t, device)
    metrics = compute_metrics(test_df['label'].values, test_probs, threshold=t_star)

    prev = float(test_df['label'].mean())
    lift = metrics['auprc'] / prev if prev > 0 else float('nan')

    print(f"    AUC={metrics['auc']:.3f}  AUPRC={metrics['auprc']:.3f}  "
          f"lift=×{lift:.1f}  F1={metrics['f1']:.3f}  "
          f"thr={t_star:.3f}  ({train_sec:.0f}s)")

    return {
        'model':           'ComplEx',
        'split':           'temporal',
        'dataset':         dataset_name,
        'type':            'static_kg_baseline',
        'seed':            SEED,
        'dim':             DIM,
        'epochs':          EPOCHS,
        'neg_samples':     NEG_SAMPLES,
        'train_sec':       round(train_sec, 2),
        'n_entities':      n_ent,
        'n_relations':     n_rel,
        'n_train_compliant': int(len(train_compliant)),
        'n_val':           len(val_df),
        'n_test':          len(test_df),
        'n_pos_test':      int(test_df['label'].sum()),
        'prevalence':      round(prev, 4),
        'threshold':       round(float(t_star), 4),
        'auprc_lift':      round(float(lift), 2),
        'metrics':         metrics,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ComplEx static KG baseline for EPC violation detection')
    parser.add_argument('--dataset',  choices=['single', 'multi', 'multi_varied', 'all'],
                        default='all', help='Which dataset(s) to run')
    parser.add_argument('--data_dir', default='data/UseCase4')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Static KG Baseline (ComplEx)')
    print(f'  Device:  {device}')
    print(f'  dim={DIM}, epochs={EPOCHS}, neg_samples={NEG_SAMPLES}, lr={LR}')
    print(f'  Max train triples: {MAX_TRAIN_TRIPLES:,}')
    print(f'  Dataset: {args.dataset}')

    run_all = args.dataset == 'all'
    results = []

    if args.dataset in ('single',) or run_all:
        df = load_single_project(args.data_dir)
        results.append(run_complex_baseline(df, 'single', device))

    if args.dataset in ('multi',) or run_all:
        df = load_multi_project(args.data_dir)
        results.append(run_complex_baseline(df, 'multi', device))

    if args.dataset in ('multi_varied',) or run_all:
        df = load_multi_varied(args.data_dir)
        results.append(run_complex_baseline(df, 'multi_varied', device))

    out_path = RESULTS_DIR / 'static_baseline.json'
    # merge with any existing results for other datasets
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            existing = json.load(f).get('results', [])
        existing_keys = {(r['dataset'],) for r in existing}
        for r in results:
            if (r['dataset'],) not in existing_keys:
                existing.append(r)
            else:
                existing = [r if e['dataset'] == r['dataset'] else e for e in existing]
        results_to_save = existing
    else:
        results_to_save = results

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results_to_save}, f, indent=2)
    print(f'\nSaved -> {out_path}')

    # ── Summary table ──────────────────────────────────────────────────────────
    ref_rows = [
        ('Random',       'single',       0.500, 0.002, 1.0,   0.000),
        ('LR',           'single',       0.924, 0.059, 32.8,  0.000),
        ('RF',           'single',       0.968, 0.161, 89.4,  0.163),
        ('TGN',          'single',       0.985, 0.178, 98.9,  0.084),
        ('TGAT',         'multi',        1.000, 0.955, 454.8, 0.922),
        ('TGAT',         'multi_varied', 0.992, 0.646, 309.0, 0.603),
    ]

    print('\n' + '=' * 80)
    print(f"  {'Model':<22} {'Dataset':<14} {'AUC':>6}  {'AUPRC':>6}  "
          f"{'Lift':>7}  {'F1':>6}")
    print('  ' + '-' * 70)
    for model, ds, auc, auprc, lift, f1 in ref_rows:
        print(f"  {model:<22} {ds:<14} {auc:>6.3f}  {auprc:>6.3f}  "
              f"×{lift:>6.1f}  {f1:>6.3f}")
    print('  ' + '-' * 70)
    for r in results_to_save:
        m = r['metrics']
        print(f"  {'ComplEx (static KG)':<22} {r['dataset']:<14} "
              f"{m['auc']:>6.3f}  {m['auprc']:>6.3f}  "
              f"×{r['auprc_lift']:>6.1f}  {m['f1']:>6.3f}")
    print('=' * 80)


if __name__ == '__main__':
    main()
