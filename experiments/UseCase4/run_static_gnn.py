"""
Static GNN Baseline — GCN for EPC violation detection.

Purpose:
    Isolate the contribution of temporal information by placing a "structure-only"
    model between static KG embeddings and temporal GNNs:

        ComplEx   (no structure aggregation, no time)  AUPRC ≈ 0.002
        Static GNN (structure via message-passing, no time)  AUPRC = ?
        TGN        (structure + temporal memory)        AUPRC = 0.178

    If Static GNN ≈ RF (feature-driven, ~0.16): graph structure alone adds little.
    If Static GNN > RF: neighbourhood context helps beyond individual features.
    Either way, the gap to TGN quantifies the value of temporal modelling.

Architecture:
    1. Learnable node embeddings (hidden_dim=64) — workers and steps.
    2. k-layer GCN (Kipf & Welling 2017) over the STATIC training graph.
       Depth k ∈ {1, 2} selected on validation AUPRC.
    3. Edge classifier: MLP([src_emb ‖ dst_emb ‖ edge_feats]) → violation prob.

Key design choices:
    - Training graph built from TRAINING edges only (no future leakage).
    - All training edges used (compliant + violation), same as TGN.
    - Edge features = same FEAT_COLS as TGN (fair comparison).
    - Class imbalance: weighted BCE (pos_weight = n_neg / n_pos).
    - Threshold tuned on validation AUPRC (same as all other models).

Protocol (identical to TGN / TGAT / DyRep benchmark):
    Temporal 70/15/15 split → val-threshold → AUC / AUPRC / F1 on test.

Usage:
    python experiments/UseCase4/run_static_gnn.py [--dataset single|multi|multi_varied|all]

Output:
    experiments/UseCase4/results/static_gnn.json
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader    import load_single_project, load_multi_project, load_multi_varied, FEAT_COLS
from eval_framework import split_dataset, find_best_threshold, compute_metrics

RESULTS_DIR = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

SEED       = 42
HIDDEN     = 64
EPOCHS     = 50
LR         = 1e-3
BATCH_SIZE = 4096
DEPTHS     = [1, 2]      # evaluated on val; best depth chosen


# ─── GCN layer (Kipf & Welling 2017, D^{-1/2} A D^{-1/2} normalisation) ─────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim), adj_norm: sparse (N, N). Returns (N, out_dim)."""
        agg = torch.sparse.mm(adj_norm, x)
        return F.relu(self.norm(self.linear(agg)))


# ─── Static GNN model ─────────────────────────────────────────────────────────

class StaticGNN(nn.Module):
    """
    GCN encoder + MLP edge classifier.
    Node embeddings are updated via k rounds of message-passing on the
    static training graph (no timestamps). Edge classification uses
    [src_emb ‖ dst_emb ‖ edge_features].
    """
    def __init__(self, n_entities: int, n_edge_feats: int,
                 hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.node_emb = nn.Embedding(n_entities, hidden)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.convs = nn.ModuleList(
            [GCNLayer(hidden, hidden) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + n_edge_feats, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def encode(self, adj_norm: torch.Tensor) -> torch.Tensor:
        """Run GCN message-passing; returns node embeddings (N, hidden)."""
        x = self.node_emb.weight
        for conv in self.convs:
            x = conv(x, adj_norm)
        return x

    def forward(self, src: torch.Tensor, dst: torch.Tensor,
                edge_feat: torch.Tensor, node_embs: torch.Tensor) -> torch.Tensor:
        """Classify a batch of edges. Returns logits (batch,)."""
        h_src  = node_embs[src]
        h_dst  = node_embs[dst]
        edge_repr = torch.cat([h_src, h_dst, edge_feat], dim=-1)
        return self.classifier(edge_repr).squeeze(-1)


# ─── Graph construction ───────────────────────────────────────────────────────

def build_vocab(df):
    workers = sorted(df['worker_id'].unique())
    steps   = sorted(df['step_id'].unique())
    w2idx   = {w: i for i, w in enumerate(workers)}
    s2idx   = {s: i + len(workers) for i, s in enumerate(steps)}
    return w2idx, s2idx, len(workers) + len(steps)


def build_adj(src_ids: np.ndarray, dst_ids: np.ndarray,
              n: int, device: torch.device) -> torch.Tensor:
    """
    Symmetric normalised adjacency  D^{-1/2} (A + I) D^{-1/2}  as a sparse tensor.
    Includes self-loops and both edge directions (undirected graph).
    """
    # Undirected + self-loops
    rows = np.concatenate([src_ids, dst_ids, np.arange(n)])
    cols = np.concatenate([dst_ids, src_ids, np.arange(n)])

    # Degree for normalisation
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)

    vals = (deg_inv_sqrt[rows] * deg_inv_sqrt[cols]).astype(np.float32)

    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values  = torch.tensor(vals, dtype=torch.float32)
    adj     = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    return adj.to(device)


# ─── Training helpers ─────────────────────────────────────────────────────────

def make_edge_tensors(df, w2idx, s2idx, scaler, device):
    src  = torch.tensor([w2idx.get(w, 0) for w in df['worker_id']], dtype=torch.long,  device=device)
    dst  = torch.tensor([s2idx.get(s, 0) for s in df['step_id']],   dtype=torch.long,  device=device)
    feat = torch.tensor(
        scaler.transform(df[FEAT_COLS].fillna(0).values).astype(np.float32),
        dtype=torch.float32, device=device)
    lbl  = torch.tensor(df['label'].values, dtype=torch.float32, device=device)
    return src, dst, feat, lbl


def train_one_depth(n_entities, adj_norm, train_src, train_dst, train_feat, train_lbl,
                    val_src, val_dst, val_feat, val_lbl_np,
                    n_layers, device) -> tuple[StaticGNN, float]:
    """Train a StaticGNN with n_layers GCN layers; return (model, val_auprc)."""
    rng   = np.random.default_rng(SEED)
    model = StaticGNN(n_entities, len(FEAT_COLS), HIDDEN, n_layers).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    n_pos = int(train_lbl.sum().item())
    n_neg = len(train_lbl) - n_pos
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    n_train  = len(train_src)
    best_val = -1.0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=device)

        # Recompute node embeddings once per epoch (full-graph message passing)
        node_embs = model.encode(adj_norm)

        epoch_loss = 0.0; nb = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx  = perm[start:start + BATCH_SIZE]
            logits = model(train_src[idx], train_dst[idx], train_feat[idx], node_embs)
            loss   = F.binary_cross_entropy_with_logits(logits, train_lbl[idx], pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); nb += 1

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                node_embs_eval = model.encode(adj_norm)
                val_logits     = model(val_src, val_dst, val_feat, node_embs_eval)
                val_probs      = torch.sigmoid(val_logits).cpu().numpy()
            from sklearn.metrics import average_precision_score
            val_ap = average_precision_score(val_lbl_np, val_probs)
            print(f'    depth={n_layers} epoch {epoch+1:3d}/{EPOCHS}  '
                  f'loss={epoch_loss/nb:.4f}  val_AUPRC={val_ap:.4f}')
            if val_ap > best_val:
                best_val = val_ap

    return model, best_val


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_static_gnn(df, dataset_name: str, device: torch.device) -> dict:
    print(f'\n  [StaticGNN] [{dataset_name}] building graph + splitting...')

    train_df, val_df, test_df = split_dataset(
        df, method='temporal', label_col='label', time_col='tau')

    w2idx, s2idx, n_ent = build_vocab(df)
    print(f'    n_entities={n_ent:,}  train={len(train_df):,}  '
          f'val={len(val_df):,}  test={len(test_df):,}')

    # Fit scaler on training set only (same as RF/TGN)
    scaler = StandardScaler()
    scaler.fit(train_df[FEAT_COLS].fillna(0).values)

    # Build static adjacency from training edges (no future leakage)
    train_src_np = np.array([w2idx.get(w, 0) for w in train_df['worker_id']])
    train_dst_np = np.array([s2idx.get(s, 0) for s in train_df['step_id']])

    try:
        adj_norm = build_adj(train_src_np, train_dst_np, n_ent, device)
    except (RuntimeError, MemoryError) as e:
        print(f'    [OOM] adjacency too large for n_entities={n_ent:,} — skipping')
        return None

    train_src, train_dst, train_feat, train_lbl = make_edge_tensors(train_df, w2idx, s2idx, scaler, device)
    val_src,   val_dst,   val_feat,   val_lbl   = make_edge_tensors(val_df,   w2idx, s2idx, scaler, device)
    test_src,  test_dst,  test_feat,  test_lbl  = make_edge_tensors(test_df,  w2idx, s2idx, scaler, device)

    val_lbl_np  = val_df['label'].values
    test_lbl_np = test_df['label'].values

    # ── Select depth on validation AUPRC ──────────────────────────────────────
    print(f'    Training with depths={DEPTHS}, selecting best on val AUPRC...')
    t0 = time.time()

    best_depth, best_auprc, best_model = 1, -1.0, None
    for d in DEPTHS:
        model, val_ap = train_one_depth(
            n_ent, adj_norm,
            train_src, train_dst, train_feat, train_lbl,
            val_src,   val_dst,   val_feat,   val_lbl_np,
            n_layers=d, device=device)
        print(f'    depth={d}  val_AUPRC={val_ap:.4f}')
        if val_ap > best_auprc:
            best_auprc = val_ap
            best_depth = d
            best_model = model

    train_sec = time.time() - t0
    print(f'    Best depth: {best_depth} (val_AUPRC={best_auprc:.4f})')

    # ── Evaluate best model on test set ───────────────────────────────────────
    best_model.eval()
    with torch.no_grad():
        node_embs  = best_model.encode(adj_norm)
        val_probs  = torch.sigmoid(best_model(val_src, val_dst, val_feat, node_embs)).cpu().numpy()
        test_probs = torch.sigmoid(best_model(test_src, test_dst, test_feat, node_embs)).cpu().numpy()

    t_star  = find_best_threshold(val_lbl_np, val_probs)
    metrics = compute_metrics(test_lbl_np, test_probs, threshold=t_star)

    prev = float(test_df['label'].mean())
    lift = metrics['auprc'] / prev if prev > 0 else float('nan')

    print(f'    AUC={metrics["auc"]:.3f}  AUPRC={metrics["auprc"]:.3f}  '
          f'lift=\xd7{lift:.1f}  F1={metrics["f1"]:.3f}  '
          f'thr={t_star:.3f}  ({train_sec:.0f}s)')

    return {
        'model':       'StaticGNN',
        'split':       'temporal',
        'dataset':     dataset_name,
        'type':        'static_gnn',
        'seed':        SEED,
        'hidden':      HIDDEN,
        'best_depth':  best_depth,
        'depths_tried': DEPTHS,
        'epochs':      EPOCHS,
        'n_entities':  n_ent,
        'n_edge_feats': len(FEAT_COLS),
        'feat_cols':   FEAT_COLS,
        'n_train':     len(train_df),
        'n_val':       len(val_df),
        'n_test':      len(test_df),
        'n_pos_test':  int(test_df['label'].sum()),
        'prevalence':  round(prev, 4),
        'threshold':   round(float(t_star), 4),
        'train_sec':   round(train_sec, 1),
        'auprc_lift':  round(float(lift), 2),
        'metrics':     metrics,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Static GNN baseline for EPC violation detection')
    parser.add_argument('--dataset', choices=['single', 'multi', 'multi_varied', 'all'],
                        default='all')
    parser.add_argument('--data_dir', default='data/UseCase4')
    args = parser.parse_args()

    np.random.seed(SEED); torch.manual_seed(SEED)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_all = args.dataset == 'all'

    print('Static GNN Baseline')
    print(f'  Device:  {device}')
    print(f'  hidden={HIDDEN}, epochs={EPOCHS}/depth, depths={DEPTHS}')
    print(f'  Dataset: {args.dataset}')

    loaders = []
    if args.dataset == 'single'       or run_all: loaders.append(('single',       load_single_project))
    if args.dataset == 'multi'        or run_all: loaders.append(('multi',        load_multi_project))
    if args.dataset == 'multi_varied' or run_all: loaders.append(('multi_varied', load_multi_varied))

    results = []
    for ds_name, loader_fn in loaders:
        print(f'\nLoading {ds_name}...')
        df = loader_fn(args.data_dir)
        r  = run_static_gnn(df, ds_name, device)
        if r is not None:
            results.append(r)

    out_path = RESULTS_DIR / 'static_gnn.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, indent=2)
    print(f'\nSaved -> {out_path}')

    # ── Summary table ──────────────────────────────────────────────────────────
    ref = [
        ('Random',    'single',       0.500, 0.002,   1.0),
        ('ComplEx',   'single',       0.440, 0.002,   1.0),
        ('LR',        'single',       0.738, 0.161,  87.8),
        ('RF',        'single',       0.978, 0.160,  87.0),
        ('TGN',       'single',       0.985, 0.178,  98.9),
        ('TGAT',      'multi',        1.000, 0.955, 454.8),
        ('TGAT',      'multi_varied', 0.992, 0.646, 309.0),
    ]
    print('\n' + '=' * 75)
    print(f"  {'Model':<14} {'Dataset':<14} {'AUC':>6}  {'AUPRC':>6}  {'Lift':>7}")
    print('  ' + '-' * 60)
    for m, ds, auc, auprc, lift in ref:
        print(f"  {m:<14} {ds:<14} {auc:>6.3f}  {auprc:>6.3f}  \xd7{lift:>6.1f}")
    print('  ' + '-' * 60)
    for r in results:
        m = r['metrics']
        print(f"  {'StaticGNN(d='+str(r['best_depth'])+')':<14} {r['dataset']:<14} "
              f"{m['auc']:>6.3f}  {m['auprc']:>6.3f}  \xd7{r['auprc_lift']:>6.1f}")
    print('=' * 75)


if __name__ == '__main__':
    main()
