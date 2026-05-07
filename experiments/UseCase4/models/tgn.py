"""TGN (Temporal Graph Network) — standalone module for benchmark runner."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

# ── Model classes (identical to nb06 cell 17) ─────────────────────────────────

class MemoryModule(nn.Module):
    def __init__(self, num_nodes, memory_dim, message_dim=None):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.gru    = nn.GRUCell(message_dim or memory_dim, memory_dim)

    def get(self, ids):  return self.memory[ids]
    def reset(self):     nn.init.zeros_(self.memory)

    def update(self, ids, msgs):
        self.memory[ids] = self.gru(msgs, self.memory[ids]).detach()


class MessageFunction(nn.Module):
    def __init__(self, memory_dim, edge_dim, message_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim + 1, message_dim), nn.ReLU(),
            nn.Linear(message_dim, message_dim))

    def forward(self, src_mem, dst_mem, ef, dt):
        return self.mlp(torch.cat([src_mem, dst_mem, ef, dt], dim=-1))


class TGN(nn.Module):
    def __init__(self, num_nodes, memory_dim=32, message_dim=32, embed_dim=32, edge_dim=6):
        super().__init__()
        self.memory  = MemoryModule(num_nodes, memory_dim, message_dim)
        self.message = MessageFunction(memory_dim, edge_dim, message_dim)
        self.mlp     = nn.Sequential(
            nn.Linear(memory_dim + edge_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim))

    def encode(self, src, dst, ef, dt, update=True):
        src_m = self.memory.get(src)
        dst_m = self.memory.get(dst)
        msg   = self.message(src_m, dst_m, ef, dt)
        if update:
            self.memory.update(src, msg)
            self.memory.update(dst, msg)
        return self.mlp(torch.cat([dst_m, ef], dim=-1))


class TGNClassifier(nn.Module):
    def __init__(self, num_nodes, memory_dim=32, message_dim=32,
                 embed_dim=32, edge_dim=6):
        super().__init__()
        self.tgn  = TGN(num_nodes, memory_dim, message_dim, embed_dim, edge_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(16, 1))

    def forward(self, src, dst, ef, dt, update=True):
        emb = self.tgn.encode(src, dst, ef, dt, update)
        return torch.sigmoid(self.head(emb)).squeeze(-1)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(num_nodes: int, edge_dim: int = 6, **hparams) -> TGNClassifier:
    return TGNClassifier(
        num_nodes    = num_nodes,
        memory_dim   = hparams.get('memory_dim',   32),
        message_dim  = hparams.get('message_dim',  32),
        embed_dim    = hparams.get('embed_dim',    32),
        edge_dim     = edge_dim,
    )


# ── Data preparation ──────────────────────────────────────────────────────────

def _to_tensors(df, feat_cols, scaler, dt_max, device):
    """Convert DataFrame partition to TGN input tensors."""
    src = torch.tensor(df['src'].values,   dtype=torch.long,    device=device)
    dst = torch.tensor(df['dst'].values,   dtype=torch.long,    device=device)
    ef  = torch.tensor(
        np.nan_to_num(scaler.transform(df[feat_cols].fillna(0).values)).astype(np.float32),
        dtype=torch.float32, device=device)
    ts  = df['tau'].values.astype(np.float64)
    dt  = (np.diff(ts, prepend=ts[0]) / dt_max).astype(np.float32)
    dt_t = torch.tensor(dt, dtype=torch.float32, device=device).unsqueeze(1)
    y   = torch.tensor(df['label'].values, dtype=torch.float32, device=device)
    return src, dst, ef, dt_t, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: TGNClassifier,
    train_df,
    val_df,
    feat_cols: list[str],
    scaler,
    device=None,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    pos_weight_factor: float = 1.0,
    **_,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ts_all = np.concatenate([train_df['tau'].values, val_df['tau'].values])
    dt_all = np.diff(ts_all.astype(np.float64), prepend=ts_all[0])
    dt_max = float(np.abs(dt_all).max()) + 1e-8

    src_tr, dst_tr, ef_tr, dt_tr, y_tr = _to_tensors(train_df, feat_cols, scaler, dt_max, device)
    src_va, dst_va, ef_va, dt_va, y_va = _to_tensors(val_df,   feat_cols, scaler, dt_max, device)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pw    = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)],
                         dtype=torch.float32, device=device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    val_history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        model.tgn.memory.reset()
        idx = np.random.permutation(len(src_tr))

        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            s, d = src_tr[b], dst_tr[b]
            e, t, yb = ef_tr[b], dt_tr[b], y_tr[b]
            opt.zero_grad()
            emb    = model.tgn.encode(s, d, e, t, update=True)
            logits = model.head(emb).squeeze(-1)
            crit(logits, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val AUPRC
        model.eval()
        model.tgn.memory.reset()
        with torch.no_grad():
            probs = []
            for start in range(0, len(src_va), batch_size):
                p = model(src_va[start:start+batch_size],
                          dst_va[start:start+batch_size],
                          ef_va[start:start+batch_size],
                          dt_va[start:start+batch_size], update=False)
                probs.extend(p.cpu().numpy())
        y_np = y_va.cpu().numpy()
        auprc = average_precision_score(y_np, probs) if y_np.sum() > 0 else float('nan')
        val_history.append(auprc)

        if ep % 5 == 0 or ep == 1:
            print(f'  [TGN] Epoch {ep:3d}/{n_epochs}  val_AUPRC={auprc:.4f}')

    model._dt_max = dt_max
    return model, val_history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: TGNClassifier, df, feat_cols, scaler, device=None) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt_max = getattr(model, '_dt_max', 1.0)
    src, dst, ef, dt, y = _to_tensors(df, feat_cols, scaler, dt_max, device)

    model.eval()
    model.tgn.memory.reset()
    probs = []
    with torch.no_grad():
        for start in range(0, len(src), 256):
            p = model(src[start:start+256], dst[start:start+256],
                      ef[start:start+256],  dt[start:start+256], update=False)
            probs.extend(p.cpu().numpy())

    import sys
    sys.path.insert(0, str(__file__.split('models')[0]))
    from eval_framework import compute_metrics
    return compute_metrics(y.cpu().numpy().astype(int), np.array(probs))
