"""TGAT (Xu et al., ICLR 2020) — standalone module for benchmark runner."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


# ── Time encoding ─────────────────────────────────────────────────────────────

class TimeEncode(nn.Module):
    """Bochner Fourier time encoding: phi(dt) = cos(w * dt + b)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


# ── TGAT model ────────────────────────────────────────────────────────────────

class TGAT_EPC(nn.Module):
    """
    Temporal Graph Attention for EPC violation detection.

    No persistent memory — embeddings computed on-the-fly from local
    interaction history via multi-head attention. Inductive by design.
    """

    def __init__(self, num_nodes, embed_dim=64, edge_dim=6,
                 n_heads=4, n_neighbors=20):
        super().__init__()
        self.embed_dim   = embed_dim
        self.n_neighbors = n_neighbors

        self.node_emb  = nn.Embedding(num_nodes, embed_dim)
        self.time_enc  = TimeEncode(embed_dim)
        self.attn      = nn.MultiheadAttention(embed_dim, n_heads,
                                               dropout=0.1, batch_first=True)
        self.edge_proj = nn.Linear(edge_dim, embed_dim)
        self.merge     = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU())
        self.head      = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1))

        self._history: list = [[] for _ in range(num_nodes)]

    def _embed(self, node_ids: torch.Tensor,
               t_now: torch.Tensor) -> torch.Tensor:
        B   = node_ids.size(0)
        K   = self.n_neighbors
        dev = node_ids.device

        z_static = self.node_emb(node_ids)                              # [B, D]

        # Pre-allocate padded history tensors (padding mask: True = ignore)
        past_n   = torch.zeros(B, K, dtype=torch.long,    device=dev)
        past_t   = torch.zeros(B, K, dtype=torch.float32, device=dev)
        pad_mask = torch.ones(B,  K, dtype=torch.bool,    device=dev)

        for i, (nid, t) in enumerate(zip(node_ids.tolist(), t_now.tolist())):
            hist = self._history[nid]
            if not hist:
                continue
            h = hist[-K:]
            L = len(h)
            past_t[i, :L] = torch.tensor([x[0] for x in h], dtype=torch.float32)
            past_n[i, :L] = torch.tensor([x[1] for x in h], dtype=torch.long)
            pad_mask[i, :L] = False

        has_hist = ~pad_mask.all(dim=1)                                 # [B]
        if not has_hist.any():
            return z_static

        # Only attend over nodes that actually have history — avoids all-masked
        # rows feeding into softmax(-inf,...,-inf) = NaN on CUDA.
        idx = has_hist.nonzero(as_tuple=True)[0]                        # [H]

        dt = (t_now[idx].unsqueeze(1) - past_t[idx]).clamp(min=0.0)    # [H, K]
        kv = self.node_emb(past_n[idx]) + self.time_enc(dt)             # [H, K, D]

        q_t = self.time_enc(torch.zeros(len(idx), dtype=torch.float32, device=dev))  # [H, D]
        q   = (z_static[idx] + q_t).unsqueeze(1)                        # [H, 1, D]

        attn_out, _ = self.attn(q, kv, kv,
                                key_padding_mask=pad_mask[idx])          # [H, 1, D]
        out = z_static.clone()
        out[idx] = attn_out.squeeze(1)                                   # [H, D]
        return out

    def _update_history(self, src_list, dst_list, t_list):
        cap = self.n_neighbors * 2
        for s, d, t in zip(src_list, dst_list, t_list):
            self._history[s].append((t, d))
            self._history[d].append((t, s))
            if len(self._history[s]) > cap:
                self._history[s] = self._history[s][-cap:]
            if len(self._history[d]) > cap:
                self._history[d] = self._history[d][-cap:]

    def reset_history(self):
        self._history = [[] for _ in range(self.node_emb.num_embeddings)]

    def forward(self, src, dst, ef, t, update=True):
        z_src  = self._embed(src, t)
        z_dst  = self._embed(dst, t)
        ef_d   = self.edge_proj(ef)
        h      = self.merge(torch.cat([z_dst + z_src, ef_d], dim=-1))
        if update:
            self._update_history(src.tolist(), dst.tolist(), t.tolist())
        return torch.sigmoid(self.head(h)).squeeze(-1)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(num_nodes: int, edge_dim: int = 6, **hparams) -> TGAT_EPC:
    return TGAT_EPC(
        num_nodes   = num_nodes,
        embed_dim   = hparams.get('embed_dim',    64),
        edge_dim    = edge_dim,
        n_heads     = hparams.get('n_heads',       4),
        n_neighbors = hparams.get('n_neighbors',  20),
    )


# ── Data preparation ──────────────────────────────────────────────────────────

def _to_tensors(df, feat_cols, scaler, tau_min, tau_range, device):
    src  = torch.tensor(df['src'].values,   dtype=torch.long,    device=device)
    dst  = torch.tensor(df['dst'].values,   dtype=torch.long,    device=device)
    ef   = torch.tensor(
        np.nan_to_num(scaler.transform(df[feat_cols].fillna(0).values)).astype(np.float32),
        dtype=torch.float32, device=device)
    t    = torch.tensor(
        ((df['tau'].values.astype(np.float64) - tau_min) / tau_range).astype(np.float32),
        dtype=torch.float32, device=device)
    y    = torch.tensor(df['label'].values, dtype=torch.float32, device=device)
    return src, dst, ef, t, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model: TGAT_EPC,
    train_df,
    val_df,
    feat_cols: list[str],
    scaler,
    device=None,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
    pos_weight_factor: float = 15.0,
    **_,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tau_min   = float(train_df['tau'].min())
    tau_range = float(train_df['tau'].max() - tau_min) + 1e-9
    model._tau_min   = tau_min
    model._tau_range = tau_range

    src_tr, dst_tr, ef_tr, t_tr, y_tr = _to_tensors(
        train_df, feat_cols, scaler, tau_min, tau_range, device)
    src_va, dst_va, ef_va, t_va, y_va = _to_tensors(
        val_df,   feat_cols, scaler, tau_min, tau_range, device)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pw    = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)],
                         dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    val_history = []

    for ep in range(1, n_epochs + 1):
        model.train()
        model.reset_history()
        idx = np.random.permutation(len(src_tr))

        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            s, d = src_tr[b], dst_tr[b]
            e, t, yb = ef_tr[b], t_tr[b], y_tr[b]
            opt.zero_grad()
            prob   = model(s, d, e, t, update=True)
            weight = torch.where(yb == 1, pw.expand_as(yb), torch.ones_like(yb))
            loss   = (nn.functional.binary_cross_entropy(prob, yb) * weight).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = []
            for start in range(0, len(src_va), 256):
                sl = slice(start, start + 256)
                p  = model(src_va[sl], dst_va[sl], ef_va[sl], t_va[sl], update=False)
                probs.extend(p.cpu().numpy())
        y_np  = y_va.cpu().numpy()
        auprc = average_precision_score(y_np, probs) if y_np.sum() > 0 else float('nan')
        val_history.append(auprc)

        if ep % 5 == 0 or ep == 1:
            print(f'  [TGAT] Epoch {ep:3d}/{n_epochs}  val_AUPRC={auprc:.4f}')

    return model, val_history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: TGAT_EPC, df, feat_cols, scaler, device=None) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_min   = getattr(model, '_tau_min',   0.0)
    tau_range = getattr(model, '_tau_range', 1.0)
    src, dst, ef, t, y = _to_tensors(df, feat_cols, scaler, tau_min, tau_range, device)

    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(src), 256):
            sl = slice(start, start + 256)
            p  = model(src[sl], dst[sl], ef[sl], t[sl], update=False)
            probs.extend(p.cpu().numpy())

    import sys
    sys.path.insert(0, str(__file__).rsplit('models', 1)[0])
    from eval_framework import compute_metrics
    return compute_metrics(y.cpu().numpy().astype(int), np.array(probs))
