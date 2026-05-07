"""DyRep (Trivedi et al., NeurIPS 2018) — standalone module for benchmark runner."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


# ── Model (identical to nb06 cell 46) ─────────────────────────────────────────

class DyRepEPC(nn.Module):
    """
    DyRep for EPC violation prediction.

    k=0 (association): after_rc=1 -> rule-change events, slow psi_assoc
    k=1 (communication): after_rc=0 -> assignment events, fast psi_comm
    """

    def __init__(self, num_nodes, embed_dim=32, edge_dim=6, n_neighbors=10):
        super().__init__()
        self.embed_dim   = embed_dim
        self.n_neighbors = n_neighbors
        self.num_nodes   = num_nodes

        self.register_buffer('z', torch.zeros(num_nodes, embed_dim))
        self.log_psi_comm  = nn.Parameter(torch.tensor(0.0))
        self.log_psi_assoc = nn.Parameter(torch.tensor(-2.0))
        self.omega_comm    = nn.Linear(2 * embed_dim, 1, bias=False)
        self.omega_assoc   = nn.Linear(2 * embed_dim, 1, bias=False)
        self.W_struct = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_rec    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_t      = nn.Linear(1, embed_dim, bias=False)
        self.W_h      = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('s_vec', torch.zeros(num_nodes))
        self._nbrs = [[] for _ in range(num_nodes)]
        self.head  = nn.Sequential(
            nn.Linear(embed_dim + edge_dim, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1))

    @property
    def psi_comm(self):  return self.log_psi_comm.exp()
    @property
    def psi_assoc(self): return self.log_psi_assoc.exp()

    def reset(self):
        self.z.zero_()
        self.s_vec.zero_()
        self._nbrs = [[] for _ in range(self.num_nodes)]

    def intensity(self, z_u, z_v, k):
        psi   = self.psi_comm  if k == 1 else self.psi_assoc
        omega = self.omega_comm if k == 1 else self.omega_assoc
        g = omega(torch.cat([z_u, z_v], dim=-1))
        return psi * torch.log1p(torch.exp(g / (psi.abs() + 1e-6)))

    def _aggregate(self, node_ids_list):
        dev = self.z.device
        out = []
        for nid in node_ids_list:
            nbrs = self._nbrs[nid][-self.n_neighbors:]
            if not nbrs:
                out.append(torch.zeros(self.embed_dim, device=dev))
                continue
            nbr_t = torch.tensor(nbrs, dtype=torch.long, device=dev)
            h     = torch.tanh(self.W_h(self.z[nbr_t]))
            attn  = torch.softmax(self.s_vec[nbr_t], dim=0).unsqueeze(-1)
            out.append(torch.sigmoid(attn * h).max(dim=0)[0])
        return torch.stack(out)

    def _update_state(self, src_list, dst_list, dt):
        dev = self.z.device
        h_s = self._aggregate(dst_list)
        h_d = self._aggregate(src_list)
        st  = torch.tensor(src_list, dtype=torch.long, device=dev)
        dt_ = torch.tensor(dst_list, dtype=torch.long, device=dev)
        z_s, z_d = self.z[st], self.z[dt_]
        self.z[st]  = torch.sigmoid(self.W_struct(h_s) + self.W_rec(z_s) + self.W_t(dt)).detach()
        self.z[dt_] = torch.sigmoid(self.W_struct(h_d) + self.W_rec(z_d) + self.W_t(dt)).detach()
        alpha = 0.1
        for u, v in zip(src_list, dst_list):
            if not self._nbrs[u] or self._nbrs[u][-1] != v: self._nbrs[u].append(v)
            if not self._nbrs[v] or self._nbrs[v][-1] != u: self._nbrs[v].append(u)
            self.s_vec[u] = (1 - alpha) * self.s_vec[u] + alpha
            self.s_vec[v] = (1 - alpha) * self.s_vec[v] + alpha

    def forward(self, src, dst, ef, dt, update=True):
        k   = 0 if float(ef[:, 2].mean()) > 0.5 else 1
        lam = self.intensity(self.z[src], self.z[dst], k)
        if update:
            self._update_state(src.tolist(), dst.tolist(), dt)
        logit = self.head(torch.cat([self.z[dst], ef], dim=-1))
        return torch.sigmoid(logit).squeeze(-1), lam


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(num_nodes: int, edge_dim: int = 6, **hparams) -> DyRepEPC:
    return DyRepEPC(
        num_nodes   = num_nodes,
        embed_dim   = hparams.get('embed_dim',    32),
        edge_dim    = edge_dim,
        n_neighbors = hparams.get('n_neighbors',  10),
    )


# ── Data preparation ──────────────────────────────────────────────────────────

def _to_tensors(df, feat_cols, scaler, dt_max, device):
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
    model: DyRepEPC,
    train_df,
    val_df,
    feat_cols: list[str],
    scaler,
    device=None,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    intensity_reg: float = 0.01,
    pos_weight_factor: float = 1.0,
    **_,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ts_all = np.concatenate([train_df['tau'].values, val_df['tau'].values])
    dt_max = float(np.abs(np.diff(ts_all.astype(np.float64),
                                   prepend=ts_all[0])).max()) + 1e-8

    src_tr, dst_tr, ef_tr, dt_tr, y_tr = _to_tensors(train_df, feat_cols, scaler, dt_max, device)
    src_va, dst_va, ef_va, dt_va, y_va = _to_tensors(val_df,   feat_cols, scaler, dt_max, device)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pw    = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)],
                         dtype=torch.float32, device=device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    val_history = []
    for ep in range(1, n_epochs + 1):
        model.train()
        model.reset()
        for start in range(0, len(src_tr), batch_size):
            sl = slice(start, start + batch_size)
            opt.zero_grad()
            prob, lam = model(src_tr[sl], dst_tr[sl], ef_tr[sl], dt_tr[sl], update=True)
            bce  = nn.functional.binary_cross_entropy(
                prob.clamp(1e-6, 1 - 1e-6), y_tr[sl],
                weight=torch.where(y_tr[sl] == 1, pw.expand_as(y_tr[sl]),
                                   torch.ones_like(y_tr[sl])))
            loss = bce - intensity_reg * lam.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Val AUPRC
        model.eval()
        model.reset()
        with torch.no_grad():
            probs = []
            for start in range(0, len(src_va), 256):
                sl = slice(start, start + 256)
                p, _ = model(src_va[sl], dst_va[sl], ef_va[sl], dt_va[sl], update=False)
                probs.extend(p.cpu().numpy())
        y_np  = y_va.cpu().numpy()
        auprc = average_precision_score(y_np, probs) if y_np.sum() > 0 else float('nan')
        val_history.append(auprc)

        if ep % 5 == 0 or ep == 1:
            print(f'  [DyRep] Epoch {ep:3d}/{n_epochs}  val_AUPRC={auprc:.4f}'
                  f'  psi_comm={model.psi_comm.item():.3f}'
                  f'  psi_assoc={model.psi_assoc.item():.4f}')

    model._dt_max = dt_max
    return model, val_history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: DyRepEPC, df, feat_cols, scaler, device=None) -> dict:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dt_max = getattr(model, '_dt_max', 1.0)
    src, dst, ef, dt, y = _to_tensors(df, feat_cols, scaler, dt_max, device)

    model.eval()
    model.reset()
    probs = []
    with torch.no_grad():
        for start in range(0, len(src), 256):
            sl = slice(start, start + 256)
            p, _ = model(src[sl], dst[sl], ef[sl], dt[sl], update=False)
            probs.extend(p.cpu().numpy())

    import sys
    sys.path.insert(0, str(__file__).rsplit('models', 1)[0])
    from eval_framework import compute_metrics
    result = compute_metrics(y.cpu().numpy().astype(int), np.array(probs))
    result['psi_comm']  = float(model.psi_comm.item())
    result['psi_assoc'] = float(model.psi_assoc.item())
    return result
