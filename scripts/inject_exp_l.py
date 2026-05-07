"""Inject Experiment L (TGAT) into nb06 as cells 47 and 48."""
import json
from pathlib import Path

NB = Path('notebooks/UseCase4/06_tkg_models.ipynb')
nb = json.load(NB.open(encoding='utf-8'))

MD_L = """## Experiment L -- TGAT: Temporal Graph Attention Network

**TGAT** (Xu et al., ICLR 2020) computes node embeddings dynamically via
multi-head attention over a node's K most recent past interactions -- no
persistent memory module.

| Property | TGN | DyRep | TGAT |
|----------|-----|-------|------|
| Memory module | GRU per node | Intensity function | None (attention only) |
| Inductive (new nodes) | No | No | **Yes** |
| Temporal encoding | raw dt | implicit via psi | Bochner Fourier phi(dt) |
| Training signal | BCE | BCE + intensity reg | BCE |

**Bochner time encoding** (key innovation):
phi(dt) = cos(w * dt + b) where w, b are learned -- generalises RBF kernels to
encode relative time differences as a continuous, learnable feature vector.

**Inductive protocol** (from TGAT paper): 10 % of worker nodes are withheld
from training; at test time, their embeddings are computed on-the-fly from the
attention over whatever interactions they did have.  This tests generalisation
to workers who join the project mid-stream.

**Experiment structure:**
- **L-single**: TGAT on single real project, all 4 split methods via eval_framework
- **L-inductive**: dedicated inductive evaluation with new-node breakdown
"""

CODE_L = r"""import sys, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path('experiments/UseCase4').resolve()))
from eval_framework import split_dataset, compute_metrics, split_info, save_results

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'TGAT running on: {DEVICE}')

# =============================================================================
# TimeEncode -- Bochner Fourier time encoding (Xu et al. ICLR 2020)
# =============================================================================
class TimeEncode(nn.Module):
    '''phi(dt) = cos(w * dt + b), learnable w and b.'''
    def __init__(self, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_model) * 0.1)
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: (B,) or (B, K) -> same shape + d_model
        return torch.cos(dt.unsqueeze(-1) * self.w + self.b)


# =============================================================================
# TGAT_EPC
# =============================================================================
class TGAT_EPC(nn.Module):
    '''
    Temporal Graph Attention for EPC violation detection.

    No persistent memory: embeddings are computed on-the-fly from
    local interaction history -> inductive by design.
    '''

    def __init__(self, num_nodes, embed_dim=64, edge_dim=6,
                 n_heads=4, n_neighbors=20):
        super().__init__()
        self.embed_dim   = embed_dim
        self.n_neighbors = n_neighbors

        self.node_emb  = nn.Embedding(num_nodes, embed_dim)
        self.time_enc  = TimeEncode(embed_dim)

        # Multi-head attention over temporal neighbourhood
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                           dropout=0.1, batch_first=True)

        # Edge feature projection: edge_dim -> embed_dim
        self.edge_proj = nn.Linear(edge_dim, embed_dim)

        # Merge layer: fuse attended embedding + edge features
        self.merge = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        # History: node_id -> list of (t: float, neighbour_id: int)
        self._history: list = [[] for _ in range(num_nodes)]

    # -------------------------------------------------------------------------
    def _embed(self, node_ids: torch.Tensor,
               t_now: torch.Tensor) -> torch.Tensor:
        '''
        Compute embeddings via temporal attention over past interactions.
        New nodes (empty history) fall back to static table embedding.
        '''
        z_static = self.node_emb(node_ids)          # (B, D)
        out      = z_static.clone()

        for i, (nid, t) in enumerate(zip(node_ids.tolist(), t_now.tolist())):
            hist = self._history[nid]
            if not hist:
                continue                             # new node -> static fallback

            hist_k  = hist[-self.n_neighbors:]
            past_t  = torch.tensor([h[0] for h in hist_k],
                                   dtype=torch.float32, device=node_ids.device)
            past_n  = torch.tensor([h[1] for h in hist_k],
                                   dtype=torch.long,    device=node_ids.device)

            dt      = (t - past_t).clamp(min=0.0)   # (K,)
            t_enc   = self.time_enc(dt)              # (K, D)
            kv      = self.node_emb(past_n) + t_enc  # (K, D)

            # Query: static embedding + time encoding at dt=0
            q = (z_static[i] + self.time_enc(
                    torch.zeros(1, device=node_ids.device)).squeeze(0)
                 ).unsqueeze(0).unsqueeze(0)          # (1,1,D)

            attn_out, _ = self.attn(q, kv.unsqueeze(0), kv.unsqueeze(0))
            out[i] = attn_out.squeeze(0).squeeze(0)

        return out                                   # (B, D)

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
        '''
        src, dst : (B,) LongTensor
        ef       : (B, EDGE_DIM) FloatTensor edge features
        t        : (B,) FloatTensor normalised timestamps
        update   : whether to update history after this batch
        Returns  : (B,) violation probabilities
        '''
        z_src = self._embed(src, t)                 # (B, D)
        z_dst = self._embed(dst, t)                 # (B, D)
        ef_d  = self.edge_proj(ef)                  # (B, D)

        # Fuse: target embedding + source context + edge features
        h = self.merge(torch.cat([z_dst + z_src, ef_d], dim=-1))  # (B, D)

        if update:
            self._update_history(src.tolist(), dst.tolist(), t.tolist())

        return torch.sigmoid(self.head(h)).squeeze(-1)             # (B,)


# =============================================================================
# Training helpers
# =============================================================================
def _df_to_tensors(part_df, feat_cols, scaler, device):
    '''Convert a split DataFrame to tensors for TGAT forward pass.'''
    src = torch.tensor(part_df['src'].values, dtype=torch.long,    device=device)
    dst = torch.tensor(part_df['dst'].values, dtype=torch.long,    device=device)
    ef  = torch.tensor(scaler.transform(part_df[feat_cols].fillna(0).values),
                       dtype=torch.float32, device=device)
    t   = torch.tensor(part_df['tau_norm'].values, dtype=torch.float32, device=device)
    y   = torch.tensor(part_df['label'].values,    dtype=torch.float32, device=device)
    return src, dst, ef, t, y


def train_tgat(model, train_df, val_df, feat_cols, scaler,
               n_epochs=30, lr=1e-3, batch_size=512,
               pos_weight_factor=10.0, device=DEVICE):
    '''Train TGAT with focal-weighted BCE.  Returns (model, val_auprc_history).'''
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_pos = train_df['label'].sum()
    n_neg = len(train_df) - n_pos
    pw    = torch.tensor([pos_weight_factor * n_neg / max(n_pos, 1)],
                          dtype=torch.float32, device=device)
    loss_fn = nn.BCELoss()

    model.to(device)
    history = []

    for ep in range(1, n_epochs + 1):
        model.train()
        model.reset_history()
        idx   = np.random.permutation(len(train_df))
        epoch_loss = 0.0

        for start in range(0, len(idx), batch_size):
            batch_idx  = idx[start:start + batch_size]
            batch_df   = train_df.iloc[batch_idx]
            src, dst, ef, t, y = _df_to_tensors(batch_df, feat_cols, scaler, device)

            opt.zero_grad()
            prob   = model(src, dst, ef, t, update=True)
            weight = torch.where(y == 1, pw.expand_as(y), torch.ones_like(y))
            loss   = (loss_fn(prob, y) * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(batch_df)

        # Validation
        model.eval()
        with torch.no_grad():
            src_v, dst_v, ef_v, t_v, y_v = _df_to_tensors(
                val_df, feat_cols, scaler, device)
            prob_v = model(src_v, dst_v, ef_v, t_v, update=False)
            auprc  = average_precision_score(
                y_v.cpu().numpy(), prob_v.cpu().numpy())
        history.append(auprc)

        if ep % 5 == 0 or ep == 1:
            print(f'  Epoch {ep:3d}/{n_epochs}  loss={epoch_loss/len(train_df):.4f}'
                  f'  val_AUPRC={auprc:.4f}')

    return model, history


@torch.no_grad()
def eval_tgat(model, test_df, feat_cols, scaler, device=DEVICE):
    '''Evaluate TGAT; returns dict with metrics + raw scores.'''
    model.eval()
    src, dst, ef, t, y = _df_to_tensors(test_df, feat_cols, scaler, device)
    prob = model(src, dst, ef, t, update=False).cpu().numpy()
    y_np = y.cpu().numpy().astype(int)
    m    = compute_metrics(y_np, prob)
    m['scores'] = prob
    return m


# =============================================================================
# Experiment L-single: TGAT on real single-project data
# =============================================================================
print('=' * 65)
print('Experiment L -- TGAT on single EPC project')
print('=' * 65)

# ── Reuse df, FEAT_COLS, NUM_NODES from earlier cells ────────────────────────
# df must have: src, dst, tau, tau_norm, label, + FEAT_COLS columns
# These are set up in Experiments B/C/D above.

from sklearn.preprocessing import StandardScaler

scaler_l = StandardScaler()

# ── L-single: stratified split (baseline, comparable to TGN Exp B/C) ─────────
print('\n[L-single] Split: stratified')
tr_s, va_s, te_s = split_dataset(df, method='stratified',
                                  label_col='label', time_col='tau')
split_info(tr_s, va_s, te_s, label_col='label')

scaler_l.fit(tr_s[FEAT_COLS].fillna(0).values)

# Add normalised tau
for part in [tr_s, va_s, te_s]:
    part['tau_norm'] = (part['tau'] - tr_s['tau'].min()) / (
        tr_s['tau'].max() - tr_s['tau'].min() + 1e-9)

model_tgat = TGAT_EPC(
    num_nodes   = NUM_NODES,
    embed_dim   = 64,
    edge_dim    = EDGE_DIM,
    n_heads     = 4,
    n_neighbors = 20,
).to(DEVICE)

model_tgat, hist_l = train_tgat(
    model_tgat, tr_s, va_s, FEAT_COLS, scaler_l,
    n_epochs=30, lr=1e-3, batch_size=512,
    pos_weight_factor=15.0,
)

res_tgat_strat = eval_tgat(model_tgat, te_s, FEAT_COLS, scaler_l)
print(f'\n  Stratified -> AUC={res_tgat_strat["auc"]:.4f}'
      f'  AUPRC={res_tgat_strat["auprc"]:.4f}'
      f'  F1={res_tgat_strat["f1"]:.4f}')

# ── L-inductive: temporal split + new-node evaluation ─────────────────────────
print('\n[L-inductive] Split: inductive (10 % of workers withheld)')
tr_i, va_i, te_i = split_dataset(df, method='inductive',
                                  label_col='label', time_col='tau',
                                  inductive_frac=0.10)
split_info(tr_i, va_i, te_i, label_col='label')

for part in [tr_i, va_i, te_i]:
    part['tau_norm'] = (part['tau'] - tr_i['tau'].min()) / (
        tr_i['tau'].max() - tr_i['tau'].min() + 1e-9)

scaler_li = StandardScaler().fit(tr_i[FEAT_COLS].fillna(0).values)

model_tgat_i = TGAT_EPC(NUM_NODES, 64, EDGE_DIM, 4, 20).to(DEVICE)
model_tgat_i, _ = train_tgat(
    model_tgat_i, tr_i, va_i, FEAT_COLS, scaler_li,
    n_epochs=30, lr=1e-3, batch_size=512, pos_weight_factor=15.0)

res_ind_all  = eval_tgat(model_tgat_i, te_i,                          FEAT_COLS, scaler_li)
res_ind_new  = eval_tgat(model_tgat_i,
                          te_i[te_i['is_new_node'] == 1].reset_index(drop=True),
                          FEAT_COLS, scaler_li)
res_ind_seen = eval_tgat(model_tgat_i,
                          te_i[te_i['is_new_node'] == 0].reset_index(drop=True),
                          FEAT_COLS, scaler_li)

print(f'\n  Inductive eval:')
print(f'    All nodes -> AUC={res_ind_all["auc"]:.4f}'
      f'  AUPRC={res_ind_all["auprc"]:.4f}  F1={res_ind_all["f1"]:.4f}')
print(f'    New nodes -> AUC={res_ind_new["auc"]:.4f}'
      f'  AUPRC={res_ind_new["auprc"]:.4f}  F1={res_ind_new["f1"]:.4f}'
      f'  (n={res_ind_new["n_total"]})')
print(f'    Seen nodes-> AUC={res_ind_seen["auc"]:.4f}'
      f'  AUPRC={res_ind_seen["auprc"]:.4f}  F1={res_ind_seen["f1"]:.4f}')

# ── Model comparison summary ──────────────────────────────────────────────────
print()
print('=' * 65)
print('Experiment L complete.')
print()
print(f'{"Model":12s}  {"Split":12s}  {"AUC":6s}  {"AUPRC":6s}  {"F1":6s}')
print('-' * 50)
try:
    print(f'{"TGN":12s}  {"stratified":12s}  '
          f'{res_tgn["auc"]:.4f}  {res_tgn["auprc"]:.4f}  {res_tgn["f1"]:.4f}')
except NameError:
    print('  (res_tgn not found -- run Experiment B/D first)')
print(f'{"DyRep":12s}  {"stratified":12s}  '
      f'(run Experiment K)')
print(f'{"TGAT":12s}  {"stratified":12s}  '
      f'{res_tgat_strat["auc"]:.4f}  {res_tgat_strat["auprc"]:.4f}  {res_tgat_strat["f1"]:.4f}')
print(f'{"TGAT":12s}  {"inductive":12s}  '
      f'{res_ind_all["auc"]:.4f}  {res_ind_all["auprc"]:.4f}  {res_ind_all["f1"]:.4f}')
print('=' * 65)

# ── Save results ──────────────────────────────────────────────────────────────
results_l = {
    'model': 'TGAT',
    'dataset': 'single_project',
    'stratified': {k: v for k, v in res_tgat_strat.items() if k != 'scores'},
    'inductive':  {
        'all':  {k: v for k, v in res_ind_all.items()  if k != 'scores'},
        'new':  {k: v for k, v in res_ind_new.items()  if k != 'scores'},
        'seen': {k: v for k, v in res_ind_seen.items() if k != 'scores'},
    },
}
save_results(results_l, 'experiments/UseCase4/results/exp_l_tgat_single.json')
"""

# ── Inject into notebook ──────────────────────────────────────────────────────
assert len(nb['cells']) == 47, (
    f"Expected 47 cells (after Exp K), found {len(nb['cells'])}. "
    "Re-run inject_exp_k.py first if needed."
)

def make_markdown(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "exp-l-md",
        "metadata": {},
        "source": src,
    }

def make_code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "exp-l-code",
        "metadata": {},
        "outputs": [],
        "source": src,
    }

nb['cells'].append(make_markdown(MD_L))
nb['cells'].append(make_code(CODE_L))

with NB.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Injected Experiment L (TGAT) as cells 47-48')
print(f'Total cells: {len(nb["cells"])}')
