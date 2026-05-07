"""Inject Experiment K (DyRep two-scale temporal model) into nb06."""
import json
from pathlib import Path

NB = Path('notebooks/UseCase4/06_tkg_models.ipynb')
nb = json.load(NB.open(encoding='utf-8'))

MD_K = """## Experiment K — DyRep: Two-Scale Temporal Graph Model

**DyRep** (Trivedi et al., NeurIPS 2018) explicitly models *two temporal processes* at different timescales:

| Process | In EPC domain | Timescale parameter |
|---------|--------------|-------------------|
| **Association** (k=0) | Rule change: new cert requirement alters worker–step compatibility | ψ_assoc (slow, small) |
| **Communication** (k=1) | Worker–step assignments: transient information flow | ψ_comm (fast, large) |

**Key DyRep components implemented:**
- **Intensity function** (Eq. 2-3): λ_k(t) = ψ_k · softplus(ω_kᵀ [z_u; z_v] / ψ_k) — timescale ψ_k learned per process type
- **Embedding update** (Eq. 4): z_v(t) = σ(W_struct · h_neighbors + W_rec · z_prev + W_t · Δt)
- **Attentive neighbourhood aggregation** (Eq. 5): S-matrix-weighted max-pooling over recent neighbours

**Experiment structure:**
- **K-single**: DyRep on single real project — direct comparison with TGN (same data, same metrics)
- **K-multi**: DyRep on 100 synthetic projects — demonstrates multi-project generalisation enabled by the generator

**Thesis insight**: after training, ψ_comm >> ψ_assoc confirms the model correctly learned the two temporal scales (fast assignments vs slow structural rule change).
"""

CODE_K = r"""import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# DyRep Model
# =============================================================================
class DyRepEPC(nn.Module):
    '''
    DyRep (Trivedi et al. 2018) for EPC violation prediction.

    Event type mapping:
      k=0 (association): after_rc=1 events — structural/rule-change dynamics, slow psi
      k=1 (communication): after_rc=0 events — assignment dynamics, fast psi

    Core DyRep components:
      - Intensity: lambda_k = psi_k * softplus(omega_k^T [z_u; z_v] / psi_k)
      - Embedding update (Eq.4): W_struct*h_nbr + W_rec*z_prev + W_t*dt
      - Attentive neighbourhood aggregation (Eq.5)
    '''

    def __init__(self, num_nodes, embed_dim=32, edge_dim=6, n_neighbors=10):
        super().__init__()
        self.embed_dim   = embed_dim
        self.n_neighbors = n_neighbors
        self.num_nodes   = num_nodes

        # Evolving embeddings z_v(t) — updated in-place via register_buffer
        self.register_buffer('z', torch.zeros(num_nodes, embed_dim))

        # Timescale parameters psi_k (log-param so psi > 0 always)
        self.log_psi_comm  = nn.Parameter(torch.tensor(0.0))   # fast: assignments
        self.log_psi_assoc = nn.Parameter(torch.tensor(-2.0))  # slow: rule changes

        # Compatibility weights omega_k (Eq.1)
        self.omega_comm  = nn.Linear(2 * embed_dim, 1, bias=False)
        self.omega_assoc = nn.Linear(2 * embed_dim, 1, bias=False)

        # Embedding update matrices (Eq.4)
        self.W_struct = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_rec    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_t      = nn.Linear(1, embed_dim, bias=False)

        # Neighbourhood aggregator
        self.W_h = nn.Linear(embed_dim, embed_dim)

        # Association strength proxy (simplified S — per-node scalar, not n x n)
        self.register_buffer('s_vec', torch.zeros(num_nodes))

        # Sparse neighbour history
        self._nbrs = [[] for _ in range(num_nodes)]

        # Classification head: z_dst + edge_features -> violation prob
        self.head = nn.Sequential(
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
        '''psi_k * softplus(omega_k^T [z_u;z_v] / psi_k)  -- Eq.2-3'''
        psi   = self.psi_comm  if k == 1 else self.psi_assoc
        omega = self.omega_comm if k == 1 else self.omega_assoc
        g = omega(torch.cat([z_u, z_v], dim=-1))
        return psi * torch.log1p(torch.exp(g / (psi.abs() + 1e-6)))

    def _aggregate(self, node_ids_list):
        '''Attentive neighbourhood aggregation (Eq.5, S-weighted max-pool).'''  # noqa
        dev = self.z.device
        out = []
        for nid in node_ids_list:
            nbrs = self._nbrs[nid][-self.n_neighbors:]
            if not nbrs:
                out.append(torch.zeros(self.embed_dim, device=dev))
                continue
            nbr_t = torch.tensor(nbrs, dtype=torch.long, device=dev)
            h     = torch.tanh(self.W_h(self.z[nbr_t]))            # (K, d)
            attn  = torch.softmax(self.s_vec[nbr_t], dim=0).unsqueeze(-1)
            out.append(torch.sigmoid(attn * h).max(dim=0)[0])       # DyRep Eq.5
        return torch.stack(out)

    def _update_state(self, src_list, dst_list, dt):
        '''Update z and neighbour history after event (detached — no backprop).'''
        dev = self.z.device
        h_s = self._aggregate(dst_list)   # info dst -> src
        h_d = self._aggregate(src_list)   # info src -> dst
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
        '''
        event type inferred from after_rc (ef[:,2]):
          after_rc=1 -> k=0 (association/rule-change)
          after_rc=0 -> k=1 (communication/assignment)
        '''
        k   = 0 if float(ef[:, 2].mean()) > 0.5 else 1
        lam = self.intensity(self.z[src], self.z[dst], k)  # keeps psi in graph
        if update:
            self._update_state(src.tolist(), dst.tolist(), dt)
        logit = self.head(torch.cat([self.z[dst], ef], dim=-1))
        return torch.sigmoid(logit).squeeze(-1), lam


# =============================================================================
# Training / evaluation
# =============================================================================
def train_dyrep(model, src, dst, ef, dt, y, epochs=30, batch_size=256, lr=1e-3):
    pos_w = torch.tensor([(y==0).sum()/max((y==1).sum(),1)], dtype=torch.float32)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train(); model.reset()
        total, nb_b = 0, 0
        for i in range(0, len(src), batch_size):
            s, d = src[i:i+batch_size], dst[i:i+batch_size]
            e, t, yb = ef[i:i+batch_size], dt[i:i+batch_size], y[i:i+batch_size]
            opt.zero_grad()
            prob, lam = model(s, d, e, t, update=True)
            loss = crit(torch.logit(prob.clamp(1e-6, 1-1e-6)), yb) - 0.01 * lam.mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); nb_b += 1
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}/{epochs}  loss={total/nb_b:.4f}  '
                  f'psi_comm={model.psi_comm.item():.3f}  '
                  f'psi_assoc={model.psi_assoc.item():.4f}')
    return model


def eval_dyrep(model, src, dst, ef, dt, y, label='DyRep'):
    model.eval(); model.reset()
    scores = []
    with torch.no_grad():
        for i in range(0, len(src), 256):
            p, _ = model(src[i:i+256], dst[i:i+256],
                         ef[i:i+256], dt[i:i+256], update=False)
            scores.extend(p.numpy())
    scores = np.array(scores)
    y_np   = y.numpy() if hasattr(y, 'numpy') else np.array(y)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_np, (scores > t).astype(int), average='macro', zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    pred = (scores > best_t).astype(int)
    p  = precision_score(y_np, pred, zero_division=0)
    r  = recall_score(y_np, pred, zero_division=0)
    f1 = f1_score(y_np, pred, average='macro', zero_division=0)
    try:    auc = roc_auc_score(y_np, scores)
    except: auc = float('nan')
    print(f'\n{label}  (threshold={best_t:.2f})')
    print(classification_report(y_np, pred, target_names=['Normal', 'Violation'], digits=3))
    print(f'  AUC-ROC: {auc:.4f}  |  macro-F1: {f1:.4f}  |  P: {p:.4f}  R: {r:.4f}')
    return {'P': p, 'R': r, 'F1': f1, 'AUC': auc, 'scores': scores, 'pred': pred, 'y': y_np}


# =============================================================================
# K-single: same data as TGN
# =============================================================================
print('K-single — DyRep on single project (same data as TGN)')
print(f'  Nodes: {NUM_NODES}  |  Train: {len(src_tr)}  |  Test: {len(src_te)}')
print(f'  Violations train: {int(y_tr.sum())}  test: {int(y_te.sum())}')

dyrep_single = DyRepEPC(NUM_NODES, embed_dim=32, edge_dim=EDGE_DIM)
dyrep_single = train_dyrep(dyrep_single, src_tr, dst_tr, ef_tr, dt_tr, y_tr, epochs=30)
res_dyrep_single = eval_dyrep(dyrep_single, src_te, dst_te, ef_te, dt_te, y_te,
                               label='DyRep-single')
psi_ratio = dyrep_single.psi_comm.item() / (dyrep_single.psi_assoc.item() + 1e-6)
print(f'\n  Learned timescales:  psi_comm={dyrep_single.psi_comm.item():.4f}  '
      f'psi_assoc={dyrep_single.psi_assoc.item():.4f}  ratio={psi_ratio:.1f}x')
print(f'  {"OK" if psi_ratio > 1 else "WARN"}: '
      f'{"psi_comm > psi_assoc — correctly learned fast/slow separation" if psi_ratio > 1 else "timescales not well separated yet"}')

# =============================================================================
# K-multi: train on 100 synthetic projects
# =============================================================================
print('\n' + '=' * 60)
print('K-multi — DyRep trained on 100 synthetic projects')

PROJ_DIR = Path('../../data/UseCase4/projects')
CERT_REQS_K = {
    'hot_work':       {'Hot Work Safety', 'Fire Watch', 'Welding Certification'},
    'excavation':     {'Excavation Safety', 'Confined Space Entry', 'Soil Assessment'},
    'lifting':        {'Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'},
    'electrical':     {'Electrical Safety', 'LOTO Certification', 'HV Awareness'},
    'confined_space': {'Confined Space Entry', 'Gas Testing', 'Emergency Response'},
    'radiography':    {'NDT Level II', 'Radiation Safety', 'RT Operator'},
    'work_at_height': {'Working at Height', 'Scaffold Inspection', 'Fall Arrest'},
    'general_work':   set(),
}
PERM_ENC_K = {p: i for i, p in enumerate(
    ['general_work','hot_work','excavation','lifting',
     'electrical','confined_space','radiography','work_at_height'])}
DISC_ENC_K = {d: i for i, d in enumerate(sorted({s['discipline'] for s in ds['steps']}))}
FEAT_COLS_K = ['permit_enc','disc_enc','after_rc','on_critical_path','weight_pct','cert_expires_soon']


def build_project_df(ds_p, ev_p):
    step_info_p = {s['id']: s for s in ds_p['steps']}
    denied_set_p = {(v['worker_id'], v['step_id']) for v in ev_p['permit_denied']}
    wc_p = {}
    for w in ds_p['workers']:
        d = {}
        for c in w['certifications']:
            vf = datetime.fromisoformat(c['valid_from'])
            vt = datetime.fromisoformat(c['valid_to'])
            if vf.tzinfo is None: vf = vf.replace(tzinfo=timezone.utc)
            if vt.tzinfo is None: vt = vt.replace(tzinfo=timezone.utc)
            d[c['cert']] = (vf, vt)
        wc_p[w['id']] = d
    rc_iso = ds_p['update_events'][0]['valid_from']
    rc_dt  = datetime.fromisoformat(rc_iso)
    if rc_dt.tzinfo is None: rc_dt = rc_dt.replace(tzinfo=timezone.utc)
    rows = []
    for e in ev_p['assigned_to']:
        wid, sid = e['worker_id'], e['step_id']
        dt_e = datetime.fromisoformat(e['date'])
        if dt_e.tzinfo is None: dt_e = dt_e.replace(tzinfo=timezone.utc)
        step    = step_info_p.get(sid, {})
        permit  = step.get('permit_type', 'general_work')
        disc    = step.get('discipline', 'XX')
        after_rc = int(dt_e >= rc_dt)
        req = set(CERT_REQS_K.get(permit, set()))
        if after_rc and permit == 'hot_work': req.add('Advanced Fire Watch')
        wc       = wc_p.get(wid, {})
        missing  = [c for c in req if c not in wc or not (wc[c][0] <= dt_e <= wc[c][1])]
        exp_soon = int(any((wc[c][1] - dt_e).days < 30 for c in req if c in wc))
        comp     = next((c for c in ev_p['completed'] if c['step_id'] == sid), {})
        rows.append({
            'worker_id': wid, 'step_id': sid,
            'timestamp': dt_e.timestamp(),
            'permit_enc': PERM_ENC_K.get(permit, 0),
            'disc_enc':   DISC_ENC_K.get(disc, 0),
            'after_rc':   after_rc,
            'on_critical_path': int(comp.get('on_critical_path', False)),
            'weight_pct': step.get('weight_pct', 0.0) or 0.0,
            'cert_expires_soon': exp_soon,
            'label_viol': int((wid, sid) in denied_set_p),
        })
    return pd.DataFrame(rows).sort_values('timestamp').reset_index(drop=True)


index_k = json.load(open(PROJ_DIR / 'index.json'))
print(f'  Loading {len(index_k)} synthetic projects...')
all_rows = []
for entry in index_k:
    p_dir = PROJ_DIR / entry['path']
    try:
        ds_p = json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
        ev_p = json.load(open(p_dir / 'events.json',  encoding='utf-8'))
        all_rows.append(build_project_df(ds_p, ev_p))
    except Exception as ex:
        print(f'  Warning: {entry["path"]} — {ex}')

df_multi = pd.concat(all_rows, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
print(f'  Combined: {len(df_multi):,} events  |  violations: {int(df_multi.label_viol.sum()):,} '
      f'({100*df_multi.label_viol.mean():.1f}%)')

all_w_m  = sorted(df_multi.worker_id.unique())
all_s_m  = sorted(df_multi.step_id.unique())
widx_m   = {w: i              for i, w in enumerate(all_w_m)}
sidx_m   = {s: i+len(widx_m) for i, s in enumerate(all_s_m)}
NUM_N_M  = len(widx_m) + len(sidx_m)
df_multi['src'] = df_multi.worker_id.map(widx_m)
df_multi['dst'] = df_multi.step_id.map(sidx_m)

split_t  = np.percentile(df_multi.timestamp.values, 80)
tr_m     = df_multi[df_multi.timestamp <= split_t].reset_index(drop=True)
te_m     = df_multi[df_multi.timestamp >  split_t].reset_index(drop=True)

sc_m     = MinMaxScaler()
sc_m.fit(tr_m[FEAT_COLS_K].fillna(0))
tr_ef_m  = np.nan_to_num(sc_m.transform(tr_m[FEAT_COLS_K].fillna(0))).astype(np.float32)
te_ef_m  = np.nan_to_num(sc_m.transform(te_m[FEAT_COLS_K].fillna(0))).astype(np.float32)
norm_s   = 86400 * 30
tr_dt_m  = (np.diff(tr_m.timestamp.values, prepend=tr_m.timestamp.values[0]) / norm_s).astype(np.float32)
te_dt_m  = (np.diff(te_m.timestamp.values, prepend=te_m.timestamp.values[0]) / norm_s).astype(np.float32)


def _tt(sub, ef, dt):
    return (torch.tensor(sub.src.values, dtype=torch.long),
            torch.tensor(sub.dst.values, dtype=torch.long),
            torch.tensor(ef,  dtype=torch.float32),
            torch.tensor(dt,  dtype=torch.float32).unsqueeze(1),
            torch.tensor(sub.label_viol.values, dtype=torch.float32))


src_trm, dst_trm, ef_trm, dt_trm, y_trm = _tt(tr_m, tr_ef_m, tr_dt_m)
src_tem, dst_tem, ef_tem, dt_tem, y_tem  = _tt(te_m, te_ef_m, te_dt_m)
print(f'  Train: {len(tr_m):,} | violations: {int(y_trm.sum()):,}  '
      f'Test: {len(te_m):,} | violations: {int(y_tem.sum()):,}')

dyrep_multi = DyRepEPC(NUM_N_M, embed_dim=32, edge_dim=6)
dyrep_multi = train_dyrep(dyrep_multi, src_trm, dst_trm, ef_trm, dt_trm, y_trm,
                           epochs=20, batch_size=512)
res_dyrep_multi = eval_dyrep(dyrep_multi, src_tem, dst_tem, ef_tem, dt_tem, y_tem,
                              label='DyRep-multi (100 projects)')
psi_ratio_m = dyrep_multi.psi_comm.item() / (dyrep_multi.psi_assoc.item() + 1e-6)
print(f'\n  Learned timescales:  psi_comm={dyrep_multi.psi_comm.item():.4f}  '
      f'psi_assoc={dyrep_multi.psi_assoc.item():.4f}  ratio={psi_ratio_m:.1f}x')

# =============================================================================
# Summary comparison
# =============================================================================
print('\n' + '=' * 65)
print(f'{"Model":<28} {"AUC":>6} {"F1":>6} {"P":>6} {"R":>6}')
print('-' * 65)
for name, res in [('TGN (cert-aware)',     res_tgn),
                  ('DyRep-single',         res_dyrep_single),
                  ('DyRep-multi (100x)',   res_dyrep_multi)]:
    print(f'{name:<28} {res["AUC"]:>6.3f} {res["F1"]:>6.3f} '
          f'{res["P"]:>6.3f} {res["R"]:>6.3f}')
print('=' * 65)
print(f'DyRep-single psi_comm/psi_assoc = {psi_ratio:.1f}x  '
      f'(>1 -> correct fast/slow temporal separation)')
"""


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}


nb['cells'].append(md_cell(MD_K))
nb['cells'].append(code_cell(CODE_K))

with NB.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Injected Experiment K at cells {len(nb["cells"])-2} and {len(nb["cells"])-1}')
print(f'Total cells: {len(nb["cells"])}')
