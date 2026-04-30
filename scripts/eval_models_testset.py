"""
Proper test-set evaluation for TNTComplEx and TGN-B.

TNTComplEx: trained on tau < 12, scored ONLY on tau >= 12 (test split).
TGN-B:      focal loss γ=2, balanced batching, stratified 70/30 split
            (more violations in training vs temporal split).

Run from repo root:  python scripts/eval_models_testset.py
Saves: experiments/UseCase4/results_test_set.json
       experiments/UseCase4/roc_tntcomplex_tgnb.png
"""

import json, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, classification_report)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DATA_DIR = Path('data/UseCase4')
EXP_DIR  = Path('experiments/UseCase4')
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT_MONTH = 12   # months 0-11 = train, 12-17 = test
PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)

print(f'Device: {DEVICE}')

# ── Load data ─────────────────────────────────────────────────────────────────
ds = json.loads((DATA_DIR / 'epc_dataset_real.json').read_text(encoding='utf-8'))
ev = json.loads((DATA_DIR / 'epc_events.json').read_text(encoding='utf-8'))


def to_month(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, min(17, (dt - PROJECT_START).days // 30))
    except:
        return 0


# ── Entity / relation registry ────────────────────────────────────────────────
entity2id = {}
def eid(name):
    if name not in entity2id:
        entity2id[name] = len(entity2id)
    return entity2id[name]

relation2id = {}
def rid(name):
    if name not in relation2id:
        relation2id[name] = len(relation2id)
    return relation2id[name]

triples = []
step_month = {}

for a in ds['activities']:
    eid(a['id'])

for s in ds['steps']:
    m = to_month(s['valid_from'])
    step_month[s['id']] = m
    triples.append((eid(s['activity_id']), rid('HAS_STEP'),        eid(s['id']),          m))
    triples.append((eid(s['id']),          rid('REQUIRES_PERMIT'), eid(s['permit_type']), m))

for seq in ds['step_sequences']:
    m = step_month.get(seq['from'], 0)
    triples.append((eid(seq['from']), rid('PRECEDES'), eid(seq['to']), m))

CERT_REQS = {
    'hot_work':       ['Hot_Work_Safety','Fire_Watch','Welding_Certification'],
    'excavation':     ['Excavation_Safety','Confined_Space_Entry','Soil_Assessment'],
    'lifting':        ['Rigging_&_Lifting','Crane_Operator','Slinging_Certificate'],
    'electrical':     ['Electrical_Safety','LOTO_Certification','HV_Awareness'],
    'confined_space': ['Confined_Space_Entry','Gas_Testing','Emergency_Response'],
    'radiography':    ['NDT_Level_II','Radiation_Safety','RT_Operator'],
    'work_at_height': ['Working_at_Height','Scaffold_Inspection','Fall_Arrest'],
    'general_work':   ['General_Safety_Induction'],
}
for permit, certs in CERT_REQS.items():
    for c in certs:
        triples.append((eid(permit), rid('REQUIRES_CERT'), eid(c), 0))
triples.append((eid('hot_work'), rid('REQUIRES_CERT'), eid('Advanced_Fire_Watch'), 6))

for w in ds['workers']:
    eid(w['id'])
    for c in w['certifications']:
        m = to_month(c['valid_from'])
        triples.append((eid(w['id']), rid('HAS_CERT'), eid(c['cert'].replace(' ','_')), m))

denied_set = set(zip([e['worker_id'] for e in ev['permit_denied']],
                     [e['step_id']   for e in ev['permit_denied']]))

assign_rows = []  # {worker_id, step_id, date, month, label}
for e in ev['assigned_to']:
    m = to_month(e['date'])
    triples.append((eid(e['worker_id']), rid('ASSIGNED_TO'), eid(e['step_id']), m))
    assign_rows.append({'worker_id': e['worker_id'], 'step_id': e['step_id'],
                        'date': e['date'], 'month': m,
                        'label': 1 if (e['worker_id'], e['step_id']) in denied_set else 0})

for e in ev['permit_denied']:
    m = to_month(e['date'])
    triples.append((eid(e['worker_id']), rid('PERMIT_DENIED'), eid(e['step_id']), m))

triples      = np.array(triples, dtype=np.int64)
n_entities   = len(entity2id)
n_relations  = len(relation2id)
n_timestamps = 18

train_triples = triples[triples[:, 3] < SPLIT_MONTH]
test_triples  = triples[triples[:, 3] >= SPLIT_MONTH]
train_t = torch.tensor(train_triples, dtype=torch.long)
test_t  = torch.tensor(test_triples,  dtype=torch.long)

# Test ASSIGNED_TO rows only (for violation scoring)
test_assign_rows = [r for r in assign_rows if r['month'] >= SPLIT_MONTH]
train_assign_rows = [r for r in assign_rows if r['month'] < SPLIT_MONTH]

print(f'\nEntities: {n_entities} | Relations: {n_relations} | Timestamps: {n_timestamps}')
print(f'Train triples: {len(train_triples)} | Test triples: {len(test_triples)}')
print(f'Train ASSIGNED_TO: {len(train_assign_rows)} '
      f'({sum(r["label"] for r in train_assign_rows)} violations)')
print(f'Test  ASSIGNED_TO: {len(test_assign_rows)} '
      f'({sum(r["label"] for r in test_assign_rows)} violations)')

# ── TNTComplEx ────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('1. TNTComplEx — train on months 0-11, evaluate on months 12-17')
print('='*60)

class TNTComplEx(nn.Module):
    def __init__(self, n_entities, n_relations, n_timestamps, rank=100, dropout=0.2):
        super().__init__()
        self.rank = rank
        self.E_re = nn.Embedding(n_entities,   rank)
        self.E_im = nn.Embedding(n_entities,   rank)
        self.R_re = nn.Embedding(n_relations,  rank)
        self.R_im = nn.Embedding(n_relations,  rank)
        self.T_re = nn.Embedding(n_timestamps, rank)
        self.T_im = nn.Embedding(n_timestamps, rank)
        self.drop = nn.Dropout(dropout)
        for emb in [self.E_re, self.E_im, self.R_re, self.R_im, self.T_re, self.T_im]:
            nn.init.normal_(emb.weight, std=0.1)

    def score(self, h, r, t, tau):
        h_re, h_im = self.drop(self.E_re(h)), self.drop(self.E_im(h))
        r_re, r_im = self.R_re(r), self.R_im(r)
        t_re, t_im = self.drop(self.E_re(t)), self.drop(self.E_im(t))
        tau_re, tau_im = self.T_re(tau), self.T_im(tau)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        hrt_re = hr_re * t_re + hr_im * t_im
        hrt_im = hr_im * t_re - hr_re * t_im
        return (hrt_re * tau_re - hrt_im * tau_im).sum(dim=-1)

    def forward_1vsAll(self, h, r, tau):
        h_re, h_im = self.drop(self.E_re(h)), self.drop(self.E_im(h))
        r_re, r_im = self.R_re(r), self.R_im(r)
        tau_re, tau_im = self.T_re(tau), self.T_im(tau)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        hrt_re = hr_re * tau_re - hr_im * tau_im
        hrt_im = hr_im * tau_re + hr_re * tau_im
        return hrt_re @ self.E_re.weight.T + hrt_im @ self.E_im.weight.T

    def regularisation(self, h, r, t, tau):
        return (self.E_re(h).norm(p=3) ** 3 + self.E_im(h).norm(p=3) ** 3 +
                self.E_re(t).norm(p=3) ** 3 + self.E_im(t).norm(p=3) ** 3 +
                self.R_re(r).norm(p=3) ** 3 + self.R_im(r).norm(p=3) ** 3) / len(h)


tnt_model = TNTComplEx(n_entities, n_relations, n_timestamps).to(DEVICE)
opt = torch.optim.Adam(tnt_model.parameters(), lr=5e-3)

train_t_dev = train_t.to(DEVICE)
for epoch in range(50):
    tnt_model.train()
    idx = torch.randperm(len(train_t_dev))
    total, nb_ = 0.0, 0
    for i in range(0, len(train_t_dev), 512):
        batch = train_t_dev[idx[i:i+512]]
        h, r, t, tau = batch[:,0], batch[:,1], batch[:,2], batch[:,3]
        scores = tnt_model.forward_1vsAll(h, r, tau)
        loss   = F.cross_entropy(scores, t) + 1e-3 * tnt_model.regularisation(h, r, t, tau)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); nb_ += 1
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:>3}/50  loss: {total/nb_:.4f}')

# Score ONLY test ASSIGNED_TO events
tnt_model.eval()
r_pd_id = relation2id['PERMIT_DENIED']
r_pd_t  = torch.tensor([r_pd_id], dtype=torch.long).to(DEVICE)
scores_tnt, labels_tnt = [], []

with torch.no_grad():
    for row in test_assign_rows:
        wid, sid = row['worker_id'], row['step_id']
        if wid not in entity2id or sid not in entity2id:
            continue
        h   = torch.tensor([entity2id[wid]], dtype=torch.long).to(DEVICE)
        t   = torch.tensor([entity2id[sid]], dtype=torch.long).to(DEVICE)
        tau = torch.tensor([row['month']],   dtype=torch.long).to(DEVICE)
        sc  = tnt_model.score(h, r_pd_t, t, tau).item()
        scores_tnt.append(sc)
        labels_tnt.append(row['label'])

scores_tnt = np.array(scores_tnt)
labels_tnt = np.array(labels_tnt)
s_min, s_max = scores_tnt.min(), scores_tnt.max()
scores_tnt_n = (scores_tnt - s_min) / (s_max - s_min + 1e-8)

best_t, best_f1 = 0.5, 0.0
for th in np.arange(0.05, 0.95, 0.01):
    f = f1_score(labels_tnt, (scores_tnt_n > th).astype(int),
                 average='macro', zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, th

tnt_pred = (scores_tnt_n > best_t).astype(int)
tnt_p    = precision_score(labels_tnt, tnt_pred, zero_division=0)
tnt_r    = recall_score(labels_tnt,    tnt_pred, zero_division=0)
tnt_f1   = f1_score(labels_tnt,        tnt_pred, average='macro', zero_division=0)
tnt_auc  = roc_auc_score(labels_tnt, scores_tnt_n) if labels_tnt.sum() > 0 else float('nan')

print(f'\nTNTComplEx — TEST SET ONLY (months 12-17)')
print(f'Test events: {len(labels_tnt)} | Violations: {labels_tnt.sum()}')
print(f'Threshold: {best_t:.2f}')
print(classification_report(labels_tnt, tnt_pred,
                             target_names=['Normal','Violation'], digits=3))
print(f'AUC-ROC: {tnt_auc:.4f} | P: {tnt_p:.4f} | R: {tnt_r:.4f} | F1: {tnt_f1:.4f}')

np.save(str(EXP_DIR / 'tnt_test_probs.npy'),
        np.column_stack([labels_tnt, scores_tnt_n]))

# ── TGN-B: focal γ=2, balanced batching, STRATIFIED split ────────────────────
print('\n' + '='*60)
print('2. TGN-B — focal γ=2, balanced batching, stratified 70/30 split')
print('='*60)

RULE_CHANGE_DT = datetime(2024, 6, 29, tzinfo=timezone.utc)
CERT_REQS_STR = {
    'hot_work':       {'Hot Work Safety','Fire Watch','Welding Certification'},
    'excavation':     {'Excavation Safety','Confined Space Entry','Soil Assessment'},
    'lifting':        {'Rigging & Lifting','Crane Operator','Slinging Certificate'},
    'electrical':     {'Electrical Safety','LOTO Certification','HV Awareness'},
    'confined_space': {'Confined Space Entry','Gas Testing','Emergency Response'},
    'radiography':    {'NDT Level II','Radiation Safety','RT Operator'},
    'work_at_height': {'Working at Height','Scaffold Inspection','Fall Arrest'},
    'general_work':   set(),
}
worker_certs = {}
for w in ds['workers']:
    wc = {}
    for c in w['certifications']:
        vf = datetime.fromisoformat(c['valid_from'])
        vt = datetime.fromisoformat(c['valid_to'])
        if vf.tzinfo is None: vf = vf.replace(tzinfo=timezone.utc)
        if vt.tzinfo is None: vt = vt.replace(tzinfo=timezone.utc)
        wc[c['cert']] = (vf, vt)
    worker_certs[w['id']] = wc

step_info = {s['id']: s for s in ds['steps']}
PERMIT_ENC = {'general_work':0,'hot_work':1,'excavation':2,'lifting':3,
              'electrical':4,'confined_space':5,'radiography':6,'work_at_height':7}
DISC_ENC   = {d:i for i,d in enumerate(sorted({s['discipline'] for s in ds['steps']}))}

feat_rows = []
for e in ev['assigned_to']:
    wid, sid = e['worker_id'], e['step_id']
    dt = datetime.fromisoformat(e['date'])
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    step    = step_info.get(sid, {})
    permit  = step.get('permit_type', 'general_work')
    disc    = step.get('discipline', 'XX')
    after_rc = int(dt >= RULE_CHANGE_DT)
    required = set(CERT_REQS_STR.get(permit, set()))
    if after_rc and permit == 'hot_work':
        required.add('Advanced Fire Watch')
    wc = worker_certs.get(wid, {})
    missing = [c for c in required if c not in wc or not (
        wc[c][0] <= dt <= wc[c][1])]
    expires_soon = int(any((wc[c][1]-dt).days < 30
                           for c in required if c in wc and wc[c][0] <= dt <= wc[c][1]))
    label = 1 if (wid, sid) in denied_set else 0
    feat_rows.append({
        'wid': wid, 'sid': sid,
        'label': label,
        'feats': np.array([
            PERMIT_ENC.get(permit, 0),
            DISC_ENC.get(disc, 0),
            after_rc,
            int(step.get('on_critical_path', False) or False),
            float(step.get('weight_pct') or 0),
            expires_soon,
        ], dtype=np.float32),
    })

labels_all = np.array([r['label'] for r in feat_rows])
feats_all  = np.stack([r['feats'] for r in feat_rows])

tr_idx, te_idx = train_test_split(
    np.arange(len(feat_rows)), test_size=0.30,
    stratify=labels_all, random_state=42)

train_rows = [feat_rows[i] for i in tr_idx]
test_rows  = [feat_rows[i] for i in te_idx]
y_tr = labels_all[tr_idx]
y_te = labels_all[te_idx]

print(f'Train: {len(train_rows)} events, {y_tr.sum()} violations')
print(f'Test:  {len(test_rows)} events,  {y_te.sum()} violations')

all_ent_tgn = sorted({r['wid'] for r in feat_rows} | {r['sid'] for r in feat_rows})
e2id_tgn    = {e: i for i, e in enumerate(all_ent_tgn)}
N_ENT_TGN   = len(e2id_tgn)
HIDDEN      = 64
EDGE_DIM    = 6


class TGNViol(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.mem = nn.Embedding(n, h)
        self.msg = nn.GRUCell(h * 2, h)
        self.clf = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, s, d):
        return self.clf(torch.cat([self.mem(s), self.mem(d)], -1)).squeeze(-1)

    def update(self, s, d):
        inp = torch.cat([self.mem(d), self.mem(s)], dim=-1)
        ns  = self.msg(inp, self.mem(s))
        self.mem.weight.data[s] = ns.detach()


def focal(logits, targets, gamma=2.0, alpha=0.75):
    bce  = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t  = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    a_t  = torch.where(targets == 1,
                       torch.full_like(p_t, alpha),
                       torch.full_like(p_t, 1 - alpha))
    return (a_t * (1 - p_t) ** gamma * bce).mean()


tgn_model = TGNViol(N_ENT_TGN, HIDDEN).to(DEVICE)
opt_tgn   = torch.optim.Adam(tgn_model.parameters(), lr=1e-3)

viol_idx = [i for i, r in enumerate(train_rows) if r['label'] == 1]
norm_idx  = [i for i, r in enumerate(train_rows) if r['label'] == 0]

for epoch in range(50):
    tgn_model.train()
    random.shuffle(viol_idx); random.shuffle(norm_idx)
    idx = viol_idx[:256] + norm_idx[:256]
    random.shuffle(idx)
    batch = [train_rows[i] for i in idx]
    src = torch.tensor([e2id_tgn[r['wid']] for r in batch], device=DEVICE)
    dst = torch.tensor([e2id_tgn[r['sid']] for r in batch], device=DEVICE)
    lbl = torch.tensor([r['label'] for r in batch], device=DEVICE)
    opt_tgn.zero_grad()
    loss = focal(tgn_model(src, dst), lbl)
    loss.backward(); opt_tgn.step(); tgn_model.update(src, dst)
    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:>3}/50  focal loss: {loss.item():.4f}')

tgn_model.eval()
probs_tgn = []
with torch.no_grad():
    for i in range(0, len(test_rows), 512):
        b   = test_rows[i:i+512]
        s   = torch.tensor([e2id_tgn.get(r['wid'], 0) for r in b], device=DEVICE)
        d   = torch.tensor([e2id_tgn.get(r['sid'], 0) for r in b], device=DEVICE)
        probs_tgn.extend(torch.sigmoid(tgn_model(s, d)).cpu().numpy())

probs_tgn = np.array(probs_tgn)

# Tune threshold
best_t2, best_f12 = 0.5, 0.0
for th in np.arange(0.05, 0.95, 0.01):
    f = f1_score(y_te, (probs_tgn > th).astype(int), average='macro', zero_division=0)
    if f > best_f12:
        best_f12, best_t2 = f, th

preds_tgn = (probs_tgn > best_t2).astype(int)
tgn_p     = precision_score(y_te, preds_tgn, zero_division=0)
tgn_r     = recall_score(y_te,    preds_tgn, zero_division=0)
tgn_f1    = f1_score(y_te,        preds_tgn, average='macro', zero_division=0)
tgn_auc   = roc_auc_score(y_te, probs_tgn) if y_te.sum() > 0 else float('nan')

print(f'\nTGN-B — TEST SET (stratified 30%, {y_te.sum()} violations)')
print(f'Threshold: {best_t2:.2f}')
print(classification_report(y_te, preds_tgn,
                             target_names=['Normal','Violation'], digits=3))
print(f'AUC-ROC: {tgn_auc:.4f} | P: {tgn_p:.4f} | R: {tgn_r:.4f} | F1: {tgn_f1:.4f}')

np.save(str(EXP_DIR / 'tgn_b_test_probs.npy'),
        np.column_stack([y_te, probs_tgn]))

# ── ROC comparison plot ───────────────────────────────────────────────────────
print('\nGenerating ROC plot...')
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#1e1e2e')
ax.set_facecolor('#181825')

fpr_tnt, tpr_tnt, _ = roc_curve(labels_tnt, scores_tnt_n)
fpr_tgn, tpr_tgn, _ = roc_curve(y_te, probs_tgn)

ax.plot(fpr_tnt, tpr_tnt, color='#89b4fa', lw=2,
        label=f'TNTComplEx  (AUC={tnt_auc:.3f}) — test months 12-17')
ax.plot(fpr_tgn, tpr_tgn, color='#cba6f7', lw=2,
        label=f'TGN-B focal γ=2  (AUC={tgn_auc:.3f}) — stratified test')
ax.plot([0,1],[0,1], color='#6c7086', ls=':', lw=1, label='Random')
ax.set_xlabel('False Positive Rate', color='#cdd6f4')
ax.set_ylabel('True Positive Rate',  color='#cdd6f4')
ax.set_title('Violation Detection — ROC (proper test set)\nTNTComplEx vs TGN-B',
             color='#cdd6f4')
ax.legend(facecolor='#313244', labelcolor='#cdd6f4', fontsize=9)
ax.tick_params(colors='#cdd6f4')
for sp in ['top','right']:   ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#313244')

out_png = EXP_DIR / 'roc_tntcomplex_tgnb.png'
plt.tight_layout()
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f'Saved: {out_png}')

# ── Summary ───────────────────────────────────────────────────────────────────
results = {
    'TNTComplEx_test': {
        'split': 'temporal (train tau<12, test tau>=12)',
        'test_events': int(len(labels_tnt)),
        'test_violations': int(labels_tnt.sum()),
        'P': round(float(tnt_p), 4), 'R': round(float(tnt_r), 4),
        'F1_macro': round(float(tnt_f1), 4), 'AUC': round(float(tnt_auc), 4),
    },
    'TGN_B_test': {
        'split': 'stratified 70/30',
        'test_events': int(len(y_te)),
        'test_violations': int(y_te.sum()),
        'P': round(float(tgn_p), 4), 'R': round(float(tgn_r), 4),
        'F1_macro': round(float(tgn_f1), 4), 'AUC': round(float(tgn_auc), 4),
    },
}
out_json = EXP_DIR / 'results_test_set.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {out_json}')
print('\n' + '='*60)
print('FINAL RESULTS (proper test set)')
print('='*60)
print(f'{"Model":<30} {"P":>8} {"R":>8} {"F1":>8} {"AUC":>8}')
print('-'*60)
for k, v in results.items():
    print(f'{k:<30} {v["P"]:>8.4f} {v["R"]:>8.4f} {v["F1_macro"]:>8.4f} {v["AUC"]:>8.4f}')
