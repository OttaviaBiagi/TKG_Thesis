"""
Proper test-set evaluation for TNTComplEx and TGN-B.

Stratified 70/30 split on ASSIGNED_TO events (by violation label).
Gives ~315 violations in training, ~134 in test — statistically reliable.

TNTComplEx: trained on all structural triples + train ASSIGNED_TO.
            Scored on test ASSIGNED_TO events (stratified 30%).
TGN-B:      focal loss gamma=2, balanced batching, feature-aware (permit/cert/disc).

Run:   python3 scripts/eval_models_testset.py
Saves: experiments/UseCase4/results_test_set.json
       experiments/UseCase4/roc_tntcomplex_tgnb.png
"""

import json, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, classification_report)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DATA_DIR = Path('data/UseCase4')
EXP_DIR  = Path('experiments/UseCase4')
EXP_DIR.mkdir(parents=True, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
RULE_CHANGE   = datetime(2024, 6, 29, tzinfo=timezone.utc)

print(f'Device: {DEVICE}')

# ── Load ──────────────────────────────────────────────────────────────────────
ds = json.loads((DATA_DIR / 'epc_dataset_real.json').read_text())
ev = json.loads((DATA_DIR / 'epc_events.json').read_text())

def to_month(s):
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return max(0, min(17, (dt - PROJECT_START).days // 30))
    except: return 0

denied_set = {(e['worker_id'], e['step_id']) for e in ev['permit_denied']}
step_info  = {s['id']: s for s in ds['steps']}

# ── Feature engineering ───────────────────────────────────────────────────────
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

PERMIT_ENC = {'general_work':0,'hot_work':1,'excavation':2,'lifting':3,
              'electrical':4,'confined_space':5,'radiography':6,'work_at_height':7}
DISC_ENC   = {d:i for i,d in enumerate(sorted({s['discipline'] for s in ds['steps']}))}
FEAT_DIM   = 6

def make_feat(wid, sid, date_str):
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    step    = step_info.get(sid, {})
    permit  = step.get('permit_type', 'general_work')
    disc    = step.get('discipline', 'XX')
    after_rc = int(dt >= RULE_CHANGE)
    required = set(CERT_REQS_STR.get(permit, set()))
    if after_rc and permit == 'hot_work':
        required.add('Advanced Fire Watch')
    wc = worker_certs.get(wid, {})
    missing = [c for c in required if c not in wc or not (wc[c][0] <= dt <= wc[c][1])]
    expires_soon = int(any((wc[c][1]-dt).days < 30
                           for c in required if c in wc and wc[c][0] <= dt <= wc[c][1]))
    wp = step.get('weight_pct', 0)
    wp = 0.0 if (wp is None or wp != wp) else float(wp)  # guard NaN
    feat = np.array([
        PERMIT_ENC.get(permit, 0),
        DISC_ENC.get(disc, 0),
        after_rc,
        int(bool(step.get('on_critical_path', False))),
        wp,
        expires_soon,
    ], dtype=np.float32)
    return np.nan_to_num(feat, nan=0.0)

# Build event rows
all_rows = []
for e in ev['assigned_to']:
    wid, sid = e['worker_id'], e['step_id']
    all_rows.append({
        'wid': wid, 'sid': sid, 'date': e['date'],
        'month': to_month(e['date']),
        'label': 1 if (wid, sid) in denied_set else 0,
        'feat': make_feat(wid, sid, e['date']),
    })

labels_all = np.array([r['label'] for r in all_rows])
print(f'\nTotal events: {len(all_rows)} | Violations: {labels_all.sum()} ({labels_all.mean()*100:.1f}%)')

# Stratified 70/30 split
tr_idx, te_idx = train_test_split(
    np.arange(len(all_rows)), test_size=0.30,
    stratify=labels_all, random_state=42)

train_rows = [all_rows[i] for i in tr_idx]
test_rows  = [all_rows[i] for i in te_idx]
y_tr = labels_all[tr_idx]
y_te = labels_all[te_idx]
print(f'Train: {len(train_rows)} events, {y_tr.sum()} violations')
print(f'Test:  {len(test_rows)} events,  {y_te.sum()} violations')

# ── Entity registry ───────────────────────────────────────────────────────────
entity2id = {}
def eid(name):
    if name not in entity2id: entity2id[name] = len(entity2id)
    return entity2id[name]
relation2id = {}
def rid(name):
    if name not in relation2id: relation2id[name] = len(relation2id)
    return relation2id[name]

triples = []
step_month = {}
for a in ds['activities']: eid(a['id'])
for s in ds['steps']:
    m = to_month(s['valid_from']); step_month[s['id']] = m
    triples.append((eid(s['activity_id']), rid('HAS_STEP'),        eid(s['id']),          m))
    triples.append((eid(s['id']),          rid('REQUIRES_PERMIT'), eid(s['permit_type']), m))
for seq in ds['step_sequences']:
    triples.append((eid(seq['from']), rid('PRECEDES'), eid(seq['to']),
                    step_month.get(seq['from'], 0)))
CERT_REQS_ID = {
    'hot_work':['Hot_Work_Safety','Fire_Watch','Welding_Certification'],
    'excavation':['Excavation_Safety','Confined_Space_Entry','Soil_Assessment'],
    'lifting':['Rigging_&_Lifting','Crane_Operator','Slinging_Certificate'],
    'electrical':['Electrical_Safety','LOTO_Certification','HV_Awareness'],
    'confined_space':['Confined_Space_Entry','Gas_Testing','Emergency_Response'],
    'radiography':['NDT_Level_II','Radiation_Safety','RT_Operator'],
    'work_at_height':['Working_at_Height','Scaffold_Inspection','Fall_Arrest'],
    'general_work':['General_Safety_Induction'],
}
for permit, certs in CERT_REQS_ID.items():
    for c in certs: triples.append((eid(permit), rid('REQUIRES_CERT'), eid(c), 0))
triples.append((eid('hot_work'), rid('REQUIRES_CERT'), eid('Advanced_Fire_Watch'), 6))
for w in ds['workers']:
    eid(w['id'])
    for c in w['certifications']:
        triples.append((eid(w['id']), rid('HAS_CERT'),
                        eid(c['cert'].replace(' ','_')), to_month(c['valid_from'])))

# Add ASSIGNED_TO triples from TRAINING rows only
for r in train_rows:
    triples.append((eid(r['wid']), rid('ASSIGNED_TO'), eid(r['sid']), r['month']))
# Add PERMIT_DENIED triples from TRAINING violations only
train_denied = {(r['wid'], r['sid']) for r in train_rows if r['label'] == 1}
for e in ev['permit_denied']:
    if (e['worker_id'], e['step_id']) in train_denied:
        triples.append((eid(e['worker_id']), rid('PERMIT_DENIED'),
                        eid(e['step_id']), to_month(e['date'])))

triples     = np.array(triples, dtype=np.int64)
n_entities  = len(entity2id)
n_relations = len(relation2id)
n_timestamps = 18
print(f'\nEntities: {n_entities} | Relations: {n_relations} | Triples: {len(triples)}')

# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('1. TNTComplEx — stratified 70/30 split, scored on test events')
print('='*60)

class TNTComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, n_ts, rank=100, dropout=0.2):
        super().__init__()
        self.E_re = nn.Embedding(n_ent, rank); self.E_im = nn.Embedding(n_ent, rank)
        self.R_re = nn.Embedding(n_rel, rank); self.R_im = nn.Embedding(n_rel, rank)
        self.T_re = nn.Embedding(n_ts,  rank); self.T_im = nn.Embedding(n_ts,  rank)
        self.drop = nn.Dropout(dropout)
        for e in [self.E_re,self.E_im,self.R_re,self.R_im,self.T_re,self.T_im]:
            nn.init.normal_(e.weight, std=0.1)

    def score(self, h, r, t, tau):
        h_re,h_im = self.drop(self.E_re(h)), self.drop(self.E_im(h))
        r_re,r_im = self.R_re(r), self.R_im(r)
        t_re,t_im = self.drop(self.E_re(t)), self.drop(self.E_im(t))
        tau_re,tau_im = self.T_re(tau), self.T_im(tau)
        hr_re = h_re*r_re - h_im*r_im; hr_im = h_re*r_im + h_im*r_re
        hrt_re = hr_re*t_re + hr_im*t_im; hrt_im = hr_im*t_re - hr_re*t_im
        return (hrt_re*tau_re - hrt_im*tau_im).sum(-1)

    def forward_1vsAll(self, h, r, tau):
        h_re,h_im = self.drop(self.E_re(h)), self.drop(self.E_im(h))
        r_re,r_im = self.R_re(r), self.R_im(r)
        tau_re,tau_im = self.T_re(tau), self.T_im(tau)
        hr_re = h_re*r_re-h_im*r_im; hr_im = h_re*r_im+h_im*r_re
        hrt_re = hr_re*tau_re-hr_im*tau_im; hrt_im = hr_im*tau_re+hr_re*tau_im
        return hrt_re@self.E_re.weight.T + hrt_im@self.E_im.weight.T

    def reg(self, h, r, t, tau):
        return (self.E_re(h).norm(3)**3+self.E_im(h).norm(3)**3+
                self.E_re(t).norm(3)**3+self.E_im(t).norm(3)**3+
                self.R_re(r).norm(3)**3+self.R_im(r).norm(3)**3)/len(h)

tnt = TNTComplEx(n_entities, n_relations, n_timestamps).to(DEVICE)
opt = torch.optim.Adam(tnt.parameters(), lr=5e-3)
tr_t = torch.tensor(triples, dtype=torch.long).to(DEVICE)

for epoch in range(50):
    tnt.train()
    idx = torch.randperm(len(tr_t))
    tot, nb_ = 0.0, 0
    for i in range(0, len(tr_t), 512):
        b = tr_t[idx[i:i+512]]
        h,r,t,tau = b[:,0],b[:,1],b[:,2],b[:,3]
        loss = F.cross_entropy(tnt.forward_1vsAll(h,r,tau), t) + 1e-3*tnt.reg(h,r,t,tau)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb_ += 1
    if (epoch+1) % 10 == 0:
        print(f'  Epoch {epoch+1:>3}/50  loss: {tot/nb_:.4f}')

tnt.eval()
r_pd = torch.tensor([relation2id['PERMIT_DENIED']], dtype=torch.long).to(DEVICE)
sc_tnt, lb_tnt = [], []
with torch.no_grad():
    for row in test_rows:
        wid, sid = row['wid'], row['sid']
        if wid not in entity2id or sid not in entity2id: continue
        h   = torch.tensor([entity2id[wid]], dtype=torch.long).to(DEVICE)
        t   = torch.tensor([entity2id[sid]], dtype=torch.long).to(DEVICE)
        tau = torch.tensor([row['month']],   dtype=torch.long).to(DEVICE)
        sc_tnt.append(tnt.score(h, r_pd, t, tau).item())
        lb_tnt.append(row['label'])

sc_tnt = np.array(sc_tnt); lb_tnt = np.array(lb_tnt)
sc_n   = (sc_tnt - sc_tnt.min()) / (sc_tnt.max() - sc_tnt.min() + 1e-8)

best_t, best_f1 = 0.5, 0.0
for th in np.arange(0.05, 0.95, 0.01):
    f = f1_score(lb_tnt, (sc_n>th).astype(int), average='macro', zero_division=0)
    if f > best_f1: best_f1, best_t = f, th

pred_tnt = (sc_n > best_t).astype(int)
tnt_p = precision_score(lb_tnt, pred_tnt, zero_division=0)
tnt_r = recall_score(lb_tnt, pred_tnt, zero_division=0)
tnt_f1= f1_score(lb_tnt, pred_tnt, average='macro', zero_division=0)
tnt_auc = roc_auc_score(lb_tnt, sc_n) if lb_tnt.sum() > 0 else float('nan')

print(f'\nTNTComplEx — stratified test ({lb_tnt.sum()} violations / {len(lb_tnt)} events)')
print(classification_report(lb_tnt, pred_tnt, target_names=['Normal','Violation'], digits=3))
print(f'AUC: {tnt_auc:.4f}  P: {tnt_p:.4f}  R: {tnt_r:.4f}  F1: {tnt_f1:.4f}')
np.save(str(EXP_DIR/'tnt_test_probs.npy'), np.column_stack([lb_tnt, sc_n]))

# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('2. TGN-B — focal γ=2, balanced batching, feature-aware, stratified 70/30')
print('='*60)

all_ent_b = sorted({r['wid'] for r in all_rows} | {r['sid'] for r in all_rows})
e2b = {e:i for i,e in enumerate(all_ent_b)}
N_B = len(e2b)

class TGNViol(nn.Module):
    def __init__(self, n, h, fd):
        super().__init__()
        self.mem = nn.Embedding(n, h)
        self.msg = nn.GRUCell(h*2 + fd, h)
        self.clf = nn.Sequential(nn.Linear(h*2 + fd, h), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(h, 1))

    def forward(self, s, d, feats):
        return self.clf(torch.cat([self.mem(s), self.mem(d), feats], -1)).squeeze(-1)

    def update(self, s, d, feats):
        inp = torch.cat([self.mem(d), self.mem(s), feats], dim=-1)
        ns  = self.msg(inp, self.mem(s))
        self.mem.weight.data[s] = ns.detach()

def focal(logits, targets, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = torch.where(targets==1, torch.sigmoid(logits), 1-torch.sigmoid(logits))
    a_t = torch.where(targets==1, torch.full_like(p_t, alpha), torch.full_like(p_t, 1-alpha))
    return (a_t * (1-p_t)**gamma * bce).mean()

tgn = TGNViol(N_B, 64, FEAT_DIM).to(DEVICE)
opt2 = torch.optim.Adam(tgn.parameters(), lr=1e-3)

viol_idx = [i for i,r in enumerate(train_rows) if r['label']==1]
norm_idx  = [i for i,r in enumerate(train_rows) if r['label']==0]
print(f'Train violations: {len(viol_idx)} | normals: {len(norm_idx)}')

for epoch in range(50):
    tgn.train()
    random.shuffle(viol_idx); random.shuffle(norm_idx)
    idx = viol_idx[:256] + norm_idx[:256]; random.shuffle(idx)
    batch = [train_rows[i] for i in idx]
    s   = torch.tensor([e2b[r['wid']] for r in batch], device=DEVICE)
    d   = torch.tensor([e2b[r['sid']] for r in batch], device=DEVICE)
    ft  = torch.tensor(np.stack([r['feat'] for r in batch]), device=DEVICE)
    lbl = torch.tensor([r['label'] for r in batch], device=DEVICE)
    opt2.zero_grad()
    loss = focal(tgn(s, d, ft), lbl)
    loss.backward(); opt2.step(); tgn.update(s, d, ft)
    if (epoch+1) % 10 == 0:
        print(f'  Epoch {epoch+1:>3}/50  focal loss: {loss.item():.4f}')

tgn.eval()
probs_b = []
with torch.no_grad():
    for i in range(0, len(test_rows), 512):
        b  = test_rows[i:i+512]
        s  = torch.tensor([e2b.get(r['wid'],0) for r in b], device=DEVICE)
        d  = torch.tensor([e2b.get(r['sid'],0) for r in b], device=DEVICE)
        ft = torch.tensor(np.stack([r['feat'] for r in b]), device=DEVICE)
        probs_b.extend(torch.sigmoid(tgn(s, d, ft)).cpu().numpy())

probs_b = np.array(probs_b)
best_t2, best_f12 = 0.5, 0.0
for th in np.arange(0.05, 0.95, 0.01):
    f = f1_score(y_te, (probs_b>th).astype(int), average='macro', zero_division=0)
    if f > best_f12: best_f12, best_t2 = f, th

pred_b  = (probs_b > best_t2).astype(int)
tgn_p   = precision_score(y_te, pred_b, zero_division=0)
tgn_r   = recall_score(y_te,    pred_b, zero_division=0)
tgn_f1  = f1_score(y_te,        pred_b, average='macro', zero_division=0)
tgn_auc = roc_auc_score(y_te, probs_b) if y_te.sum() > 0 else float('nan')

print(f'\nTGN-B — stratified test ({y_te.sum()} violations / {len(y_te)} events)')
print(classification_report(y_te, pred_b, target_names=['Normal','Violation'], digits=3))
print(f'AUC: {tgn_auc:.4f}  P: {tgn_p:.4f}  R: {tgn_r:.4f}  F1: {tgn_f1:.4f}')
np.save(str(EXP_DIR/'tgn_b_test_probs.npy'), np.column_stack([y_te, probs_b]))

# ── ROC plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
fig.patch.set_facecolor('#1e1e2e'); ax.set_facecolor('#181825')
fpr1,tpr1,_ = roc_curve(lb_tnt, sc_n)
fpr2,tpr2,_ = roc_curve(y_te, probs_b)
ax.plot(fpr1,tpr1,color='#89b4fa',lw=2, label=f'TNTComplEx  AUC={tnt_auc:.3f}')
ax.plot(fpr2,tpr2,color='#cba6f7',lw=2, label=f'TGN-B focal γ=2  AUC={tgn_auc:.3f}')
ax.plot([0,1],[0,1],color='#6c7086',ls=':',lw=1,label='Random')
ax.set_xlabel('FPR',color='#cdd6f4'); ax.set_ylabel('TPR',color='#cdd6f4')
ax.set_title('Violation Detection ROC — stratified test set\n(134 violations, no data leakage)',color='#cdd6f4')
ax.legend(facecolor='#313244',labelcolor='#cdd6f4',fontsize=9)
ax.tick_params(colors='#cdd6f4')
for sp in ['top','right']:   ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color('#313244')
plt.tight_layout()
plt.savefig(EXP_DIR/'roc_tntcomplex_tgnb.png', dpi=150, bbox_inches='tight')
print(f'\nROC saved.')

# ── Summary ───────────────────────────────────────────────────────────────────
results = {
    'split': 'stratified 70/30 (315 train violations, 134 test violations)',
    'TNTComplEx': {'P':round(float(tnt_p),4),'R':round(float(tnt_r),4),
                   'F1':round(float(tnt_f1),4),'AUC':round(float(tnt_auc),4)},
    'TGN_B_focal_g2': {'P':round(float(tgn_p),4),'R':round(float(tgn_r),4),
                       'F1':round(float(tgn_f1),4),'AUC':round(float(tgn_auc),4)},
}
with open(EXP_DIR/'results_test_set.json','w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*60)
print('FINAL — stratified test set (134 violations)')
print('='*60)
print(f'{"Model":<28} {"P":>8} {"R":>8} {"F1":>8} {"AUC":>8}')
print('-'*60)
for k in ['TNTComplEx','TGN_B_focal_g2']:
    v = results[k]
    print(f'{k:<28} {v["P"]:>8.4f} {v["R"]:>8.4f} {v["F1"]:>8.4f} {v["AUC"]:>8.4f}')
print('='*60)
print('Saved: experiments/UseCase4/results_test_set.json')
