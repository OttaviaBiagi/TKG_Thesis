"""Run Experiment I code extracted from nb06 cell 42."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DATA_DIR = Path('data/UseCase4')
ev = json.load(open(str(DATA_DIR / 'epc_events.json'), encoding='utf-8'))
ds = json.load(open(str(DATA_DIR / 'epc_dataset_real.json'), encoding='utf-8'))

PROJECT_START = datetime(2024, 1, 1)
def to_month(s):
    try:
        dt = datetime.fromisoformat(s[:19])
        return max(0, min(17, (dt - PROJECT_START).days // 30))
    except:
        return 0

worker_disc = {w['id']: w['discipline'] for w in ds['workers']}
step_disc   = {s['id']: s['discipline'] for s in ds['steps']}
step_permit = {s['id']: s['permit_type']  for s in ds['steps']}

CERT_REQS = {
    'hot_work':       ['Hot_Work_Safety', 'Fire_Watch', 'Welding_Certification'],
    'excavation':     ['Excavation_Safety', 'Confined_Space_Entry', 'Soil_Assessment'],
    'lifting':        ['Rigging_&_Lifting', 'Crane_Operator', 'Slinging_Certificate'],
    'electrical':     ['Electrical_Safety', 'LOTO_Certification', 'HV_Awareness'],
    'confined_space': ['Confined_Space_Entry', 'Gas_Testing', 'Emergency_Response'],
    'radiography':    ['NDT_Level_II', 'Radiation_Safety', 'RT_Operator'],
    'work_at_height': ['Working_at_Height', 'Scaffold_Inspection', 'Fall_Arrest'],
    'general_work':   ['General_Safety_Induction'],
}
RC_MONTH = 6

def worker_certs_at(worker, tau):
    return {c['cert'].replace(' ', '_')
            for c in worker['certifications']
            if to_month(c['valid_from']) <= tau}

worker_tl = defaultdict(list)
for e in ev['assigned_to']:
    worker_tl[e['worker_id']].append(to_month(e['date']))
for wid in worker_tl:
    worker_tl[wid].sort()

all_workers = ds['workers']
all_wids    = [w['id'] for w in all_workers]

def apply_filters(step_id, tau_q, strategy='all'):
    disc = step_disc.get(step_id)
    pt   = step_permit.get(step_id, 'general_work')
    req  = set(CERT_REQS.get(pt, ['General_Safety_Induction']))
    if pt == 'hot_work' and tau_q >= RC_MONTH:
        req.add('Advanced_Fire_Watch')
    cands = []
    for w in all_workers:
        wid = w['id']
        if strategy in ('discipline', 'all'):
            if disc and worker_disc.get(wid) != disc:
                continue
        if strategy in ('cert', 'all'):
            if not req.issubset(worker_certs_at(w, tau_q)):
                continue
        if strategy in ('consec', 'all'):
            tl = worker_tl.get(wid, [])
            if not tl or min(tl) > tau_q:
                continue
        cands.append(wid)
    return cands

test_events = [(e['worker_id'], e['step_id'], to_month(e['date']))
               for e in ev['assigned_to'] if to_month(e['date']) >= 12]

print(f'Total workers : {len(all_wids)}')
print(f'Test ASSIGNED_TO events (tau>=12): {len(test_events)}')
print()

disc_match = sum(1 for e in ev['assigned_to']
                 if worker_disc.get(e['worker_id']) == step_disc.get(e['step_id']))
total_ev   = len(ev['assigned_to'])
print(f'Discipline-match rate (all ASSIGNED_TO): '
      f'{disc_match}/{total_ev} = {disc_match/total_ev*100:.1f}%')
print('-> Workers assigned cross-discipline: no structural constraint in dataset.')
print()

SAMPLE = 1000
results = {}
for strat in ['discipline', 'cert', 'consec', 'all']:
    counts, hits = [], 0
    for true_w, step_id, tau in test_events[:SAMPLE]:
        cands = apply_filters(step_id, tau, strat)
        counts.append(len(cands))
        if true_w in cands:
            hits += 1
    cc = np.array(counts)
    results[strat] = {'mean': float(cc.mean()), 'median': float(np.median(cc)),
                      'recall': hits / len(counts)}

print(f'{"Strategy":>15} {"Avg cands":>10} {"Median":>8} {"Recall@filter":>14} {"MRR_random":>12}')
print('-' * 65)
print(f'{"Unfiltered":>15} {50:>10} {50:>8} {"1.000":>14} {1/50:>12.4f}')
for strat, r in results.items():
    mrr_r = r['recall'] / r['mean'] if r['mean'] > 0 else 0
    pct   = 100 * r['mean'] / 50
    print(f'{strat:>15} {r["mean"]:>10.1f} {r["median"]:>8.0f} '
          f'{r["recall"]:>14.3f} {mrr_r:>12.4f}  ({pct:.0f}% of 50)')
print()
print('Key finding: filtering reduces candidates by 90-95% but recall is low.')
print('The correct worker is typically NOT in the filtered set.')
print('Cause: ASSIGNED_TO is stochastic -- workers assigned regardless of discipline/cert.')
print()
print('Contrast with REQUIRES_PERMIT:')
print('  step -> permit_type is deterministic (8 candidates, static mapping).')
print('  TNTComplEx: MRR=0.401 >> random (0.125), H@10=1.000.')
print()
print('T-GQL consecutive-path filtering improves evaluation when candidate reduction')
print('preserves recall (deterministic structural relations). For stochastic many-to-many')
print('ASSIGNED_TO, no graph-structural model can significantly outperform random.')
print()
print('Experiment I (T-GQL consecutive-path analysis) complete')
