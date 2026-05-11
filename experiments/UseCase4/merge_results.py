#!/usr/bin/env python3
"""Reconstruct complete benchmark from git history + log files."""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / 'results'

def git_json(commit):
    r = subprocess.run(
        ['git', 'show', f'{commit}:experiments/UseCase4/results/benchmark.json'],
        capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else {'results': []}

def get_auc(m):
    if 'overall' in m: return m['overall'].get('auc', float('nan'))
    return m.get('auc', float('nan'))

def get_auprc(m):
    if 'overall' in m: return m['overall'].get('auprc', float('nan'))
    return m.get('auprc', float('nan'))

def get_f1(m):
    if 'overall' in m: return m['overall'].get('f1', float('nan'))
    return m.get('f1', float('nan'))

# ── Collect from git history (later overrides earlier) ────────────────────────
all_r = {}

# c437b84: original full single benchmark (TGN good, DyRep broken, TGAT all 4)
for r in git_json('c437b84')['results']:
    all_r[(r['model'], r['split'], r['dataset'])] = r

# 77fb9b1: DyRep single rerun (stable, 3/4 splits: stratified/temporal/6slot)
for r in git_json('77fb9b1')['results']:
    all_r[(r['model'], r['split'], r['dataset'])] = r

# a57cd1b: TGN multi (all 4 splits)
for r in git_json('a57cd1b')['results']:
    all_r[(r['model'], r['split'], r['dataset'])] = r

# current benchmark.json: DyRep multi (all 4 splits, AUC=0.500)
curr = json.load(open(RESULTS_DIR / 'benchmark.json'))
for r in curr['results']:
    all_r[(r['model'], r['split'], r['dataset'])] = r

# ── Inject DyRep inductive single from bench_dyrep2.log (not committed) ──────
# Log line: AUC=0.651  AUPRC=0.029  (111s)
# This is the 4th result (inductive) from the stable DyRep rerun
dyrep_inductive_single = {
    'model': 'DyRep', 'split': 'inductive', 'dataset': 'single',
    'train_sec': 111.0, 'n_train': None, 'n_test': None, 'n_pos_test': None,
    'metrics': {'overall': {'auc': 0.651, 'auprc': 0.029, 'f1': float('nan')}},
    'hparams': {'embed_dim': 32, 'n_neighbors': 10, 'lr': 1e-4, 'intensity_reg': 0.001,
                'pos_weight_factor': 10.0, 'batch_size': 256},
    '_source': 'bench_dyrep2.log',
}
all_r[('DyRep', 'inductive', 'single')] = dyrep_inductive_single

# ── Print current state ───────────────────────────────────────────────────────
ALL = [(m, s, d) for d in ('single', 'multi')
       for m in ('TGN', 'DyRep', 'TGAT')
       for s in ('stratified', 'temporal', '6slot', 'inductive')]

print(f'\n{"Model":8s}  {"Split":12s}  {"Dataset":8s}  {"AUC":7s}  {"AUPRC":7s}  {"F1":6s}')
print('-' * 65)
for k in ALL:
    r = all_r.get(k)
    if r is None:
        print(f'{k[0]:8s}  {k[1]:12s}  {k[2]:8s}  MISSING')
        continue
    if r.get('skipped'):
        print(f'{k[0]:8s}  {k[1]:12s}  {k[2]:8s}  SKIPPED')
        continue
    m = r.get('metrics', {})
    print(f'{k[0]:8s}  {k[1]:12s}  {k[2]:8s}  '
          f'{get_auc(m):7.4f}  {get_auprc(m):7.4f}  {get_f1(m):6.4f}')

missing = [k for k in ALL if k not in all_r]
print(f'\nTotal collected: {len(all_r)}/24')
if missing:
    print(f'Still missing:   {missing}')

# ── Save merged CSV ───────────────────────────────────────────────────────────
rows = []
for k in ALL:
    r = all_r.get(k)
    if r is None or r.get('skipped'):
        continue
    m = r.get('metrics', {})
    rows.append({
        'model':      k[0], 'split': k[1], 'dataset': k[2],
        'auc':        round(get_auc(m),   4),
        'auprc':      round(get_auprc(m), 4),
        'f1':         round(get_f1(m),    4),
        'train_sec':  r.get('train_sec'),
        'n_train':    r.get('n_train'),
        'n_test':     r.get('n_test'),
        'n_pos_test': r.get('n_pos_test'),
    })

df = pd.DataFrame(rows)
out_csv  = RESULTS_DIR / 'benchmark_merged.csv'
out_json = RESULTS_DIR / 'benchmark_merged.json'
df.to_csv(out_csv, index=False)
json.dump({'results': list(all_r.values())}, open(out_json, 'w'), indent=2)
print(f'\nSaved -> {out_csv}')
print(f'Saved -> {out_json}')
