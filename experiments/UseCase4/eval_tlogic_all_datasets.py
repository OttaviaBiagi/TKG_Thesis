#!/usr/bin/env python3
"""
T-Logic symbolic reasoning evaluation across all dataset variants.
Tests R1, R2, and their combination on single, multi, and multi_varied.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

DATA_DIR = Path('data/UseCase4')
RULE_CHANGE = datetime(2024, 6, 29, tzinfo=timezone.utc)

# Certification requirements
CERT_REQS = {
    'hot_work':       {'Hot Work Safety', 'Fire Watch', 'Welding Certification'},
    'excavation':     {'Excavation Safety', 'Confined Space Entry', 'Soil Assessment'},
    'lifting':        {'Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'},
    'electrical':     {'Electrical Safety', 'LOTO Certification', 'HV Awareness'},
    'confined_space': {'Confined Space Entry', 'Gas Testing', 'Emergency Response'},
    'radiography':    {'NDT Level II', 'Radiation Safety', 'RT Operator'},
    'work_at_height': {'Working at Height', 'Scaffold Inspection', 'Fall Arrest'},
    'general_work':   set(),
}

def load_single_project() -> tuple[dict, dict]:
    """Load real Meram EPC project."""
    with open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8') as f:
        dataset = json.load(f)
    with open(DATA_DIR / 'epc_events.json', encoding='utf-8') as f:
        events = json.load(f)
    return dataset, events


def load_multi_projects() -> tuple[list[dict], list[dict]]:
    """Load all 100 synthetic multi-project datasets."""
    proj_dir = DATA_DIR / 'projects'
    index = json.load(open(proj_dir / 'index.json'))
    
    all_datasets = []
    all_events = []
    for entry in index:
        p_dir = proj_dir / entry['path']
        try:
            ds_p = json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
            ev_p = json.load(open(p_dir / 'events.json', encoding='utf-8'))
            all_datasets.append(ds_p)
            all_events.append(ev_p)
        except Exception as e:
            print(f'  Warning: skipping {entry["path"]} — {e}')
    return all_datasets, all_events


def load_multi_varied_projects() -> tuple[list[dict], list[dict]]:
    """Load all structurally varied multi-project datasets."""
    proj_dir = DATA_DIR / 'projects_varied'
    index = json.load(open(proj_dir / 'index.json'))
    
    all_datasets = []
    all_events = []
    for entry in index:
        p_dir = proj_dir / entry['path']
        try:
            ds_p = json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
            ev_p = json.load(open(p_dir / 'events.json', encoding='utf-8'))
            all_datasets.append(ds_p)
            all_events.append(ev_p)
        except Exception as e:
            print(f'  Warning: skipping {entry["path"]} — {e}')
    return all_datasets, all_events


def build_cert_index(workers: list[dict]) -> dict[str, dict]:
    """Build worker certification validity windows."""
    idx = {}
    for w in workers:
        idx[w['id']] = {}
        for c in w['certifications']:
            cid = c['cert'].replace(' ', '_')
            vf = datetime.fromisoformat(c['valid_from'])
            vt = datetime.fromisoformat(c['valid_to'])
            if vf.tzinfo is None:
                vf = vf.replace(tzinfo=timezone.utc)
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            idx[w['id']][cid] = (vf, vt)
    return idx


def check_r1(worker_id: str, step_id: str, assignment_date: datetime,
             cert_idx: dict, step_map: dict) -> bool:
    """
    R1: Check if worker is missing required cert at assignment date.
    """
    step = step_map.get(step_id, {})
    permit = step.get('permit_type', 'general_work')
    required = CERT_REQS.get(permit, set())
    
    wc = cert_idx.get(worker_id, {})
    for req_cert in required:
        req_cert_normalized = req_cert.replace(' ', '_')
        if req_cert_normalized not in wc:
            return True  # Missing cert
        vf, vt = wc[req_cert_normalized]
        if not (vf <= assignment_date <= vt):
            return True  # Cert expired
    return False


def check_r2(worker_id: str, step_id: str, assignment_date: datetime,
             cert_idx: dict, step_map: dict) -> bool:
    """
    R2: Check post-rule-change hot_work without Advanced_Fire_Watch cert.
    """
    if assignment_date < RULE_CHANGE:
        return False
    
    step = step_map.get(step_id, {})
    permit = step.get('permit_type', 'general_work')
    if permit != 'hot_work':
        return False
    
    wc = cert_idx.get(worker_id, {})
    if 'Advanced_Fire_Watch' not in wc:
        return True
    vf, vt = wc['Advanced_Fire_Watch']
    return not (vf <= assignment_date <= vt)


def evaluate_dataset(dataset: dict, events: dict, dataset_name: str = 'single') -> dict:
    """Evaluate T-Logic rules on a single dataset."""
    workers = dataset['workers']
    steps = dataset['steps']
    assigned = events['assigned_to']
    denied_set = {(e['worker_id'], e['step_id']) for e in events['permit_denied']}
    
    cert_idx = build_cert_index(workers)
    step_map = {s['id']: s for s in steps}
    
    # Predictions
    r1_pred = []
    r2_pred = []
    r_both = []
    y_true = []
    
    for e in assigned:
        wid = e['worker_id']
        sid = e['step_id']
        dt = datetime.fromisoformat(e['date'])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        label = 1 if (wid, sid) in denied_set else 0
        y_true.append(label)
        
        r1 = check_r1(wid, sid, dt, cert_idx, step_map)
        r2 = check_r2(wid, sid, dt, cert_idx, step_map)
        r_both_pred = r1 or r2
        
        r1_pred.append(1 if r1 else 0)
        r2_pred.append(1 if r2 else 0)
        r_both.append(1 if r_both_pred else 0)
    
    y_true = np.array(y_true)
    r1_pred = np.array(r1_pred)
    r2_pred = np.array(r2_pred)
    r_both = np.array(r_both)
    
    def _metrics(y_pred, y_true):
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        }
    
    return {
        'dataset': dataset_name,
        'n_assignments': len(assigned),
        'n_violations': int(y_true.sum()),
        'r1': _metrics(r1_pred, y_true),
        'r2': _metrics(r2_pred, y_true),
        'r1_r2_combined': _metrics(r_both, y_true),
    }


def main():
    print('\n=== T-LOGIC EVALUATION ACROSS ALL DATASETS ===\n')
    
    results = []
    
    # Single project
    print('[1/3] Loading single project...')
    ds, ev = load_single_project()
    m = ds['metadata']
    print(f'  Steps: {m["total_steps"]} | Workers: {m["total_workers"]} | '
          f'Violations: {len(ev["permit_denied"])} / {len(ev["assigned_to"])}')
    r = evaluate_dataset(ds, ev, 'single')
    results.append(r)
    
    # Multi projects
    print('[2/3] Loading multi projects...')
    dss, evs = load_multi_projects()
    print(f'  Loaded {len(dss)} projects')
    # Concatenate all
    all_workers = []
    all_steps = []
    all_assigned = []
    all_denied = []
    for ds, ev in zip(dss, evs):
        all_workers.extend(ds['workers'])
        all_steps.extend(ds['steps'])
        all_assigned.extend(ev['assigned_to'])
        all_denied.extend(ev['permit_denied'])
    
    combined_ds = {
        'workers': all_workers,
        'steps': all_steps,
        'certifications': [],
        'metadata': {
            'total_workers': len(all_workers),
            'total_steps': len(all_steps),
            'total_certs': 0,
            'total_permits': 0,
        }
    }
    combined_ev = {
        'assigned_to': all_assigned,
        'permit_denied': all_denied,
    }
    r = evaluate_dataset(combined_ds, combined_ev, 'multi')
    results.append(r)
    
    # Multi varied
    print('[3/3] Loading multi_varied projects...')
    dss, evs = load_multi_varied_projects()
    print(f'  Loaded {len(dss)} projects')
    # Concatenate all
    all_workers = []
    all_steps = []
    all_assigned = []
    all_denied = []
    for ds, ev in zip(dss, evs):
        all_workers.extend(ds['workers'])
        all_steps.extend(ds['steps'])
        all_assigned.extend(ev['assigned_to'])
        all_denied.extend(ev['permit_denied'])
    
    combined_ds = {
        'workers': all_workers,
        'steps': all_steps,
        'certifications': [],
        'metadata': {
            'total_workers': len(all_workers),
            'total_steps': len(all_steps),
            'total_certs': 0,
            'total_permits': 0,
        }
    }
    combined_ev = {
        'assigned_to': all_assigned,
        'permit_denied': all_denied,
    }
    r = evaluate_dataset(combined_ds, combined_ev, 'multi_varied')
    results.append(r)
    
    # Print summary
    print('\n=== T-LOGIC RESULTS SUMMARY ===\n')
    print(f'{"Dataset":<12} {"Assignments":>12} {"Violations":>10} {"Rule":<15} {"Prec":>6} {"Rec":>6} {"F1":>6}')
    print('-' * 90)
    
    for r in results:
        ds_name = r['dataset']
        n_assign = r['n_assignments']
        n_viol = r['n_violations']
        
        for rule_name, rule_metrics in [('R1', r['r1']), ('R2', r['r2']), ('R1+R2', r['r1_r2_combined'])]:
            prec = rule_metrics['precision']
            rec = rule_metrics['recall']
            f1 = rule_metrics['f1']
            
            if rule_name == 'R1':
                print(f'{ds_name:<12} {n_assign:>12,d} {n_viol:>10d}', end='')
            else:
                print(f'{"":12s} {"":>12s} {"":>10s}', end='')
            
            print(f' {rule_name:<15} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}')
    
    # Save results
    output_path = Path('experiments/UseCase4/results/tlogic_all_datasets.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved -> {output_path}')


if __name__ == '__main__':
    main()
