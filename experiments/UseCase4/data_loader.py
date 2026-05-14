"""
EPC data loader for the benchmark runner.

Functions
---------
load_single_project(data_dir) -> pd.DataFrame
    Loads real Meram EPC data, returns a standardized event DataFrame.

load_multi_project(data_dir) -> pd.DataFrame
    Loads all 100 synthetic projects, concatenates into one DataFrame.

Both return DataFrames with columns:
    src, dst, tau, label, label_viol,
    permit_enc, disc_enc, after_rc, on_critical_path, weight_pct,
    cert_expires_soon, worker_id, step_id

The 'label' column is identical to 'label_viol' (alias for eval_framework).
'tau' is the Unix timestamp (float) used for temporal splitting and ordering.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEAT_COLS = [
    'permit_enc', 'disc_enc', 'after_rc', 'on_critical_path',
    'weight_pct', 'cert_expires_soon',
]
EDGE_DIM = len(FEAT_COLS)

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
PERMIT_ENCODE = {p: i for i, p in enumerate(sorted(CERT_REQS.keys()))}


def _build_disc_encode(steps: list[dict]) -> dict[str, int]:
    return {d: i for i, d in enumerate(sorted({s['discipline'] for s in steps}))}


def _build_worker_certs(workers: list[dict]) -> dict:
    wc = {}
    for w in workers:
        d = {}
        for c in w['certifications']:
            vf = datetime.fromisoformat(c['valid_from'])
            vt = datetime.fromisoformat(c['valid_to'])
            if vf.tzinfo is None: vf = vf.replace(tzinfo=timezone.utc)
            if vt.tzinfo is None: vt = vt.replace(tzinfo=timezone.utc)
            d[c['cert']] = (vf, vt)
        wc[w['id']] = d
    return wc


def _events_to_df(
    assigned_to: list[dict],
    step_info: dict,
    worker_certs: dict,
    denied_set: set,
    completed_map: dict,
    disc_encode: dict,
    rule_change: datetime,
) -> pd.DataFrame:
    records = []
    for e in assigned_to:
        wid = e['worker_id']
        sid = e['step_id']
        dt  = datetime.fromisoformat(e['date'])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        step     = step_info.get(sid, {})
        permit   = step.get('permit_type', 'general_work')
        disc     = step.get('discipline', 'XX')
        after_rc = int(dt >= rule_change)

        req = set(CERT_REQS.get(permit, set()))
        if after_rc and permit == 'hot_work':
            req.add('Advanced Fire Watch')

        wc      = worker_certs.get(wid, {})
        missing = [c for c in req if c not in wc or not (wc[c][0] <= dt <= wc[c][1])]
        exp_soon = int(any((wc[c][1] - dt).days < 30 for c in req if c in wc))

        comp  = completed_map.get(sid, {})
        on_cp = int(comp.get('on_critical_path', False))
        wt    = float(comp.get('weight_pct', 0.0) or 0.0)

        records.append({
            'worker_id':         wid,
            'step_id':           sid,
            'tau':               float(dt.timestamp()),
            'permit_enc':        PERMIT_ENCODE.get(permit, 0),
            'disc_enc':          disc_encode.get(disc, 0),
            'after_rc':          after_rc,
            'on_critical_path':  on_cp,
            'weight_pct':        wt,
            'cert_expires_soon': exp_soon,
            'label_viol':        int((wid, sid) in denied_set),
        })

    df = pd.DataFrame(records).sort_values('tau').reset_index(drop=True)
    df['label'] = df['label_viol']
    return df


def _add_node_indices(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add src/dst node index columns; return (df, num_nodes)."""
    all_workers = sorted(df['worker_id'].unique())
    all_steps   = sorted(df['step_id'].unique())
    widx = {w: i for i, w in enumerate(all_workers)}
    sidx = {s: i + len(widx) for i, s in enumerate(all_steps)}
    df = df.copy()
    df['src'] = df['worker_id'].map(widx)
    df['dst'] = df['step_id'].map(sidx)
    return df, len(widx) + len(sidx)


# ── Public API ────────────────────────────────────────────────────────────────

def load_single_project(data_dir: str | Path = 'data/UseCase4') -> pd.DataFrame:
    """
    Load real Meram EPC data from epc_dataset_real.json + epc_events.json.

    Returns a standardized DataFrame with all columns needed by the benchmark.
    """
    data_dir = Path(data_dir)
    ds = json.load(open(data_dir / 'epc_dataset_real.json', encoding='utf-8'))
    ev = json.load(open(data_dir / 'epc_events.json',       encoding='utf-8'))

    disc_encode   = _build_disc_encode(ds['steps'])
    worker_certs  = _build_worker_certs(ds['workers'])
    step_info     = {s['id']: s for s in ds['steps']}
    denied_set    = {(v['worker_id'], v['step_id']) for v in ev['permit_denied']}
    completed_map = {c['step_id']: c for c in ev['completed']}

    rc_str  = ds['update_events'][0]['valid_from']
    rc_dt   = datetime.fromisoformat(rc_str)
    if rc_dt.tzinfo is None:
        rc_dt = rc_dt.replace(tzinfo=timezone.utc)

    df = _events_to_df(ev['assigned_to'], step_info, worker_certs,
                       denied_set, completed_map, disc_encode, rc_dt)
    df, num_nodes = _add_node_indices(df)
    df.attrs['num_nodes'] = num_nodes
    df.attrs['edge_dim']  = EDGE_DIM
    df.attrs['feat_cols'] = FEAT_COLS

    print(f'[single] {len(df):,} events  '
          f'violations={int(df["label"].sum())} ({100*df["label"].mean():.1f}%)  '
          f'num_nodes={num_nodes}')
    return df


def load_multi_project(data_dir: str | Path = 'data/UseCase4') -> pd.DataFrame:
    """
    Load all synthetic projects from data/UseCase4/projects/ and concatenate.

    Each project's worker/step IDs are made globally unique by prefixing with
    the project_id so that src/dst node indices are globally consistent.
    """
    data_dir  = Path(data_dir)
    proj_dir  = data_dir / 'projects'
    index     = json.load(open(proj_dir / 'index.json'))

    all_dfs = []
    for entry in index:
        p_dir = proj_dir / entry['path']
        try:
            ds_p = json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
            ev_p = json.load(open(p_dir / 'events.json',  encoding='utf-8'))
        except Exception as exc:
            print(f'  Warning: skipping {entry["path"]} — {exc}')
            continue

        pid           = ds_p['project']['project_id']
        disc_encode   = _build_disc_encode(ds_p['steps'])
        worker_certs  = _build_worker_certs(ds_p['workers'])
        step_info     = {s['id']: s for s in ds_p['steps']}
        denied_set    = {(v['worker_id'], v['step_id']) for v in ev_p['permit_denied']}
        completed_map = {c['step_id']: c for c in ev_p['completed']}

        rc_str = ds_p['update_events'][0]['valid_from']
        rc_dt  = datetime.fromisoformat(rc_str)
        if rc_dt.tzinfo is None:
            rc_dt = rc_dt.replace(tzinfo=timezone.utc)

        part = _events_to_df(ev_p['assigned_to'], step_info, worker_certs,
                              denied_set, completed_map, disc_encode, rc_dt)
        # Make IDs globally unique by prefixing project tag
        part['worker_id'] = pid + ':' + part['worker_id'].astype(str)
        part['step_id']   = pid + ':' + part['step_id'].astype(str)
        all_dfs.append(part)

    df = pd.concat(all_dfs, ignore_index=True).sort_values('tau').reset_index(drop=True)
    df, num_nodes = _add_node_indices(df)
    df.attrs['num_nodes'] = num_nodes
    df.attrs['edge_dim']  = EDGE_DIM
    df.attrs['feat_cols'] = FEAT_COLS

    print(f'[multi]  {len(df):,} events  '
          f'violations={int(df["label"].sum()):,} ({100*df["label"].mean():.1f}%)  '
          f'num_nodes={num_nodes}')
    return df


def load_multi_varied(data_dir: str | Path = 'data/UseCase4') -> pd.DataFrame:
    """
    Load structurally varied multi-project dataset from projects_varied/.

    Each project uses 50–80% of Meram families → different step counts,
    different discipline mixes, genuinely different graph topologies.
    Worker/step IDs are prefixed with project_id to keep them globally unique.
    """
    data_dir = Path(data_dir)
    proj_dir = data_dir / 'projects_varied'
    index    = json.load(open(proj_dir / 'index.json'))

    all_dfs = []
    for entry in index:
        p_dir = proj_dir / entry['path']
        try:
            ds_p = json.load(open(p_dir / 'dataset.json', encoding='utf-8'))
            ev_p = json.load(open(p_dir / 'events.json',  encoding='utf-8'))
        except Exception as exc:
            print(f'  Warning: skipping {entry["path"]} — {exc}')
            continue

        pid           = ds_p['project']['project_id']
        disc_encode   = _build_disc_encode(ds_p['steps'])
        worker_certs  = _build_worker_certs(ds_p['workers'])
        step_info     = {s['id']: s for s in ds_p['steps']}
        denied_set    = {(v['worker_id'], v['step_id']) for v in ev_p['permit_denied']}
        completed_map = {c['step_id']: c for c in ev_p['completed']}

        rc_str = ds_p['update_events'][0]['valid_from']
        rc_dt  = datetime.fromisoformat(rc_str)
        if rc_dt.tzinfo is None:
            rc_dt = rc_dt.replace(tzinfo=timezone.utc)

        part = _events_to_df(ev_p['assigned_to'], step_info, worker_certs,
                              denied_set, completed_map, disc_encode, rc_dt)
        part['worker_id'] = pid + ':' + part['worker_id'].astype(str)
        part['step_id']   = pid + ':' + part['step_id'].astype(str)
        all_dfs.append(part)

    df = pd.concat(all_dfs, ignore_index=True).sort_values('tau').reset_index(drop=True)
    df, num_nodes = _add_node_indices(df)
    df.attrs['num_nodes'] = num_nodes
    df.attrs['edge_dim']  = EDGE_DIM
    df.attrs['feat_cols'] = FEAT_COLS

    print(f'[varied] {len(df):,} events  '
          f'violations={int(df["label"].sum()):,} ({100*df["label"].mean():.1f}%)  '
          f'num_nodes={num_nodes}')
    return df
