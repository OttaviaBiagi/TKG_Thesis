"""
Multi-project synthetic dataset generator for TKG training.

Generates N complete EPC project instances by varying:
  - random seed           → different timing, assignments, delays
  - delay_profile         → mild / moderate / severe
  - rule_change_month     → 4 / 6 / 8  (when hot_work cert changes)
  - n_workers             → 30 / 50 / 70

Each project saved as:
  data/UseCase4/projects/proj_NNN/dataset.json
  data/UseCase4/projects/proj_NNN/events.json

Unified index:
  data/UseCase4/projects/index.json
"""

import json, random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))

import generate_epc_dataset as ged
import simulate_events      as sev

PROJECT_START = ged.PROJECT_START
CRITICAL_DISC = {'ME', 'PI', 'EL', 'ST'}
OUTPUT_ROOT   = Path(__file__).parent / 'projects'

# ── Delay profiles ────────────────────────────────────────────────────────────
DELAY_PROFILES = {
    'mild':     {'prob': 0.10, 'min_days': 2,  'max_days': 10, 'cascade': 0.30},
    'moderate': {'prob': 0.25, 'min_days': 5,  'max_days': 30, 'cascade': 0.67},
    'severe':   {'prob': 0.40, 'min_days': 10, 'max_days': 60, 'cascade': 0.80},
}


def simulate_delays_profile(steps, seqs, profile_name, rng):
    """Delay simulation with configurable profile and seeded rng."""
    p = DELAY_PROFILES[profile_name]
    steps_dict = {s['id']: s for s in steps}
    seq_map    = defaultdict(list)
    for s in seqs:
        seq_map[s['from']].append(s['to'])

    delays = {}
    critical = [s['id'] for s in steps if s.get('discipline', '') in CRITICAL_DISC]
    for sid in critical:
        if rng.random() < p['prob']:
            delays[sid] = rng.randint(p['min_days'], p['max_days'])

    for sid in sorted(delays, key=lambda x: steps_dict[x].get('order', 0)):
        for ch in seq_map[sid]:
            if ch not in delays:
                raw = int(delays[sid] * p['cascade'])
                if raw > 0:
                    delays[ch] = raw
    return delays


def generate_one_project(proj_idx, seed, delay_profile, rule_change_month, n_workers,
                          family_steps, meram):
    """Generate dataset + events for one project instance."""
    proj_tag  = f'P{proj_idx:03d}'
    rc_date   = PROJECT_START + timedelta(days=rule_change_month * 30)
    rng_data  = random.Random(seed)
    rng_evts  = random.Random(seed + 1000)
    rng_delay = random.Random(seed + 2000)

    # ── Dataset generation ────────────────────────────────────────────────────
    random.seed(seed)                        # ged uses module-level random
    dataset = ged.generate_epc_dataset(family_steps, meram)

    # Override rule-change date
    dataset['update_events'][0]['valid_from'] = rc_date.isoformat()

    # Override worker count
    all_certs = {c['cert'] for w in dataset['workers'] for c in w['certifications']}
    dataset['workers'] = ged.generate_workers(all_certs, n=n_workers)

    # Tag everything
    for key in ('activities', 'steps', 'families', 'work_permits',
                 'certifications', 'workers'):
        for obj in dataset[key]:
            obj['project_id'] = proj_tag
    dataset['project']['id']         = f'PROJ-{proj_tag}'
    dataset['project']['project_id'] = proj_tag
    dataset['metadata'].update({
        'project_id':        proj_tag,
        'seed':              seed,
        'delay_profile':     delay_profile,
        'rule_change_month': rule_change_month,
        'n_workers':         n_workers,
    })

    # ── Events generation (monkey-patch RULE_CHANGE then restore) ─────────────
    original_rc    = sev.RULE_CHANGE
    sev.RULE_CHANGE = rc_date
    random.seed(seed + 1000)               # sev uses module-level random

    worker_index = sev.build_worker_cert_index(dataset['workers'])
    assignments  = sev.assign_workers(dataset['steps'], dataset['workers'], worker_index)
    delays       = simulate_delays_profile(
        dataset['steps'], dataset['step_sequences'], delay_profile, rng_delay)
    events       = sev.generate_events(
        dataset['steps'], dataset['workers'], assignments, delays, worker_index)

    sev.RULE_CHANGE = original_rc          # restore

    for ev_list in events.values():
        for e in ev_list:
            e['project_id'] = proj_tag

    return dataset, events


def run(n_projects=5):
    print('Loading shared base data (Meram + Family Steps)...')
    family_steps, meram = ged.load_data()
    OUTPUT_ROOT.mkdir(exist_ok=True)

    profiles      = ['mild', 'moderate', 'severe']
    rc_months     = [4, 6, 8]
    worker_counts = [30, 50, 70]

    index = []
    for i in range(n_projects):
        seed      = 100 + i * 13          # spread seeds deterministically
        profile   = profiles[i % 3]
        rc_month  = rc_months[(i // 3) % 3]
        n_workers = worker_counts[(i // 9) % 3]

        print(f'\n  [{i+1:3d}/{n_projects}] seed={seed}  profile={profile:8s}  '
              f'RC_month={rc_month}  workers={n_workers}', end='  ')

        proj_dir = OUTPUT_ROOT / f'proj_{i:03d}'
        proj_dir.mkdir(exist_ok=True)

        try:
            dataset, events = generate_one_project(
                i, seed, profile, rc_month, n_workers, family_steps, meram)

            with open(proj_dir / 'dataset.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=1, default=str, ensure_ascii=False,
                          allow_nan=False)
            with open(proj_dir / 'events.json', 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=1, default=str, ensure_ascii=False)

            n_viol = len(events['permit_denied'])
            n_del  = sum(1 for e in events['completed'] if e.get('delay_days', 0) > 0)
            index.append({
                'project_id':         dataset['project']['id'],
                'seed':               seed,
                'delay_profile':      profile,
                'rule_change_month':  rc_month,
                'n_workers':          n_workers,
                'n_activities':       dataset['metadata']['total_activities'],
                'n_steps':            dataset['metadata']['total_steps'],
                'n_violations':       n_viol,
                'n_delayed_steps':    n_del,
                'path':               f'proj_{i:03d}',
            })
            print(f'OK  violations={n_viol}  delayed={n_del}')
        except Exception as exc:
            import traceback
            print(f'FAILED: {exc}')
            traceback.print_exc()

    with open(OUTPUT_ROOT / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Generated {len(index)}/{n_projects} projects  ->  {OUTPUT_ROOT}')
    if index:
        import statistics as st
        print(f'Violations:    min={min(p["n_violations"] for p in index)}  '
              f'max={max(p["n_violations"] for p in index)}  '
              f'mean={st.mean(p["n_violations"] for p in index):.0f}')
        print(f'Delayed steps: min={min(p["n_delayed_steps"] for p in index)}  '
              f'max={max(p["n_delayed_steps"] for p in index)}  '
              f'mean={st.mean(p["n_delayed_steps"] for p in index):.0f}')
    return index


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n-projects', type=int, default=5)
    args = ap.parse_args()
    run(n_projects=args.n_projects)
