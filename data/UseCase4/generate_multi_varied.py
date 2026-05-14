"""
Varied multi-project dataset generator — structural diversity test.

Unlike generate_multi_project.py (which uses ALL 5,555 Meram activities),
this generator samples a random 50–80% of Meram family codes per project,
so each project has a genuinely different activity/step subset and discipline mix.

Dimension of variation per project:
  - family_frac  : 50–80% of all Meram family codes (cycles through 7 levels)
  - delay_profile: mild / moderate / severe
  - rule_change  : month 4 / 6 / 8
  - n_workers    : 30 / 50 / 70
  - seed         : 300 + i*17  (spread deterministically)

Output directory: data/UseCase4/projects_varied/
  proj_NNN/dataset.json
  proj_NNN/events.json
  index.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from datetime import timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent))

import generate_epc_dataset as ged
import simulate_events as sev

PROJECT_START = ged.PROJECT_START
CRITICAL_DISC = {'ME', 'PI', 'EL', 'ST'}
OUTPUT_ROOT   = Path(__file__).parent / 'projects_varied'

DELAY_PROFILES = {
    'mild':     {'prob': 0.10, 'min_days': 2,  'max_days': 10,  'cascade': 0.30},
    'moderate': {'prob': 0.25, 'min_days': 5,  'max_days': 30,  'cascade': 0.67},
    'severe':   {'prob': 0.40, 'min_days': 10, 'max_days': 60,  'cascade': 0.80},
}


def simulate_delays_profile(steps, seqs, profile_name, rng):
    p          = DELAY_PROFILES[profile_name]
    steps_dict = {s['id']: s for s in steps}
    seq_map    = defaultdict(list)
    for s in seqs:
        seq_map[s['from']].append(s['to'])

    delays   = {}
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


def generate_one_varied_project(proj_idx, seed, delay_profile, rule_change_month,
                                  n_workers, family_steps, meram, family_frac):
    """
    Generate dataset + events for one project using a random family subset.

    family_frac : fraction of Meram family codes to include (0.50–0.80)
    """
    proj_tag  = f'V{proj_idx:03d}'
    rc_date   = PROJECT_START + timedelta(days=rule_change_month * 30)
    rng_data  = random.Random(seed)
    rng_delay = random.Random(seed + 2000)

    # Sample a structurally diverse subset of families
    all_families = sorted(meram['Fami'].dropna().unique().tolist())
    n_sample     = max(10, int(len(all_families) * family_frac))
    sampled_fams = set(rng_data.sample(all_families, n_sample))

    # Activities whose family is in the sample, plus unmatched (Fami==NaN or no template)
    meram_sub = meram[
        meram['Fami'].isin(sampled_fams) | meram['Fami'].isna()
    ].copy().reset_index(drop=True)

    # Generate dataset from subset (generate_epc_dataset accepts any meram DataFrame)
    random.seed(seed)
    dataset = ged.generate_epc_dataset(family_steps, meram_sub)

    # Override rule-change date
    dataset['update_events'][0]['valid_from'] = rc_date.isoformat()

    # Override worker count with fresh synthetic workers
    all_certs = {c['cert'] for w in dataset['workers'] for c in w['certifications']}
    dataset['workers'] = ged.generate_workers(all_certs, n=n_workers)

    # Tag every entity with this project's ID
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
        'family_frac':       family_frac,
        'n_sampled_families': len(sampled_fams),
    })

    # Event simulation
    original_rc    = sev.RULE_CHANGE
    sev.RULE_CHANGE = rc_date
    random.seed(seed + 1000)

    worker_index = sev.build_worker_cert_index(dataset['workers'])
    assignments  = sev.assign_workers(dataset['steps'], dataset['workers'], worker_index)
    delays       = simulate_delays_profile(
        dataset['steps'], dataset['step_sequences'], delay_profile, rng_delay)
    events       = sev.generate_events(
        dataset['steps'], dataset['workers'], assignments, delays, worker_index)

    sev.RULE_CHANGE = original_rc

    for ev_list in events.values():
        for e in ev_list:
            e['project_id'] = proj_tag

    return dataset, events, len(sampled_fams), len(meram_sub)


def run(n_projects: int = 30):
    print('Loading shared base data (Meram + Family Steps)...')
    family_steps, meram = ged.load_data()
    OUTPUT_ROOT.mkdir(exist_ok=True)

    all_families = sorted(meram['Fami'].dropna().unique().tolist())
    print(f'Total unique Meram families: {len(all_families)}  '
          f'(projects will sample 50–80% = {int(len(all_families)*0.50)}–{int(len(all_families)*0.80)})')

    profiles      = ['mild', 'moderate', 'severe']
    rc_months     = [4, 6, 8]
    worker_counts = [30, 50, 70]
    frac_options  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]  # 7 levels

    index = []
    for i in range(n_projects):
        seed      = 300 + i * 17
        profile   = profiles[i % 3]
        rc_month  = rc_months[(i // 3) % 3]
        n_workers = worker_counts[(i // 9) % 3]
        frac      = frac_options[i % len(frac_options)]

        print(f'\n  [{i+1:3d}/{n_projects}] seed={seed}  {profile:8s}  '
              f'RC={rc_month}mo  workers={n_workers}  fam_frac={frac:.0%}', end='  ')

        proj_dir = OUTPUT_ROOT / f'proj_{i:03d}'
        proj_dir.mkdir(exist_ok=True)

        try:
            dataset, events, n_fams, n_acts = generate_one_varied_project(
                i, seed, profile, rc_month, n_workers,
                family_steps, meram, frac)

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
                'family_frac':        frac,
                'n_families':         n_fams,
                'n_activities':       dataset['metadata']['total_activities'],
                'n_steps':            dataset['metadata']['total_steps'],
                'n_violations':       n_viol,
                'n_delayed_steps':    n_del,
                'path':               f'proj_{i:03d}',
            })
            print(f'OK  fams={n_fams:3d}  acts={n_acts:5d}  '
                  f'steps={dataset["metadata"]["total_steps"]:6d}  viol={n_viol}')
        except Exception as exc:
            import traceback
            print(f'FAILED: {exc}')
            traceback.print_exc()

    with open(OUTPUT_ROOT / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)

    print(f'\n{"="*70}')
    print(f'Generated {len(index)}/{n_projects} varied projects  ->  {OUTPUT_ROOT}')
    if index:
        import statistics as st
        print(f'Steps:      min={min(p["n_steps"] for p in index):6d}  '
              f'max={max(p["n_steps"] for p in index):6d}  '
              f'mean={st.mean(p["n_steps"] for p in index):6.0f}')
        print(f'Families:   min={min(p["n_families"] for p in index):3d}  '
              f'max={max(p["n_families"] for p in index):3d}  '
              f'mean={st.mean(p["n_families"] for p in index):.0f}')
        print(f'Violations: min={min(p["n_violations"] for p in index):4d}  '
              f'max={max(p["n_violations"] for p in index):4d}  '
              f'mean={st.mean(p["n_violations"] for p in index):.0f}')
    return index


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n-projects', type=int, default=30)
    args = ap.parse_args()
    run(n_projects=args.n_projects)
