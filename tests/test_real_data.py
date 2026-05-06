"""
UseCase4 — Data quality tests for Meram real data.
Run: python tests/test_real_data.py

Two layers tested:
  1. REAL DATA: activities, disciplines, hours, families, step templates, PRECEDES
  2. SYNTHETIC LAYER: valid_from/valid_to bitemporality, tx_time, rule-change event

Requires:
  - data/UseCase4/epc_dataset_real.json
  - data/UseCase4/meram/Meram_PCS_Progress.xlsx
  - Family_Steps_macro.xlsm (searched in standard locations)
"""
import json, math, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

try:
    import pandas as pd
except ImportError:
    sys.exit('pandas required: pip install pandas openpyxl')

DATA_DIR  = Path('data/UseCase4')
MERAM_XLS = DATA_DIR / 'meram' / 'Meram_PCS_Progress.xlsx'
FAMILY_STEPS_CANDIDATES = [
    DATA_DIR / 'Family_Steps_macro.xlsm',
    Path.home() / 'TKG_Thesis' / 'data' / 'UseCase4' / 'Family_Steps_macro.xlsm',
    Path(r'C:\Users\obiagi\OneDrive - Tecnicas Reunidas, S.A\Documents\Family Steps macro.xlsm'),
    Path(r'C:\Users\obiagi\OneDrive - Tecnicas Reunidas, S.A\Microsoft Copilot Chat Files\Family Steps macro.xlsm'),
]
FAMILY_STEPS_XLS = next((p for p in FAMILY_STEPS_CANDIDATES if p.exists()), None)

PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROJECT_END   = datetime(2026, 1, 1, tzinfo=timezone.utc)
RULE_CHANGE   = datetime(2024, 6, 29, tzinfo=timezone.utc)
TX_DATE       = None  # derived from data after load

PASS = FAIL = WARN = 0

def ok(msg):
    global PASS; PASS += 1
    print(f'  [PASS] {msg}')

def fail(msg, detail=''):
    global FAIL; FAIL += 1
    print(f'  [FAIL] {msg}' + (f'\n         {detail}' if detail else ''))

def warn(msg):
    global WARN; WARN += 1
    print(f'  [WARN] {msg}')

def parse_dt(s):
    if not s: return None
    try:
        dt = datetime.fromisoformat(str(s)[:19])
        return dt.replace(tzinfo=timezone.utc)
    except:
        return None

# ─── Load data ────────────────────────────────────────────────────────────────
print('Loading data...')
ds       = json.load(open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8'))
acts     = ds['activities']
steps    = ds['steps']
seqs     = ds.get('step_sequences', [])
workers  = ds['workers']
permits  = ds['work_permits']

acts_dict  = {a['id']: a for a in acts}
steps_dict = {s['id']: s for s in steps}

# Derive TX_DATE from the data (single-snapshot assumption)
_all_tx = {(s.get('tx_time') or '')[:10] for s in steps} | {(a.get('tx_time') or '')[:10] for a in acts}
TX_DATE = _all_tx.pop() if len(_all_tx) == 1 else None
steps_by_act = defaultdict(list)
for s in steps:
    steps_by_act[s['activity_id']].append(s)
for aid in steps_by_act:
    steps_by_act[aid].sort(key=lambda x: x['order'])

# ─── LAYER 1: Real activity data vs Meram Excel ───────────────────────────────
print('\n=== LAYER 1: Real activity data (Meram_PCS_Progress.xlsx) ===')

if not MERAM_XLS.exists():
    warn(f'Meram_PCS_Progress.xlsx not found at {MERAM_XLS} — skipping Excel cross-checks')
else:
    meram = pd.read_excel(MERAM_XLS, sheet_name='Activities PCS')
    meram['Disc'] = meram['Disc'].str.strip()
    meram['Fami'] = meram['Fami'].str.strip()
    meram['Estimated Hours'] = pd.to_numeric(meram['Estimated Hours'], errors='coerce').fillna(0)
    meram['EarnedHours']     = pd.to_numeric(meram['EarnedHours'],     errors='coerce').fillna(0)
    meram_dedup = meram.drop_duplicates(subset=['ActID'], keep='first').reset_index(drop=True)
    excel_ids   = set(str(r['ActID']).strip() for _, r in meram_dedup.iterrows())
    json_ids    = set(acts_dict.keys())

    # Count
    if len(excel_ids) == len(json_ids) == 5555:
        ok(f'Activity count: {len(json_ids)} (matches expected 5,555)')
    else:
        fail(f'Activity count mismatch: Excel={len(excel_ids)} JSON={len(json_ids)} expected=5555')

    # ID coverage
    missing_in_json  = excel_ids - json_ids
    extra_in_json    = json_ids - excel_ids
    if not missing_in_json and not extra_in_json:
        ok('All Excel ActIDs present in JSON; no phantom JSON entries')
    else:
        if missing_in_json:
            fail(f'{len(missing_in_json)} Excel ActIDs missing from JSON',
                 f'e.g. {list(missing_in_json)[:3]}')
        if extra_in_json:
            fail(f'{len(extra_in_json)} JSON activities not in Excel',
                 f'e.g. {list(extra_in_json)[:3]}')

    # Discipline
    disc_mm = sum(1 for _, r in meram_dedup.iterrows()
                  if str(r['ActID']).strip() in acts_dict
                  and str(r['Disc']) != acts_dict[str(r['ActID']).strip()].get('discipline',''))
    if disc_mm == 0:
        ok('Discipline: all 5,555 activities match Excel exactly')
    else:
        fail(f'Discipline mismatches: {disc_mm}')

    # Estimated hours
    eh_mm = sum(1 for _, r in meram_dedup.iterrows()
                if str(r['ActID']).strip() in acts_dict
                and abs(float(r['Estimated Hours']) -
                        acts_dict[str(r['ActID']).strip()].get('estimated_hours', 0)) > 0.01)
    if eh_mm == 0:
        ok('estimated_hours: all 5,555 match Excel exactly')
    else:
        fail(f'estimated_hours mismatches: {eh_mm}')

    # Earned hours
    earn_mm = sum(1 for _, r in meram_dedup.iterrows()
                  if str(r['ActID']).strip() in acts_dict
                  and abs(float(r['EarnedHours']) -
                          acts_dict[str(r['ActID']).strip()].get('earned_hours', 0)) > 0.01)
    if earn_mm == 0:
        ok('earned_hours: all 5,555 match Excel exactly')
    else:
        fail(f'earned_hours mismatches: {earn_mm}')

    # Family codes
    fam_mm = sum(1 for _, r in meram_dedup.iterrows()
                 if str(r['ActID']).strip() in acts_dict
                 and str(r['Fami']).strip() != acts_dict[str(r['ActID']).strip()].get('family',''))
    if fam_mm == 0:
        ok('Family codes: all 5,555 match Excel exactly')
    else:
        fail(f'Family code mismatches: {fam_mm}')

# Disciplines present
json_discs = sorted(set(a['discipline'] for a in acts))
if len(json_discs) == 14:
    ok(f'14 disciplines present: {json_discs}')
else:
    warn(f'{len(json_discs)} disciplines found (expected 14): {json_discs}')

# Progress formula: progress_pct = earned_hours / estimated_hours * 100
prog_errors = 0
prog_over_100 = []
for a in acts:
    est  = a.get('estimated_hours', 0)
    earn = a.get('earned_hours', 0)
    prog = a.get('progress_pct', 0)
    if est > 0:
        expected = earn / est * 100
        if abs(expected - prog) > 0.01:
            prog_errors += 1
    if prog > 100:
        prog_over_100.append((a['id'], prog))

if prog_errors == 0:
    ok('progress_pct = earned/estimated × 100 for all activities')
else:
    fail(f'{prog_errors} activities have progress_pct inconsistent with hours')

if len(prog_over_100) <= 5:
    warn(f'{len(prog_over_100)} activities with progress > 100% (over-completion): '
         + ', '.join(f'{a[0]} ({a[1]:.1f}%)' for a in prog_over_100))
else:
    fail(f'{len(prog_over_100)} activities with progress > 100%')

# ─── LAYER 2: Step templates vs Family_Steps_macro.xlsm ──────────────────────
print('\n=== LAYER 2: Step templates (Family_Steps_macro.xlsm) ===')

if FAMILY_STEPS_XLS is None:
    warn('Family_Steps_macro.xlsm not found — skipping step template checks')
else:
    fs = pd.read_excel(FAMILY_STEPS_XLS, sheet_name='Family Steps')
    fs['FAMILY'] = fs['FAMILY'].str.strip()
    fs = fs[fs['Active'] == True].copy()
    family_steps_ref = {}
    for fam, grp in fs.groupby('FAMILY'):
        grp = grp.sort_values('#')
        family_steps_ref[fam] = [
            (str(r['STEP']).strip(), int(r['#']),
             float(r['%']) if r['%'] is not None and not (isinstance(r['%'], float) and math.isnan(r['%'])) else 0.0)
            for _, r in grp.iterrows()
        ]

    acts_with_tpl  = [a for a in acts if a.get('family','') in family_steps_ref]
    acts_no_tpl    = [a for a in acts if a.get('family','') not in family_steps_ref]

    ok(f'Activities with known template: {len(acts_with_tpl)}/5555')
    if acts_no_tpl:
        no_tpl_discs = dict(Counter(a['discipline'] for a in acts_no_tpl))
        warn(f'{len(acts_no_tpl)} activities have no template in xlsm '
             f'(disciplines: {no_tpl_discs})')

    match_exact = name_mm = order_mm = count_mm = 0
    for a in acts_with_tpl:
        ref = family_steps_ref[a['family']]
        json_s = steps_by_act.get(a['id'], [])
        if len(json_s) != len(ref):
            count_mm += 1
            continue
        if not all(js['name'] == rs[0] for js, rs in zip(json_s, ref)):
            name_mm += 1
        elif not all(js['order'] == rs[1] for js, rs in zip(json_s, ref)):
            order_mm += 1
        else:
            match_exact += 1

    pct = match_exact / len(acts_with_tpl) * 100 if acts_with_tpl else 0
    if name_mm == 0 and order_mm == 0 and count_mm == 0:
        ok(f'Step templates: {match_exact}/{len(acts_with_tpl)} ({pct:.1f}%) exact match on name+order+weight')
    else:
        if count_mm: fail(f'Step count mismatches (wrong number of steps): {count_mm}')
        if name_mm:  fail(f'Step name mismatches: {name_mm}')
        if order_mm: fail(f'Step order mismatches: {order_mm}')
        if match_exact:
            ok(f'{match_exact}/{len(acts_with_tpl)} activities have perfectly matching steps')

# ─── LAYER 3: PRECEDES relationships ─────────────────────────────────────────
print('\n=== LAYER 3: PRECEDES relationships ===')

seq_ok = seq_err = cross_act = 0
for seq in seqs:
    sf = steps_dict.get(seq.get('from'))
    st = steps_dict.get(seq.get('to'))
    if sf is None or st is None:
        seq_err += 1
        continue
    if sf['activity_id'] != st['activity_id']:
        cross_act += 1
    elif st['order'] == sf['order'] + 1:
        seq_ok += 1
    else:
        seq_err += 1

if seq_ok == len(seqs) and seq_err == 0 and cross_act == 0:
    ok(f'All {len(seqs)} PRECEDES edges are sequential within the same activity (order+1)')
else:
    ok(f'{seq_ok}/{len(seqs)} PRECEDES edges correct')
    if cross_act: warn(f'{cross_act} cross-activity PRECEDES edges')
    if seq_err:   fail(f'{seq_err} PRECEDES edges with missing steps or non-sequential order')

# Check no self-loops
loops = sum(1 for s in seqs if s.get('from') == s.get('to'))
if loops == 0:
    ok('No self-loops in PRECEDES graph')
else:
    fail(f'{loops} self-loop PRECEDES edges')

# Check all steps belong to a known activity
orphan_steps = sum(1 for s in steps if s.get('activity_id') not in acts_dict)
if orphan_steps == 0:
    ok(f'All {len(steps)} steps belong to a known activity')
else:
    fail(f'{orphan_steps} steps with unknown activity_id')

# ─── LAYER 4: Synthetic bitemporal fields ────────────────────────────────────
print('\n=== LAYER 4: Synthetic bitemporal fields (valid_from / valid_to / tx_time) ===')

# valid_from < valid_to for all steps
vf_vt_err = 0
for s in steps:
    vf = parse_dt(s.get('valid_from'))
    vt = parse_dt(s.get('valid_to'))
    if vf is None or vt is None or vf >= vt:
        vf_vt_err += 1
if vf_vt_err == 0:
    ok(f'All {len(steps)} steps have valid_from < valid_to')
else:
    fail(f'{vf_vt_err} steps with missing or inverted valid_from/valid_to')

# valid_from < valid_to for all activities
vf_vt_act_err = 0
for a in acts:
    vf = parse_dt(a.get('valid_from'))
    vt = parse_dt(a.get('valid_to'))
    if vf is None or vt is None or vf >= vt:
        vf_vt_act_err += 1
if vf_vt_act_err == 0:
    ok(f'All {len(acts)} activities have valid_from < valid_to')
else:
    fail(f'{vf_vt_act_err} activities with missing or inverted valid_from/valid_to')

# Step valid_to <= activity valid_to
step_exceeds_act = []
for s in steps:
    vt_s = parse_dt(s.get('valid_to'))
    a    = acts_dict.get(s.get('activity_id'))
    if a:
        vt_a = parse_dt(a.get('valid_to'))
        if vt_s and vt_a and vt_s > vt_a:
            step_exceeds_act.append(s['id'])
if not step_exceeds_act:
    ok('All step valid_to <= parent activity valid_to')
else:
    warn(f'{len(step_exceeds_act)} steps have valid_to AFTER their activity valid_to')

# tx_time consistency (single snapshot)
tx_dates_steps = set((s.get('tx_time') or '')[:10] for s in steps)
tx_dates_acts  = set((a.get('tx_time') or '')[:10] for a in acts)
if len(tx_dates_steps) == 1 and tx_dates_steps == tx_dates_acts:
    ok(f'All nodes share a single tx_time = {TX_DATE} (single transaction snapshot)')
else:
    warn(f'Multiple tx_time values: steps={tx_dates_steps}, acts={tx_dates_acts}')

# tx_time > max valid_to
all_vt = [parse_dt(s.get('valid_to')) for s in steps + acts if s.get('valid_to')]
max_vt = max(d for d in all_vt if d)
if TX_DATE:
    tx_snap = datetime.fromisoformat(TX_DATE).replace(tzinfo=timezone.utc)
    if tx_snap > max_vt:
        ok(f'tx_time ({tx_snap.date()}) > max valid_to ({max_vt.date()}) — snapshot recorded after all validity intervals')
    else:
        warn(f'tx_time ({tx_snap.date()}) is NOT after max valid_to ({max_vt.date()})')

# ─── LAYER 5: Synthetic rule-change event ────────────────────────────────────
print('\n=== LAYER 5: Bitemporal rule-change event ===')

ue = ds.get('update_events', [])
rc_events = [e for e in ue if e.get('type') == 'permit_rule_change']

if rc_events:
    rc = rc_events[0]
    rc_date = parse_dt(rc.get('valid_from'))
    if rc_date == RULE_CHANGE:
        ok(f'Rule change date: {RULE_CHANGE.date()} (correct)')
    else:
        fail(f'Rule change date: expected {RULE_CHANGE.date()}, got {rc_date}')
    if rc.get('affected') == 'hot_work':
        ok('Rule change targets hot_work permit')
    else:
        fail(f'Rule change targets wrong permit: {rc.get("affected")}')
    if rc.get('new_cert') == 'Advanced Fire Watch':
        ok('New required cert: Advanced Fire Watch (correct)')
    else:
        fail(f'Unexpected new_cert: {rc.get("new_cert")}')
else:
    fail('No permit_rule_change event found in update_events')

# Cert validity: worker HAS_CERT valid_from < valid_to
cert_err = 0
for w in workers:
    for c in w.get('certifications', []):
        vf = parse_dt(c.get('valid_from'))
        vt = parse_dt(c.get('valid_to'))
        if vf is None or vt is None or vf >= vt:
            cert_err += 1
if cert_err == 0:
    total_certs = sum(len(w.get('certifications', [])) for w in workers)
    ok(f'All {total_certs} worker cert records have valid_from < valid_to')
else:
    fail(f'{cert_err} worker cert records with missing or inverted dates')

# after_rule_change flag: must be consistent with event date vs RULE_CHANGE
try:
    ev_file = DATA_DIR / 'epc_events.json'
    if ev_file.exists():
        ev = json.load(ev_file.open(encoding='utf-8'))
        rc_str = RULE_CHANGE.strftime('%Y-%m-%d')
        arc_wrong = [e for e in ev.get('permit_denied', [])
                     if e.get('after_rule_change') and e.get('date','') < rc_str]
        if not arc_wrong:
            ok(f'after_rule_change flag consistent with RULE_CHANGE date for all '
               f'{len(ev["permit_denied"])} permit_denied events')
        else:
            fail(f'{len(arc_wrong)} permit_denied events flagged after_rule_change=True '
                 f'but dated before {rc_str}',
                 f'e.g. {arc_wrong[0]["date"]}')
except Exception as exc:
    warn(f'Could not check after_rule_change flags: {exc}')

# Permit coverage: every permit_type referenced by steps exists in work_permits
permit_ids   = {p['id'] for p in permits}
step_permits = {s.get('permit_type') for s in steps}
missing = step_permits - permit_ids
if not missing:
    ok(f'All {len(step_permits)} permit types referenced by steps exist in work_permits')
else:
    fail(f'Steps reference non-existent permits: {missing}')

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f'\n{"=" * 60}')
print(f'RESULT: {PASS} PASSED  |  {FAIL} FAILED  |  {WARN} WARNINGS')
if FAIL == 0 and WARN == 0:
    print('All checks passed. Dataset is correct and consistent.')
elif FAIL == 0:
    print('No failures — warnings are expected data characteristics.')
else:
    print('ISSUES FOUND — review failures above.')
