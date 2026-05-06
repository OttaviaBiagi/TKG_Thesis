"""
Delay propagation analysis on the EPC TKG.

Analysis A — PRECEDES chain integrity and statistics
Analysis B — PRECEDES-aware accumulated delay (topological propagation)
Analysis C — Rule-change impact: pre/post violation counts + delay added
"""
import json
import sys
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone

DATA_DIR = Path('data/UseCase4')

# ── Load ────────────────────────────────────────────────────────────────────
ds = json.load(open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8'))
ev = json.load(open(DATA_DIR / 'epc_events.json',       encoding='utf-8'))

steps      = {s['id']: s for s in ds['steps']}
activities = {a['id']: a for a in ds['activities']}
seqs       = ds['step_sequences']                 # list of {from, to, activity_id}
completed  = {e['step_id']: e for e in ev['completed']}   # step_id → completion event
violations = ev['permit_denied']

RULE_CHANGE = '2024-06-29'

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS A — PRECEDES chain integrity and chain-length distribution
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 65)
print('ANALYSIS A — PRECEDES chain integrity & statistics')
print('=' * 65)

# Build adjacency lists (per activity)
children  = defaultdict(list)   # step_id -> [child step_id, ...]
parents   = defaultdict(list)   # step_id -> [parent step_id, ...]
for s in seqs:
    children[s['from']].append(s['to'])
    parents[s['to']].append(s['from'])

# Check: every PRECEDES edge stays within the same activity
cross_act = [s for s in seqs
             if steps[s['from']]['activity_id'] != steps[s['to']]['activity_id']]
print(f'Cross-activity PRECEDES edges (must be 0): {len(cross_act)}')

# Check: no self-loops
self_loops = [s for s in seqs if s['from'] == s['to']]
print(f'Self-loops (must be 0):                    {len(self_loops)}')

# Check: no backward edges (to.order < from.order within same activity)
backward = [s for s in seqs
            if steps[s['to']]['order'] < steps[s['from']]['order']]
print(f'Backward edges (must be 0):                {len(backward)}')

# Cycle detection via DFS (whole graph)
visited, rec_stack = set(), set()
has_cycle = False
def dfs(node):
    global has_cycle
    visited.add(node)
    rec_stack.add(node)
    for ch in children[node]:
        if ch not in visited:
            dfs(ch)
        elif ch in rec_stack:
            has_cycle = True
    rec_stack.discard(node)

sys.setrecursionlimit(100_000)
for sid in steps:
    if sid not in visited:
        dfs(sid)
print(f'Cycles detected (must be False):           {has_cycle}')

# Chain-length distribution (per activity: longest path = steps in activity)
steps_per_act = defaultdict(list)
for sid, s in steps.items():
    steps_per_act[s['activity_id']].append(s)

chain_lengths = [len(v) for v in steps_per_act.values()]
chain_lengths.sort()
import statistics as st
print(f'\nSteps per activity:')
print(f'  Activities:  {len(chain_lengths)}')
print(f'  Min:         {min(chain_lengths)}')
print(f'  Max:         {max(chain_lengths)}')
print(f'  Mean:        {st.mean(chain_lengths):.1f}')
print(f'  Median:      {st.median(chain_lengths):.0f}')

# Distribution table
from collections import Counter
dist = Counter(chain_lengths)
print(f'\n  Chain length  Count')
print(f'  ------------  -----')
for length in sorted(dist):
    print(f'  {length:12d}  {dist[length]:5d}')

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS B — PRECEDES-aware accumulated delay via topological propagation
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('ANALYSIS B — Accumulated delay via topological propagation')
print('=' * 65)

def parse_dt(s):
    if not s: return None
    return datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc)

# Topological sort (Kahn's algorithm)
in_degree = {sid: len(parents[sid]) for sid in steps}
queue = deque([sid for sid in steps if in_degree[sid] == 0])
topo_order = []
while queue:
    node = queue.popleft()
    topo_order.append(node)
    for ch in children[node]:
        in_degree[ch] -= 1
        if in_degree[ch] == 0:
            queue.append(ch)

print(f'Topological sort: {len(topo_order)}/{len(steps)} steps ordered '
      f'(cycle-free: {len(topo_order)==len(steps)})')

# For each step, compute:
#   planned_start  = step valid_from
#   own_delay      = completed[sid]['delay_days']
#   propagated_delay = max propagated_delay of all parent steps (delay cascades)
#   accumulated_delay = propagated_delay + own_delay

own_delay      = {}   # independent delay from simulation
prop_delay     = {}   # inherited from predecessors

for sid in steps:
    comp = completed.get(sid)
    own_delay[sid] = int(comp['delay_days']) if comp else 0

for sid in topo_order:
    parent_prop = max((prop_delay.get(p, 0) + own_delay.get(p, 0)
                       for p in parents[sid]), default=0)
    prop_delay[sid] = parent_prop

accumulated_delay = {sid: prop_delay[sid] + own_delay[sid] for sid in steps}

# Summary across all steps
all_acc = list(accumulated_delay.values())
delayed_steps = [v for v in all_acc if v > 0]

print(f'\nAll {len(steps)} steps:')
print(f'  Steps with accumulated delay > 0 days: {len(delayed_steps):,}')
print(f'  Max accumulated delay:                  {max(all_acc)} days')
print(f'  Mean accumulated delay (all steps):     {st.mean(all_acc):.1f} days')
print(f'  Mean accumulated delay (delayed only):  '
      f'{st.mean(delayed_steps):.1f} days' if delayed_steps else 'N/A')

# Distribution of accumulated delay
print(f'\n  Delay (days)  Steps')
print(f'  ------------  -----')
buckets = [(0,0), (1,3), (4,7), (8,14), (15,30), (31,999)]
for lo, hi in buckets:
    count = sum(1 for v in all_acc if lo <= v <= hi)
    label = f'{lo}-{hi}' if hi < 999 else f'{lo}+'
    bar = '█' * (count // 200)
    print(f'  {label:>12}  {count:5d}  {bar}')

# Top 10 most-delayed activities (by max step accumulated delay)
act_max_delay = defaultdict(int)
for sid, d in accumulated_delay.items():
    aid = steps[sid]['activity_id']
    act_max_delay[aid] = max(act_max_delay[aid], d)

top10 = sorted(act_max_delay.items(), key=lambda x: -x[1])[:10]
print(f'\nTop 10 activities by max accumulated step delay:')
print(f'  {"Activity":30s}  {"Disc":4s}  {"Steps":5s}  {"Max delay":10s}')
print(f'  {"-"*30}  {"-"*4}  {"-"*5}  {"-"*10}')
for aid, d in top10:
    act  = activities[aid]
    n_st = len(steps_per_act[aid])
    print(f'  {aid:30s}  {act["discipline"]:4s}  {n_st:5d}  {d:10d} days')

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS C — Rule-change impact
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 65)
print('ANALYSIS C — Rule-change impact (hot_work, month ≥ 6)')
print('=' * 65)

pre_rc  = [v for v in violations if not v.get('after_rule_change')]
post_rc = [v for v in violations if v.get('after_rule_change')]

print(f'Total violations:           {len(violations)}')
print(f'  Pre-rule-change  (human error, 5%): {len(pre_rc)}')
print(f'  Post-rule-change (new cert req):    {len(post_rc)}')

# Permit type breakdown for post-RC
from collections import Counter
permit_counts = Counter(v['permit_type'] for v in post_rc)
print(f'\nPost-rule-change violations by permit type:')
for pt, cnt in permit_counts.most_common():
    print(f'  {pt:<25} {cnt:4d}')

# Missing cert breakdown
mc_counts = Counter(c for v in post_rc for c in v.get('missing_certs', []))
print(f'\nMissing cert breakdown (post-rule-change):')
for cert, cnt in mc_counts.most_common():
    print(f'  {cert:<35} {cnt:4d}')

# Workers affected by rule change
affected_workers = Counter(v['worker_id'] for v in post_rc)
print(f'\nWorkers with post-rule-change violations: {len(affected_workers)}')
for wid, cnt in affected_workers.most_common(10):
    print(f'  {wid}  {cnt:4d} violations')

# Timeline: violations per month
def to_month(date_str):
    if not date_str: return -1
    d = datetime.fromisoformat(date_str[:10])
    return max(0, (d.year - 2024) * 12 + d.month - 1)

# Separate: truly caused by rule change vs. coincidental
#   "caused" = hot_work permit + Advanced_Fire_Watch in missing_certs
rc_caused = [v for v in post_rc
             if v['permit_type'] == 'hot_work'
             and 'Advanced_Fire_Watch' in v.get('missing_certs', [])]
rc_coincident = [v for v in post_rc if v not in rc_caused]

print(f'\nPost-rule-change breakdown:')
print(f'  Directly caused by new hot_work cert rule: {len(rc_caused):3d}')
print(f'  Coincident (pre-existing cert gaps, post date): {len(rc_coincident):3d}')

monthly = defaultdict(lambda: {'pre': 0, 'caused': 0, 'coincident': 0})
for v in violations:
    m = to_month(v.get('date', ''))
    if not v.get('after_rule_change'):
        monthly[m]['pre'] += 1
    elif v in rc_caused:
        monthly[m]['caused'] += 1
    else:
        monthly[m]['coincident'] += 1

print(f'\nViolations per month (τ=0 = Jan 2024):')
print(f'  {"τ":>4}  {"Date":>7}  {"Pre-RC":>6}  {"RC-caused":>9}  {"Coincident":>10}  {"Total":>5}')
print(f'  {"--":>4}  {"-------":>7}  {"------":>6}  {"---------":>9}  {"----------":>10}  {"-----":>5}')
from datetime import date
for m in range(18):
    pre  = monthly[m]['pre']
    caus = monthly[m]['caused']
    coin = monthly[m]['coincident']
    yr   = 2024 + (m // 12)
    mo   = (m % 12) + 1
    label = date(yr, mo, 1).strftime('%b %Y')
    marker = ' ← new Advanced Fire Watch rule' if m == 5 else ''
    if pre + caus + coin > 0:
        print(f'  {m:>4}  {label:>7}  {pre:>6}  {caus:>9}  {coin:>10}  {pre+caus+coin:>5}{marker}')

# Estimated delay added by rule-change violations
# Assumption: a PERMIT_DENIED event means assignment must be retried;
# we model the retry delay as the gap between the violation date and the
# NEXT assigned_to event on the same step (if any), otherwise use a fixed 3-day estimate.
assigned_by_step = defaultdict(list)
for e in ev['assigned_to']:
    assigned_by_step[e['step_id']].append(e['date'])
for sid in assigned_by_step:
    assigned_by_step[sid].sort()

retry_delays = []
for v in post_rc:
    sid   = v['step_id']
    vdate = v.get('date', '')
    later = [d for d in assigned_by_step[sid] if d > vdate]
    if later:
        from datetime import timedelta
        d0 = datetime.fromisoformat(vdate[:10])
        d1 = datetime.fromisoformat(later[0][:10])
        gap = (d1 - d0).days
        if 0 < gap < 60:   # sanity cap
            retry_delays.append(gap)

if retry_delays:
    print(f'\nEstimated retry delay from rule-change violations:')
    print(f'  Events with measurable retry gap: {len(retry_delays)}')
    print(f'  Mean retry delay:  {st.mean(retry_delays):.1f} days')
    print(f'  Median:            {st.median(retry_delays):.0f} days')
    print(f'  Max:               {max(retry_delays)} days')
else:
    print(f'\nNote: each post-rule-change PERMIT_DENIED represents a '
          f'~3-day reassignment overhead (single worker per step in simulation).')

print()
print('Analysis complete.')
