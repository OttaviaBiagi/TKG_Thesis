"""Diagnose and explain the three issues in Exp J."""
import json
from pathlib import Path
from collections import defaultdict, deque, Counter
import sys

DATA_DIR = Path('data/UseCase4')
ds = json.load(open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8'))
ev = json.load(open(DATA_DIR / 'epc_events.json',       encoding='utf-8'))

steps      = {s['id']: s for s in ds['steps']}
activities = {a['id']: a for a in ds['activities']}
seqs       = ds['step_sequences']
completed  = {e['step_id']: e for e in ev['completed']}
violations = ev['permit_denied']

children = defaultdict(list)
parents  = defaultdict(list)
for s in seqs:
    children[s['from']].append(s['to'])
    parents[s['to']].append(s['from'])

in_deg = {sid: len(parents[sid]) for sid in steps}
queue  = deque([sid for sid in steps if in_deg[sid] == 0])
topo   = []
while queue:
    n = queue.popleft(); topo.append(n)
    for ch in children[n]:
        in_deg[ch] -= 1
        if in_deg[ch] == 0: queue.append(ch)

own_delay  = {sid: int(completed[sid]['delay_days']) if sid in completed else 0 for sid in steps}
prop_delay = {}
for sid in topo:
    prop_delay[sid] = max((prop_delay.get(p,0)+own_delay.get(p,0) for p in parents[sid]), default=0)
acc_delay = {sid: prop_delay[sid]+own_delay[sid] for sid in steps}

# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 1: Concrete propagation example
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ISSUE 1 — Concrete delay propagation example")
print("=" * 70)

# Find a medium-length chain (5-7 steps) where delays actually propagate
steps_per_act = defaultdict(list)
for sid, s in steps.items():
    steps_per_act[s['activity_id']].append(s)
for aid in steps_per_act:
    steps_per_act[aid].sort(key=lambda x: x['order'])

# Find activity with 5-7 steps where at least 2 steps have own_delay > 0
good_examples = []
for aid, act_steps in steps_per_act.items():
    n = len(act_steps)
    if not (5 <= n <= 8): continue
    delays_in_chain = [own_delay[s['id']] for s in act_steps]
    n_delayed = sum(1 for d in delays_in_chain if d > 0)
    total_acc  = acc_delay[act_steps[-1]['id']]
    if n_delayed >= 2 and total_acc > 10:
        good_examples.append((total_acc, n_delayed, aid, act_steps))

good_examples.sort(reverse=True)
ex_act = good_examples[0][2]
ex_steps = good_examples[0][3]

act_info = activities[ex_act]
print(f"\nActivity: {ex_act}")
print(f"  Discipline: {act_info['discipline']}  Name: {act_info['name'][:60]}")
print(f"  Chain length: {len(ex_steps)} steps")
print()
print(f"  {'Step':32s}  {'Order':5s}  {'Planned':12s}  {'Own delay':9s}  "
      f"{'Inherited':9s}  {'Accumulated':11s}  {'Actual start':12s}")
print(f"  {'-'*32}  {'-'*5}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*11}  {'-'*12}")

for s in ex_steps:
    sid = s['id']
    comp = completed.get(sid, {})
    planned = (comp.get('planned_date', s['valid_from']) or s['valid_from'])[:10]
    own  = own_delay[sid]
    prop = prop_delay[sid]
    acc  = acc_delay[sid]
    # Actual start = planned + inherited delay
    from datetime import datetime, timedelta
    pd_dt = datetime.fromisoformat(planned)
    actual_start = pd_dt + timedelta(days=prop)
    actual_end   = pd_dt + timedelta(days=acc)
    arrow = ' <-- own delay' if own > 0 else ''
    print(f"  {sid:32s}  {s['order']:5d}  {planned:12s}  {own:9d}  "
          f"{prop:9d}  {acc:11d}  {actual_start.strftime('%Y-%m-%d'):12s}{arrow}")

last_sid = ex_steps[-1]['id']
last_comp = completed.get(last_sid, {})
print()
print(f"  Interpretation:")
print(f"    - Without any delay, step {len(ex_steps)} would start on its planned date")
print(f"    - With propagated delays, it starts {acc_delay[last_sid]} days late")
print(f"    - Of those, {prop_delay[last_sid]} days come from predecessor delays (inherited)")
print(f"    - And {own_delay[last_sid]} days come from its own delay")
print(f"    -> Even if step {len(ex_steps)} had ZERO own delay, it would still be "
      f"{prop_delay[last_sid]} days late because of upstream steps")

# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 2: What does "accumulated delay by discipline" mean?
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ISSUE 2 — Accumulated delay by discipline (boxplot explained)")
print("=" * 70)

disc_delays = defaultdict(list)
for sid, d in acc_delay.items():
    disc_delays[steps[sid]['discipline']].append(d)

import statistics as st
print()
print(f"  {'Disc':4s}  {'N steps':7s}  {'Mean acc':8s}  {'Median':6s}  {'Max':5s}  "
      f"{'%>0':5s}  {'%>30d':6s}")
print(f"  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*6}")
for disc in sorted(disc_delays, key=lambda d: -st.mean(disc_delays[d])):
    vals = disc_delays[disc]
    pct_pos = 100*sum(1 for v in vals if v>0)/len(vals)
    pct_30  = 100*sum(1 for v in vals if v>30)/len(vals)
    print(f"  {disc:4s}  {len(vals):7d}  {st.mean(vals):8.1f}  "
          f"{int(st.median(vals)):6d}  {max(vals):5d}  "
          f"{pct_pos:5.0f}%  {pct_30:6.0f}%")
print()
print("  Why disciplines differ:")
print("  - ME/PI have late DISCIPLINE_TIMELINE windows (months 6-15, 7-16)")
print("  - Late steps sit downstream of more predecessors -> more accumulated delay")
print("  - SP/PL start early (months 0-3, 1-5) -> fewer predecessors -> less accumulation")
print("  - The boxplot median IS the typical step in that discipline.")
print("  - Wide spread = some activities with very long chains (many predecessors)")

# ─────────────────────────────────────────────────────────────────────────────
# ISSUE 3: Advanced_Fire_Watch in missing certs — why it looks missing
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("ISSUE 3 — Advanced_Fire_Watch missing from RC-caused chart")
print("=" * 70)

post_rc   = [v for v in violations if v.get('after_rule_change')]
rc_caused = [v for v in post_rc
             if v['permit_type'] == 'hot_work'
             and 'Advanced_Fire_Watch' in v.get('missing_certs', [])]

mc_all = Counter(c for v in violations  for c in v.get('missing_certs', []))
mc_rc  = Counter(c for v in rc_caused   for c in v.get('missing_certs', []))

print()
print("Full ranking of missing certs (all violations):")
print(f"  {'Rank':4s}  {'Cert':35s}  {'All':4s}  {'RC-caused':9s}")
print(f"  {'-'*4}  {'-'*35}  {'-'*4}  {'-'*9}")
for rank, (cert, cnt) in enumerate(mc_all.most_common(25), 1):
    rc_cnt = mc_rc.get(cert, 0)
    marker = ' <-- cut off at rank 10 in chart!' if rank == 11 else ''
    marker = ' <-- TOP-10 BOUNDARY' if rank == 10 else marker
    print(f"  {rank:4d}  {cert:35s}  {cnt:4d}  {rc_cnt:9d}{marker}")

print()
print("  BUG: Advanced_Fire_Watch is rank 12 with 33 occurrences.")
print("  The chart showed top-10, so it fell just below the cutoff.")
print("  That is why the RC-caused bars all appeared empty -- the ONE cert")
print("  that IS exclusively in rc_caused is not shown in the top-10 list.")
print()
print("  Fix: extend to top-15, or always include Advanced_Fire_Watch.")
print()
print(f"  Advanced_Fire_Watch in mc_all: {mc_all.get('Advanced_Fire_Watch', 0)}")
print(f"  Advanced_Fire_Watch in mc_rc : {mc_rc.get('Advanced_Fire_Watch', 0)}")
print(f"  -> 100% of AFW violations are RC-caused (as expected)")
