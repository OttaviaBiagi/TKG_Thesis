"""Inject Experiment J (dependency + delay propagation) into nb06."""
import json
from pathlib import Path

NB = Path('notebooks/UseCase4/06_tkg_models.ipynb')
nb = json.load(NB.open(encoding='utf-8'))

MD_J = """\
## Experiment J — PRECEDES Dependency Validation & Accumulated Delay Propagation

Three sub-analyses on the synthetic EPC dataset:

**J1 — DAG integrity**: verify PRECEDES graph has no cycles, self-loops, or cross-activity edges.

**J2 — Accumulated delay via topological propagation**: if step Sₖ finishes late, all downstream
steps Sₖ₊₁ … Sₙ cannot start earlier → delays compound along the PRECEDES chain.
Model: for each step in topological order, `accumulated_delay = own_delay + max(parent accumulated delays)`.

**J3 — Rule-change impact**: distinguish violations *caused* by the new hot_work cert rule (month ≥ 6,
missing Advanced Fire Watch) from *coincident* pre-existing cert gaps that merely occur after the rule date.
"""

CODE_J = '''\
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone, date
import statistics as st

DATA_DIR   = Path('../../data/UseCase4')
EXP_DIR    = Path('../../experiments/UseCase4')
ds = json.load(open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8'))
ev = json.load(open(DATA_DIR / 'epc_events.json',       encoding='utf-8'))

steps      = {s['id']: s for s in ds['steps']}
activities = {a['id']: a for a in ds['activities']}
seqs       = ds['step_sequences']
completed  = {e['step_id']: e for e in ev['completed']}
violations = ev['permit_denied']
RULE_CHANGE_STR = '2024-06-29'

# ── Build graph ──────────────────────────────────────────────────────────────
children = defaultdict(list)
parents  = defaultdict(list)
for s in seqs:
    children[s['from']].append(s['to'])
    parents[s['to']].append(s['from'])

# ── J1 — DAG integrity ───────────────────────────────────────────────────────
cross_act  = sum(1 for s in seqs if steps[s['from']]['activity_id'] != steps[s['to']]['activity_id'])
self_loops = sum(1 for s in seqs if s['from'] == s['to'])
backward   = sum(1 for s in seqs if steps[s['to']]['order'] < steps[s['from']]['order'])

# Cycle detection (DFS)
visited, rec_stack, has_cycle = set(), set(), False
sys.setrecursionlimit(100_000)
def dfs(node):
    global has_cycle
    visited.add(node); rec_stack.add(node)
    for ch in children[node]:
        if ch not in visited: dfs(ch)
        elif ch in rec_stack: has_cycle = True
    rec_stack.discard(node)
for sid in steps:
    if sid not in visited: dfs(sid)

print("J1 — PRECEDES DAG integrity")
print(f"  Cross-activity edges : {cross_act}  (must be 0)")
print(f"  Self-loops           : {self_loops}  (must be 0)")
print(f"  Backward edges       : {backward}  (must be 0)")
print(f"  Cycles               : {has_cycle}  (must be False)")

# Chain-length distribution
steps_per_act = defaultdict(list)
for sid, s in steps.items():
    steps_per_act[s['activity_id']].append(s)
chain_lengths = sorted(len(v) for v in steps_per_act.values())
print(f"  Activities           : {len(chain_lengths)}")
print(f"  Steps/activity       : min={min(chain_lengths)}  max={max(chain_lengths)}  "
      f"mean={st.mean(chain_lengths):.1f}  median={int(st.median(chain_lengths))}")

# ── J2 — Topological delay propagation ──────────────────────────────────────
in_deg  = {sid: len(parents[sid]) for sid in steps}
queue   = deque([sid for sid in steps if in_deg[sid] == 0])
topo    = []
while queue:
    n = queue.popleft(); topo.append(n)
    for ch in children[n]:
        in_deg[ch] -= 1
        if in_deg[ch] == 0: queue.append(ch)

own_delay  = {sid: int(completed[sid]['delay_days']) if sid in completed else 0 for sid in steps}
prop_delay = {}
for sid in topo:
    prop_delay[sid] = max((prop_delay.get(p, 0) + own_delay.get(p, 0)
                           for p in parents[sid]), default=0)
acc_delay = {sid: prop_delay[sid] + own_delay[sid] for sid in steps}
all_acc   = list(acc_delay.values())
delayed   = [v for v in all_acc if v > 0]

print(f"\\nJ2 — Accumulated delay (topological propagation)")
print(f"  Steps with acc. delay > 0 : {len(delayed):,} / {len(steps):,} ({100*len(delayed)/len(steps):.0f}%)")
print(f"  Mean acc. delay (all)     : {st.mean(all_acc):.1f} days")
print(f"  Mean acc. delay (delayed) : {st.mean(delayed):.1f} days")
print(f"  Max acc. delay            : {max(all_acc)} days")

# ── J3 — Rule-change impact ──────────────────────────────────────────────────
pre_rc  = [v for v in violations if not v.get('after_rule_change')]
post_rc = [v for v in violations if v.get('after_rule_change')]
rc_caused = [v for v in post_rc
             if v['permit_type'] == 'hot_work'
             and 'Advanced_Fire_Watch' in v.get('missing_certs', [])]
rc_coinc  = [v for v in post_rc if v not in rc_caused]

print(f"\\nJ3 — Rule-change impact (hot_work, Advanced Fire Watch, month \\u22656)")
print(f"  Total violations      : {len(violations)}")
print(f"  Pre rule-change       : {len(pre_rc)}  (human error 5%)")
print(f"  Post RC - caused      : {len(rc_caused)}  (missing Advanced Fire Watch)")
print(f"  Post RC - coincident  : {len(rc_coinc)}  (pre-existing cert gaps)")

def to_month(d):
    if not d: return -1
    dt = datetime.fromisoformat(d[:10])
    return max(0, (dt.year-2024)*12 + dt.month - 1)

monthly = defaultdict(lambda: [0,0,0])   # [pre, caused, coincident]
for v in violations:
    m = to_month(v.get('date',''))
    if not v.get('after_rule_change'): monthly[m][0] += 1
    elif v in rc_caused:               monthly[m][1] += 1
    else:                              monthly[m][2] += 1

# ── Plots ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#1e1e2e')
BG, TX, GRID = '#181825', '#cdd6f4', '#313244'

# (1) Chain-length histogram
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor(BG)
from collections import Counter
cl_counts = Counter(chain_lengths)
ax1.bar(cl_counts.keys(), cl_counts.values(), color='#89b4fa', edgecolor=BG, width=0.7)
ax1.set_xlabel('Steps per activity', color=TX)
ax1.set_ylabel('# activities', color=TX)
ax1.set_title('J1 — Chain-length distribution', color=TX)
ax1.tick_params(colors=TX)
for sp in ['top','right']: ax1.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax1.spines[sp].set_color(GRID)

# (2) Accumulated delay histogram
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_facecolor(BG)
ax2.hist(all_acc, bins=40, color='#cba6f7', edgecolor=BG, range=(0, max(all_acc)))
ax2.axvline(st.mean(all_acc), color='#f38ba8', linestyle='--', lw=1.5,
            label=f'Mean {st.mean(all_acc):.0f}d')
ax2.axvline(st.median(all_acc), color='#a6e3a1', linestyle=':', lw=1.5,
            label=f'Median {int(st.median(all_acc))}d')
ax2.set_xlabel('Accumulated delay (days)', color=TX)
ax2.set_ylabel('# steps', color=TX)
ax2.set_title('J2 — Accumulated delay distribution\n(topological propagation)', color=TX)
ax2.tick_params(colors=TX)
ax2.legend(facecolor=GRID, labelcolor=TX, fontsize=8)
for sp in ['top','right']: ax2.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax2.spines[sp].set_color(GRID)

# (3) Delay by discipline boxplot
disc_delays = defaultdict(list)
for sid, d in acc_delay.items():
    disc_delays[steps[sid]['discipline']].append(d)
discs = sorted(disc_delays, key=lambda d: -st.mean(disc_delays[d]))
data_bp = [disc_delays[d] for d in discs]
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor(BG)
bp = ax3.boxplot(data_bp, labels=discs, patch_artist=True,
                  medianprops={'color':'#f38ba8','lw':2},
                  flierprops={'marker':'.','markersize':2,'markerfacecolor':'#6c7086'},
                  whiskerprops={'color':'#89b4fa'}, capprops={'color':'#89b4fa'})
for patch in bp['boxes']:
    patch.set_facecolor('#313244'); patch.set_edgecolor('#89b4fa')
ax3.set_xlabel('Discipline', color=TX)
ax3.set_ylabel('Accumulated delay (days)', color=TX)
ax3.set_title('J2 — Accumulated delay by discipline', color=TX)
ax3.tick_params(colors=TX, labelrotation=45)
for sp in ['top','right']: ax3.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax3.spines[sp].set_color(GRID)

# (4) Stacked bar: violations per month
months_with_data = sorted(m for m in monthly if m >= 0)
xs    = months_with_data
pre   = [monthly[m][0] for m in xs]
caus  = [monthly[m][1] for m in xs]
coinc = [monthly[m][2] for m in xs]
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_facecolor(BG)
ax4.bar(xs, pre,  color='#a6e3a1', label='Pre-RC (human error)', width=0.7)
ax4.bar(xs, caus, bottom=pre, color='#f38ba8',
        label=f'RC-caused (Adv. Fire Watch, n={len(rc_caused)})', width=0.7)
botc = [p+c for p,c in zip(pre,caus)]
ax4.bar(xs, coinc, bottom=botc, color='#fab387',
        label=f'Coincident post-RC (n={len(rc_coinc)})', width=0.7)
ax4.axvline(4.5, color='#f5c2e7', linestyle='--', lw=1.5, label='Rule change (Jun-29)')
ax4.set_xlabel('Month \\u03c4 (0 = Jan 2024)', color=TX)
ax4.set_ylabel('Violations', color=TX)
ax4.set_title('J3 — Violations per month by cause', color=TX)
ax4.tick_params(colors=TX)
ax4.legend(facecolor=GRID, labelcolor=TX, fontsize=7)
for sp in ['top','right']: ax4.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax4.spines[sp].set_color(GRID)

# (5) Delay propagation: own vs propagated for top-delayed steps
n_show = 30
top_sids = sorted(acc_delay, key=lambda s: -acc_delay[s])[:n_show]
own_vals  = [own_delay[s]  for s in top_sids]
prop_vals = [prop_delay[s] for s in top_sids]
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_facecolor(BG)
ys = range(n_show)
ax5.barh(list(ys), prop_vals, color='#cba6f7', label='Propagated from predecessors')
ax5.barh(list(ys), own_vals,  left=prop_vals, color='#f38ba8', label='Own delay')
ax5.set_xlabel('Delay (days)', color=TX)
ax5.set_ylabel(f'Top {n_show} most-delayed steps', color=TX)
ax5.set_title('J2 — Delay decomposition\\n(own vs. inherited)', color=TX)
ax5.tick_params(colors=TX)
ax5.set_yticks([])
ax5.legend(facecolor=GRID, labelcolor=TX, fontsize=8)
for sp in ['top','right']: ax5.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax5.spines[sp].set_color(GRID)

# (6) Missing certs in RC-caused violations
from collections import Counter as C2
mc_all   = C2(c for v in violations for c in v.get('missing_certs',[]))
mc_rc    = C2(c for v in rc_caused  for c in v.get('missing_certs',[]))
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor(BG)
top_mc = [k for k,_ in mc_all.most_common(10)]
vals_all = [mc_all.get(k,0) for k in top_mc]
vals_rc  = [mc_rc.get(k,0)  for k in top_mc]
x6 = np.arange(len(top_mc))
w  = 0.4
ax6.barh(x6+w/2, vals_all, w, color='#89b4fa', label='All violations')
ax6.barh(x6-w/2, vals_rc,  w, color='#f38ba8', label='RC-caused only')
ax6.set_yticks(x6)
ax6.set_yticklabels([k.replace('_',' ') for k in top_mc], fontsize=7, color=TX)
ax6.set_xlabel('# violations', color=TX)
ax6.set_title('J3 — Top missing certs\\n(all vs. RC-caused)', color=TX)
ax6.tick_params(colors=TX)
ax6.legend(facecolor=GRID, labelcolor=TX, fontsize=8)
for sp in ['top','right']: ax6.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax6.spines[sp].set_color(GRID)

plt.tight_layout(pad=2.0)
out = EXP_DIR / 'exp_j_delay_analysis.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight')
plt.close()
print(f"\\nSaved: {out}")
print("Experiment J complete.")
'''

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

nb['cells'].append(md_cell(MD_J))
nb['cells'].append(code_cell(CODE_J))

with NB.open('w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Injected Experiment J cells at positions {len(nb["cells"])-2} and {len(nb["cells"])-1}')
print(f'Total cells: {len(nb["cells"])}')
