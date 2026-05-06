"""Run Experiment J — generate delay analysis plot from notebook working dir."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, deque, Counter
from datetime import datetime, timezone
import statistics as st

# Paths relative to repo root (script is run from repo root)
DATA_DIR = Path('data/UseCase4')
EXP_DIR  = Path('experiments/UseCase4')

ds = json.load(open(DATA_DIR / 'epc_dataset_real.json', encoding='utf-8'))
ev = json.load(open(DATA_DIR / 'epc_events.json',       encoding='utf-8'))

steps      = {s['id']: s for s in ds['steps']}
activities = {a['id']: a for a in ds['activities']}
seqs       = ds['step_sequences']
completed  = {e['step_id']: e for e in ev['completed']}
violations = ev['permit_denied']

# ── Build graph ──────────────────────────────────────────────────────────────
children = defaultdict(list)
parents  = defaultdict(list)
for s in seqs:
    children[s['from']].append(s['to'])
    parents[s['to']].append(s['from'])

# ── J1 ───────────────────────────────────────────────────────────────────────
cross_act  = sum(1 for s in seqs if steps[s['from']]['activity_id'] != steps[s['to']]['activity_id'])
self_loops = sum(1 for s in seqs if s['from'] == s['to'])
backward   = sum(1 for s in seqs if steps[s['to']]['order'] < steps[s['from']]['order'])

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
print(f"  Cross-activity edges : {cross_act}   Self-loops: {self_loops}   Backward: {backward}   Cycles: {has_cycle}")

steps_per_act = defaultdict(list)
for sid, s in steps.items():
    steps_per_act[s['activity_id']].append(s)
chain_lengths = sorted(len(v) for v in steps_per_act.values())
print(f"  Steps/activity: min={min(chain_lengths)} max={max(chain_lengths)} "
      f"mean={st.mean(chain_lengths):.1f} median={int(st.median(chain_lengths))}")

# ── J2 ───────────────────────────────────────────────────────────────────────
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
    prop_delay[sid] = max((prop_delay.get(p,0) + own_delay.get(p,0) for p in parents[sid]), default=0)
acc_delay = {sid: prop_delay[sid] + own_delay[sid] for sid in steps}
all_acc   = list(acc_delay.values())
delayed   = [v for v in all_acc if v > 0]

print(f"\nJ2 — Accumulated delay")
print(f"  Delayed steps: {len(delayed)}/{len(steps)} ({100*len(delayed)/len(steps):.0f}%)  "
      f"Mean: {st.mean(all_acc):.1f}d  Max: {max(all_acc)}d")

# ── J3 ───────────────────────────────────────────────────────────────────────
pre_rc    = [v for v in violations if not v.get('after_rule_change')]
post_rc   = [v for v in violations if v.get('after_rule_change')]
rc_caused = [v for v in post_rc
             if v['permit_type'] == 'hot_work'
             and 'Advanced_Fire_Watch' in v.get('missing_certs', [])]
rc_coinc  = [v for v in post_rc if v not in rc_caused]

print(f"\nJ3 — Rule-change impact")
print(f"  Total violations: {len(violations)}  |  pre-RC: {len(pre_rc)}  "
      f"RC-caused: {len(rc_caused)}  coincident: {len(rc_coinc)}")

def to_month(d):
    if not d: return -1
    dt = datetime.fromisoformat(d[:10])
    return max(0, (dt.year-2024)*12 + dt.month - 1)

monthly = defaultdict(lambda: [0, 0, 0])
for v in violations:
    m = to_month(v.get('date', ''))
    if not v.get('after_rule_change'): monthly[m][0] += 1
    elif v in rc_caused:               monthly[m][1] += 1
    else:                              monthly[m][2] += 1

# ── Plots ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#1e1e2e')
BG, TX, GRID = '#181825', '#cdd6f4', '#313244'

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TX, fontsize=10)
    ax.set_xlabel(xlabel, color=TX)
    ax.set_ylabel(ylabel, color=TX)
    ax.tick_params(colors=TX)
    for sp in ['top', 'right']:  ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color(GRID)

# 1 — Chain-length histogram
ax1 = fig.add_subplot(2, 3, 1)
cl_counts = Counter(chain_lengths)
ax1.bar(cl_counts.keys(), cl_counts.values(), color='#89b4fa', edgecolor=BG, width=0.7)
style_ax(ax1, 'J1 — Steps-per-activity distribution', 'Steps per activity', '# activities')
ax1.axvline(st.mean(chain_lengths), color='#f38ba8', linestyle='--', lw=1.5,
            label=f'Mean {st.mean(chain_lengths):.1f}')
ax1.legend(facecolor=GRID, labelcolor=TX, fontsize=8)

# 2 — Accumulated delay histogram
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(all_acc, bins=40, color='#cba6f7', edgecolor=BG, range=(0, max(all_acc)))
ax2.axvline(st.mean(all_acc),    color='#f38ba8', linestyle='--', lw=1.5, label=f'Mean {st.mean(all_acc):.0f}d')
ax2.axvline(st.median(all_acc),  color='#a6e3a1', linestyle=':',  lw=1.5, label=f'Median {int(st.median(all_acc))}d')
style_ax(ax2, 'J2 — Accumulated delay distribution\n(topological propagation)',
         'Accumulated delay (days)', '# steps')
ax2.legend(facecolor=GRID, labelcolor=TX, fontsize=8)

# 3 — Delay by discipline (boxplot)
disc_delays = defaultdict(list)
for sid, d in acc_delay.items():
    disc_delays[steps[sid]['discipline']].append(d)
discs   = sorted(disc_delays, key=lambda d: -st.mean(disc_delays[d]))
data_bp = [disc_delays[d] for d in discs]
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor(BG)
bp = ax3.boxplot(data_bp, labels=discs, patch_artist=True,
                  medianprops={'color': '#f38ba8', 'lw': 2},
                  flierprops={'marker': '.', 'markersize': 2, 'markerfacecolor': '#6c7086'},
                  whiskerprops={'color': '#89b4fa'}, capprops={'color': '#89b4fa'})
for patch in bp['boxes']:
    patch.set_facecolor('#313244'); patch.set_edgecolor('#89b4fa')
style_ax(ax3, 'J2 — Accumulated delay by discipline', 'Discipline', 'Accumulated delay (days)')
ax3.tick_params(axis='x', labelrotation=45)

# 4 — Stacked bar: violations per month
months_with_data = sorted(m for m in monthly if m >= 0)
xs    = months_with_data
pre   = [monthly[m][0] for m in xs]
caus  = [monthly[m][1] for m in xs]
coinc = [monthly[m][2] for m in xs]
ax4 = fig.add_subplot(2, 3, 4)
ax4.set_facecolor(BG)
ax4.bar(xs, pre,  color='#a6e3a1', label='Pre-RC (human error)', width=0.7)
ax4.bar(xs, caus, bottom=pre, color='#f38ba8',
        label=f'RC-caused (new AFW cert, n={len(rc_caused)})', width=0.7)
botc = [p+c for p, c in zip(pre, caus)]
ax4.bar(xs, coinc, bottom=botc, color='#fab387',
        label=f'Coincident post-RC (n={len(rc_coinc)})', width=0.7)
ax4.axvline(4.5, color='#f5c2e7', linestyle='--', lw=1.5, label='Rule change (Jun-29)')
style_ax(ax4, 'J3 — Violations per month', 'Month tau (0 = Jan 2024)', 'Violations')
ax4.legend(facecolor=GRID, labelcolor=TX, fontsize=7)

# 5 — Delay decomposition: own vs propagated
n_show   = 30
top_sids = sorted(acc_delay, key=lambda s: -acc_delay[s])[:n_show]
own_v    = [own_delay[s]  for s in top_sids]
prop_v   = [prop_delay[s] for s in top_sids]
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_facecolor(BG)
ys = list(range(n_show))
ax5.barh(ys, prop_v, color='#cba6f7', label='Inherited from predecessors')
ax5.barh(ys, own_v,  left=prop_v, color='#f38ba8', label='Own delay')
style_ax(ax5, f'J2 — Delay decomposition\n(top {n_show} steps)', 'Delay (days)', '')
ax5.set_yticks([])
ax5.legend(facecolor=GRID, labelcolor=TX, fontsize=8)

# 6 — Missing certs: all vs RC-caused
mc_all = Counter(c for v in violations for c in v.get('missing_certs', []))
mc_rc  = Counter(c for v in rc_caused  for c in v.get('missing_certs', []))
top_mc = [k for k, _ in mc_all.most_common(10)]
x6 = np.arange(len(top_mc))
w  = 0.4
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor(BG)
ax6.barh(x6 + w/2, [mc_all.get(k, 0) for k in top_mc], w, color='#89b4fa', label='All violations')
ax6.barh(x6 - w/2, [mc_rc.get(k, 0)  for k in top_mc], w, color='#f38ba8', label='RC-caused only')
ax6.set_yticks(x6)
ax6.set_yticklabels([k.replace('_', ' ') for k in top_mc], fontsize=7, color=TX)
style_ax(ax6, 'J3 — Missing certs: all vs RC-caused', '# violations', '')
ax6.legend(facecolor=GRID, labelcolor=TX, fontsize=8)

plt.tight_layout(pad=2.0)
out = EXP_DIR / 'exp_j_delay_analysis.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
print("Experiment J complete.")
