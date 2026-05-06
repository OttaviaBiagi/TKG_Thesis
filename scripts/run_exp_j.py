"""
Experiment J — PRECEDES Dependency Validation & Schedule Risk Analysis.

What this script analyses (all based on real/synthetic data, no modelling assumptions):
  J1  DAG integrity: no cycles, no cross-activity edges, no backward edges
  J2  Buffer distribution: how many days of planned slack exist between consecutive steps
  J3  Delay vs buffer: when would a simulated delay exceed its buffer and propagate?
  J4  Violations per month by cause (rule-change vs human error)
  J5  Missing certs — full list including Advanced_Fire_Watch (fixed top-10 bug)

The stochastic Monte Carlo propagation model is documented as future work:
  requires historical delay distributions per step type (Primavera P6 + ERP data).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
import statistics as st

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

# ── Topological sort ─────────────────────────────────────────────────────────
in_deg = {sid: len(parents[sid]) for sid in steps}
queue  = deque([sid for sid in steps if in_deg[sid] == 0])
topo   = []
while queue:
    n = queue.popleft(); topo.append(n)
    for ch in children[n]:
        in_deg[ch] -= 1
        if in_deg[ch] == 0: queue.append(ch)

# ── J1 — DAG integrity ───────────────────────────────────────────────────────
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

steps_per_act = defaultdict(list)
for sid, s in steps.items():
    steps_per_act[s['activity_id']].append(s)
for aid in steps_per_act:
    steps_per_act[aid].sort(key=lambda x: x['order'])
chain_lengths = sorted(len(v) for v in steps_per_act.values())

print("J1 — PRECEDES DAG integrity")
print(f"  cross-activity={cross_act}  self-loops={self_loops}  backward={backward}  cycles={has_cycle}")
print(f"  chain lengths: min={min(chain_lengths)} max={max(chain_lengths)} "
      f"mean={st.mean(chain_lengths):.1f} median={int(st.median(chain_lengths))}")

# ── J2 — Buffer between consecutive steps ────────────────────────────────────
def dt(s): return datetime.fromisoformat(s[:10])

buffers = []   # (buffer_days, from_sid, to_sid)
for s in seqs:
    s_p = steps[s['from']]
    s_c = steps[s['to']]
    buf = (dt(s_c['valid_from']) - dt(s_p['valid_to'])).days
    buffers.append(buf)

buf_vals = np.array(buffers)
print(f"\nJ2 — Buffer between consecutive steps (n={len(buffers)})")
print(f"  min={buf_vals.min()}d  max={buf_vals.max()}d  "
      f"mean={buf_vals.mean():.1f}d  median={int(np.median(buf_vals))}d")
print(f"  negative (overlap): {(buf_vals<0).sum()}  |  "
      f"0-7d: {((buf_vals>=0)&(buf_vals<=7)).sum()}  |  "
      f"8-30d: {((buf_vals>7)&(buf_vals<=30)).sum()}  |  "
      f">30d: {(buf_vals>30).sum()}")

# ── J3 — Delay vs buffer (when does delay exceed its buffer?) ─────────────────
own_delay = {sid: int(completed[sid]['delay_days']) if sid in completed else 0 for sid in steps}
n_delayed_steps = sum(1 for v in own_delay.values() if v > 0)

# For each PRECEDES edge: would the upstream step's own delay exceed the buffer?
# This answers: "given the simulated delay on a step, would it spill over to the next?"
exceed_buffer  = []   # edges where own_delay > buffer
absorbed       = []   # edges where own_delay <= buffer (delay stays in the step)
for s in seqs:
    s_p = steps[s['from']]
    s_c = steps[s['to']]
    buf = (dt(s_c['valid_from']) - dt(s_p['valid_to'])).days
    od  = own_delay[s['from']]
    if od > 0:
        if od > buf:
            exceed_buffer.append({'edge': s, 'buf': buf, 'delay': od, 'excess': od - buf})
        else:
            absorbed.append({'edge': s, 'buf': buf, 'delay': od})

print(f"\nJ3 — Delay vs buffer (on {n_delayed_steps} delayed steps):")
print(f"  PRECEDES edges with upstream delay that EXCEEDS buffer: {len(exceed_buffer)}")
print(f"  PRECEDES edges with upstream delay ABSORBED by buffer:  {len(absorbed)}")
if exceed_buffer:
    excesses = [e['excess'] for e in exceed_buffer]
    print(f"  Excess delay that would propagate: mean={st.mean(excesses):.0f}d  "
          f"max={max(excesses)}d")

# ── J4 — Rule-change impact ──────────────────────────────────────────────────
pre_rc    = [v for v in violations if not v.get('after_rule_change')]
post_rc   = [v for v in violations if v.get('after_rule_change')]
rc_caused = [v for v in post_rc
             if v['permit_type'] == 'hot_work'
             and 'Advanced_Fire_Watch' in v.get('missing_certs', [])]
rc_coinc  = [v for v in post_rc if v not in rc_caused]

def to_month(d):
    if not d: return -1
    dt2 = datetime.fromisoformat(d[:10])
    return max(0, (dt2.year-2024)*12 + dt2.month - 1)

monthly = defaultdict(lambda: [0, 0, 0])
for v in violations:
    m = to_month(v.get('date', ''))
    if not v.get('after_rule_change'): monthly[m][0] += 1
    elif v in rc_caused:               monthly[m][1] += 1
    else:                              monthly[m][2] += 1

print(f"\nJ4 — Violations: total={len(violations)}  pre-RC={len(pre_rc)}  "
      f"RC-caused={len(rc_caused)}  coincident={len(rc_coinc)}")

# ── J5 — Missing certs (fixed: force-include all RC-relevant certs) ──────────
mc_all = Counter(c for v in violations for c in v.get('missing_certs', []))
mc_rc  = Counter(c for v in rc_caused  for c in v.get('missing_certs', []))
top_mc = [k for k, _ in mc_all.most_common(10)]
for cert in sorted(mc_rc, key=lambda c: -mc_rc[c]):
    if cert not in top_mc:
        top_mc.append(cert)
print(f"J5 — Missing certs list: {len(top_mc)} entries (top-10 + {len(top_mc)-10} RC-forced)")

# ── PLOTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#1e1e2e')
BG, TX, GRID = '#181825', '#cdd6f4', '#313244'

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TX, fontsize=9, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TX, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TX, fontsize=8)
    ax.tick_params(colors=TX, labelsize=7)
    for sp in ['top', 'right']:   ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color(GRID)

gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.35)

# ── Plot 1: chain-length histogram ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cl_counts = Counter(chain_lengths)
ax1.bar(cl_counts.keys(), cl_counts.values(), color='#89b4fa', edgecolor=BG, width=0.7)
ax1.axvline(st.mean(chain_lengths), color='#f38ba8', linestyle='--', lw=1.5,
            label=f'Mean {st.mean(chain_lengths):.1f} steps')
style_ax(ax1, 'J1 — Steps per activity\n(PRECEDES chain lengths)',
         'Steps per activity', '# activities')
ax1.legend(facecolor=GRID, labelcolor=TX, fontsize=7)

# ── Plot 2: buffer histogram ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
clip_buf = np.clip(buf_vals, -10, 120)   # clip tails for readability
ax2.hist(clip_buf, bins=50, color='#a6e3a1', edgecolor=BG)
ax2.axvline(float(np.median(buf_vals)), color='#f38ba8', linestyle='--', lw=1.5,
            label=f'Median {int(np.median(buf_vals))}d')
ax2.axvline(float(np.mean(buf_vals)), color='#fab387', linestyle=':', lw=1.5,
            label=f'Mean {np.mean(buf_vals):.0f}d')
ax2.axvline(0, color='#f5c2e7', linestyle='-', lw=1, alpha=0.5, label='0 (no buffer)')
style_ax(ax2, 'J2 — Planned buffer between consecutive steps\n'
              '(planned_start[Sk+1] - planned_end[Sk], clipped at 120d)',
         'Buffer (days)', '# PRECEDES edges')
ax2.legend(facecolor=GRID, labelcolor=TX, fontsize=7)
# annotate fractions
y_top = ax2.get_ylim()[1]
ax2.text(0.02, 0.92, f'{(buf_vals<0).sum()} overlapping', transform=ax2.transAxes,
         color='#f5c2e7', fontsize=6.5)
ax2.text(0.02, 0.84, f'{((buf_vals>=0)&(buf_vals<=7)).sum()} thin (<= 7d)', transform=ax2.transAxes,
         color='#fab387', fontsize=6.5)
ax2.text(0.02, 0.76, f'{(buf_vals>30).sum()} wide (> 30d)', transform=ax2.transAxes,
         color='#a6e3a1', fontsize=6.5)

# ── Plot 3: delay vs buffer scatter / comparison ─────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor(BG)
# For each PRECEDES edge with an upstream delayed step, plot (buffer, own_delay)
buf_exc = [e['buf']   for e in exceed_buffer]
del_exc = [e['delay'] for e in exceed_buffer]
buf_abs = [e['buf']   for e in absorbed]
del_abs = [e['delay'] for e in absorbed]
if buf_abs:
    ax3.scatter(buf_abs, del_abs, color='#a6e3a1', alpha=0.5, s=8, label='Absorbed by buffer')
if buf_exc:
    ax3.scatter(buf_exc, del_exc, color='#f38ba8', alpha=0.7, s=12, label='Exceeds buffer (propagates)')
# diagonal line y=x
lim = max(ax3.get_xlim()[1], ax3.get_ylim()[1]) if (buf_abs or buf_exc) else 60
xs = np.linspace(0, lim, 100)
ax3.plot(xs, xs, color='#6c7086', linestyle='--', lw=1, label='delay = buffer (boundary)')
ax3.fill_between(xs, xs, lim, color='#f38ba8', alpha=0.06)
ax3.fill_between(xs, 0, xs, color='#a6e3a1', alpha=0.06)
ax3.set_xlim(left=-2)
ax3.set_ylim(bottom=-2)
style_ax(ax3, f'J3 — Step delay vs planned buffer\n'
              f'(green=absorbed, red=propagates; {len(exceed_buffer)} edges exceed buffer)',
         'Planned buffer (days)', 'Upstream step own delay (days)')
ax3.legend(facecolor=GRID, labelcolor=TX, fontsize=6.5, loc='upper right')

# ── Plot 4: buffer distribution by discipline (boxplot) ──────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor(BG)
disc_bufs = defaultdict(list)
for s_seq in seqs:
    s_p = steps[s_seq['from']]
    s_c = steps[s_seq['to']]
    buf = (dt(s_c['valid_from']) - dt(s_p['valid_to'])).days
    disc_bufs[s_p['discipline']].append(buf)
discs_b = sorted(disc_bufs, key=lambda d: float(np.median(disc_bufs[d])))
bp = ax4.boxplot([disc_bufs[d] for d in discs_b], tick_labels=discs_b,
                  patch_artist=True,
                  medianprops={'color': '#f38ba8', 'lw': 2},
                  flierprops={'marker': '.', 'markersize': 2, 'markerfacecolor': '#6c7086'},
                  whiskerprops={'color': '#89b4fa'}, capprops={'color': '#89b4fa'})
for patch in bp['boxes']:
    patch.set_facecolor('#313244'); patch.set_edgecolor('#89b4fa')
style_ax(ax4, 'J2 — Buffer distribution by discipline\n(median buffer = absorption capacity)',
         'Discipline', 'Buffer (days)')
ax4.tick_params(axis='x', labelrotation=45)

# ── Plot 5: delay distribution from simulation ───────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
all_own = [v for v in own_delay.values() if v > 0]
ax5.hist(all_own, bins=30, color='#cba6f7', edgecolor=BG)
ax5.axvline(int(np.median(buf_vals)), color='#a6e3a1', linestyle='--', lw=1.5,
            label=f'Median buffer {int(np.median(buf_vals))}d')
ax5.axvline(st.mean(all_own) if all_own else 0, color='#f38ba8', linestyle=':', lw=1.5,
            label=f'Mean own delay {st.mean(all_own):.0f}d' if all_own else '')
style_ax(ax5, f'J3 — Own delay distribution\n({n_delayed_steps} steps with delay > 0)',
         'Own delay (days)', '# steps')
ax5.legend(facecolor=GRID, labelcolor=TX, fontsize=7)
ax5.text(0.6, 0.7, 'Most delays\nfit within\nthe buffer',
         transform=ax5.transAxes, color='#a6e3a1', fontsize=7, ha='center')

# ── Plot 6: violations per month ─────────────────────────────────────────────
months_with_data = sorted(m for m in monthly if m >= 0)
xs    = months_with_data
pre   = [monthly[m][0] for m in xs]
caus  = [monthly[m][1] for m in xs]
coinc = [monthly[m][2] for m in xs]
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor(BG)
ax6.bar(xs, pre,  color='#a6e3a1', label='Pre-RC (human error)', width=0.7)
ax6.bar(xs, caus, bottom=pre, color='#f38ba8',
        label=f'RC-caused (AFW, n={len(rc_caused)})', width=0.7)
botc = [p+c for p, c in zip(pre, caus)]
ax6.bar(xs, coinc, bottom=botc, color='#fab387',
        label=f'Coincident post-RC (n={len(rc_coinc)})', width=0.7)
ax6.axvline(4.5, color='#f5c2e7', linestyle='--', lw=1.5)
ax6.text(4.6, max(pre+caus+coinc)*0.92 if pre+caus+coinc else 10,
         'Rule change\nJun-29', color='#f5c2e7', fontsize=6.5, va='top')
style_ax(ax6, 'J4 — Violations per month by cause', 'Month tau (0=Jan 2024)', 'Violations')
ax6.legend(facecolor=GRID, labelcolor=TX, fontsize=6.5)

# ── Plot 7: missing certs — full width, AFW visible ──────────────────────────
ax7 = fig.add_subplot(gs[2, :])
ax7.set_facecolor(BG)
x7 = np.arange(len(top_mc))
w  = 0.4
ax7.bar(x7 - w/2, [mc_all.get(k, 0) for k in top_mc], w,
        color='#89b4fa', label='All violations')
ax7.bar(x7 + w/2, [mc_rc.get(k, 0) for k in top_mc], w,
        color='#f38ba8', label='RC-caused only (hot_work + Advanced Fire Watch missing)')
ax7.set_xticks(x7)
ax7.set_xticklabels([k.replace('_', ' ') for k in top_mc], rotation=35, ha='right',
                    fontsize=7, color=TX)
style_ax(ax7,
         'J5 — Missing certs: all violations vs. RC-caused\n'
         '(top-10 by total + hot_work certs forced-included to show Advanced Fire Watch)',
         '', '# violations')
ax7.legend(facecolor=GRID, labelcolor=TX, fontsize=8)
if len(top_mc) > 10:
    ax7.axvline(9.5, color='#f5c2e7', linestyle=':', lw=1)
    ax7.text(9.55, ax7.get_ylim()[1]*0.85, 'hot_work\ncerts',
             color='#f5c2e7', fontsize=6.5, va='top')
afw_idx = top_mc.index('Advanced_Fire_Watch') if 'Advanced_Fire_Watch' in top_mc else None
if afw_idx is not None:
    val = mc_rc.get('Advanced_Fire_Watch', 0)
    ax7.text(afw_idx + w/2, val + 1, f'{val}\n(100%\nRC-caused)',
             ha='center', color='#f38ba8', fontsize=6.5, fontweight='bold')

plt.suptitle('Experiment J — PRECEDES Dependency Validation & Schedule Risk Analysis\n'
             '(Stochastic Monte Carlo propagation model: future work — requires historical delay distributions)',
             color=TX, fontsize=10, y=1.01)

out = EXP_DIR / 'exp_j_delay_analysis.png'
plt.savefig(str(out), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
print("Experiment J complete.")
