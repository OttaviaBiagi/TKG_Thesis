"""Check zero-weight steps and hour distribution in the dataset."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
except ImportError:
    sys.exit('pandas required')

ds   = json.load(open('data/UseCase4/epc_dataset_real.json', encoding='utf-8'))
steps = ds['steps']

act_step_groups = {}
for s in steps:
    act_step_groups.setdefault(s['activity_id'], []).append(s['weight_pct'])

all_zero_acts = [a for a, ws in act_step_groups.items() if all(w == 0.0 for w in ws)]
print(f'Total steps:                          {len(steps)}')
print(f'Steps with weight_pct=0.0:            {sum(1 for s in steps if s["weight_pct"]==0.0)}')
print(f'Activities where ALL steps weight=0:  {len(all_zero_acts)}')

meram = pd.read_excel('data/UseCase4/meram/Meram_PCS_Progress.xlsx', sheet_name='Activities PCS')
meram['Estimated Hours'] = pd.to_numeric(meram['Estimated Hours'], errors='coerce').fillna(0)
est_map = dict(zip(meram['ActID'].astype(str).str.strip(), meram['Estimated Hours']))

wasted = [(a, est_map.get(a, 0)) for a in all_zero_acts if est_map.get(a, 0) > 0]
total_h = sum(h for _, h in wasted)
print(f'Of those, activities with est_h > 0:  {len(wasted)}  (total {total_h:.0f} estimated hours)')

# Discipline breakdown
meram['Disc'] = meram['Disc'].str.strip()
meram['Fami'] = meram['Fami'].str.strip()
disc_map = dict(zip(meram['ActID'].astype(str).str.strip(), meram['Disc']))
fami_map = dict(zip(meram['ActID'].astype(str).str.strip(), meram['Fami']))

from collections import Counter
disc_counts = Counter(disc_map.get(a) for a in all_zero_acts)
print(f'Discipline breakdown: {dict(disc_counts)}')

# Check: DEFAULT_STEPS template  (15/75/10 weights) vs zero-weight
# Activities NOT using defaults should have nonzero weight if template exists
fami_counts = Counter(fami_map.get(a) for a in all_zero_acts)
print(f'Top 5 families with all-zero steps: {fami_counts.most_common(5)}')
