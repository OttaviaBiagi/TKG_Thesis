"""
Patch epc_dataset_real.json: add estimated_hours to each Step.
Formula: step.estimated_hours = activity.estimated_hours * step.weight_pct / 100
Run once on any machine that has the JSON; no Excel files needed.
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / 'epc_dataset_real.json'

print(f'Loading {DATA_FILE} ...')
with open(DATA_FILE, encoding='utf-8') as f:
    d = json.load(f)

# Build activity -> estimated_hours lookup
act_hours = {a['id']: float(a.get('estimated_hours') or 0) for a in d['activities']}

# Patch steps
patched = 0
for s in d['steps']:
    if 'estimated_hours' not in s:
        ah = act_hours.get(s['activity_id'], 0.0)
        wp = float(s.get('weight_pct') or 0)
        s['estimated_hours'] = round(ah * wp / 100.0, 2)
        patched += 1

print(f'Patched {patched} steps with estimated_hours')
print(f'Sample: {d["steps"][0]["id"]} -> {d["steps"][0]["estimated_hours"]}h')

with open(DATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, default=str)

print(f'Written back to {DATA_FILE}')
