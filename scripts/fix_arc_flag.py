"""Fix the 9 permit_denied events where after_rule_change=True but date < 2024-06-29."""
import json
from pathlib import Path

PATH = Path('data/UseCase4/epc_events.json')
ev = json.load(PATH.open(encoding='utf-8'))

RULE_CHANGE = '2024-06-29'
fixed = 0
for e in ev['permit_denied']:
    if e.get('after_rule_change') and e['date'] < RULE_CHANGE:
        e['after_rule_change'] = False
        fixed += 1

print(f'Fixed {fixed} events (after_rule_change True -> False)')
print(f'Remaining after_rule_change=True: {sum(1 for e in ev["permit_denied"] if e.get("after_rule_change"))}')

with PATH.open('w', encoding='utf-8') as f:
    json.dump(ev, f, indent=2)
print('Saved.')
