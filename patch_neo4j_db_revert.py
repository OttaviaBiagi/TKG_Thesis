"""
Revert all notebooks: remove database='ucX' from driver.session() calls.
Neo4j Community Edition does not support multiple databases.
"""
import json
import re
from pathlib import Path

SKIP_DIRS = {'archive_synth_v1'}

patched = []
for nb_path in Path('.').rglob('notebooks/**/*.ipynb'):
    if any(d in nb_path.parts for d in SKIP_DIRS):
        continue

    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        new_src = []
        for line in cell['source']:
            # Remove database= parameter from session calls
            new_line = re.sub(
                r'driver\.session\(database="uc\d+"\)',
                'driver.session()',
                line
            )
            if new_line != line:
                changed = True
            new_src.append(new_line)
        cell['source'] = new_src

    if changed:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        patched.append(str(nb_path))

for p in patched:
    print('reverted:', p)
print(f'Done: {len(patched)} notebooks reverted')
