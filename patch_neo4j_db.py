"""Patch all active notebooks to use per-use-case Neo4j databases."""
import json
from pathlib import Path

MAPPING = [
    ('notebooks/UseCase1', 'uc1'),
    ('notebooks/UseCase2', 'uc2'),
    ('notebooks/UseCase3', 'uc3'),
    ('notebooks/UseCase4', 'uc4'),
]
SKIP_DIRS = {'archive_synth_v1'}

patched = []
root = Path('.')
for nb_path in root.rglob('notebooks/**/*.ipynb'):
    parts = nb_path.parts
    if any(d in parts for d in SKIP_DIRS):
        continue

    db = None
    posix = nb_path.as_posix()
    for prefix, dbname in MAPPING:
        if posix.startswith(prefix):
            db = dbname
            break
    if not db:
        continue

    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        new_src = []
        for line in cell['source']:
            new_line = line.replace(
                'driver.session()',
                f'driver.session(database="{db}")'
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
    print('patched:', p)
print(f'Done: {len(patched)} notebooks updated')
