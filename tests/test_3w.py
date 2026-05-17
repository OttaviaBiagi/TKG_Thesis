import pandas as pd
from pathlib import Path

DATA_ROOT = Path('data/UseCase2/3w_dataset')

if DATA_ROOT.exists():
    print(f'✅ Dataset trovato in {DATA_ROOT}')
    for event_dir in sorted(DATA_ROOT.iterdir()):
        if event_dir.is_dir() and event_dir.name.isdigit():
            files = list(event_dir.glob('*.parquet'))
            print(f'  Type {event_dir.name}: {len(files)} files')
else:
    print(f'❌ Dataset non trovato')