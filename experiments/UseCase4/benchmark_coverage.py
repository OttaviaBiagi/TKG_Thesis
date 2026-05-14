"""Inspect UseCase4 benchmark outputs and report missing model/split/dataset combinations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'

EXPECTED_DATASETS = ['single', 'multi', 'multi_varied']
EXPECTED_MODELS = ['TGN', 'DyRep', 'TGAT']
EXPECTED_SPLITS = ['stratified', 'temporal', '6slot', 'inductive']

SOURCE_FILES = [
    ('benchmark_merged.json', 'merged single+multi benchmark'),
    ('benchmark.json', 'current benchmark output'),
    ('benchmark_varied.json', 'varied benchmark output'),
]


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return []
    return data.get('results', []) if isinstance(data, dict) else []


def metrics_summary(metrics: dict[str, Any]) -> str:
    if 'overall' in metrics:
        metrics = metrics['overall']
    return (
        f"auc={metrics.get('auc', float('nan')):.4f} "
        f"auprc={metrics.get('auprc', float('nan')):.4f} "
        f"f1={metrics.get('f1', float('nan')):.4f}"
    )


def main() -> None:
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources = {}

    for filename, source_name in SOURCE_FILES:
        path = RESULTS / filename
        entries = load_results(path)
        for entry in entries:
            key = (entry['dataset'], entry['model'], entry['split'])
            if key not in results:
                results[key] = entry
                sources[key] = source_name

    all_expected = [
        (dataset, model, split)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for split in EXPECTED_SPLITS
    ]

    present = [k for k in all_expected if k in results]
    missing = [k for k in all_expected if k not in results]
    skipped = [k for k, r in results.items() if r.get('skipped')]

    lines = []
    lines.append('UseCase4 benchmark coverage report')
    lines.append('===================================\n')
    lines.append(f'Total expected combos: {len(all_expected)}')
    lines.append(f'Present combos:        {len(present)}')
    lines.append(f'Missing combos:        {len(missing)}')
    lines.append(f'Skipped combos:        {len(skipped)}\n')

    lines.append('Sources:')
    for filename, source_name in SOURCE_FILES:
        lines.append(f'  {source_name}: {filename} (exists={ (RESULTS / filename).exists() })')
    lines.append('')

    if missing:
        lines.append('Missing combinations:')
        for dataset, model, split in sorted(missing):
            lines.append(f'  {dataset:12s} {model:5s} {split}')
        lines.append('')

    lines.append('Present results:')
    for dataset, model, split in sorted(present):
        entry = results[(dataset, model, split)]
        source = sources[(dataset, model, split)]
        if entry.get('skipped'):
            summary = 'SKIPPED'
        else:
            summary = metrics_summary(entry.get('metrics', {}))
        lines.append(f'  {dataset:12s} {model:5s} {split:9s} {summary}  [{source}]')

    report = '\n'.join(lines)
    report_path = RESULTS / 'benchmark_coverage.txt'
    report_path.write_text(report, encoding='utf-8')
    print(report)
    print(f'\nSaved coverage report -> {report_path}')


if __name__ == '__main__':
    main()
