import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCHMARK_MERGED = ROOT / "results" / "benchmark_merged.json"
BENCHMARK_CSV = ROOT / "results" / "benchmark.csv"
OUTPUT = ROOT / "results" / "current_summary.txt"


def main() -> None:
    with BENCHMARK_MERGED.open("r", encoding="utf-8") as f:
        merged = json.load(f)["results"]

    agg: dict[tuple[str, str, str], dict] = {}
    for row in merged:
        key = (row["dataset"], row["model"], row["split"])
        metrics = row["metrics"] if isinstance(row["metrics"], dict) else row["metrics"].get("overall", {})
        agg[key] = metrics

    lines = []
    lines.append("=== benchmark_merged.json summary ===")
    for dataset, model, split in sorted(agg):
        m = agg[(dataset, model, split)]
        lines.append(
            f"{dataset} | {model} | {split} | auc={m.get('auc', float('nan')):.4f} | "
            f"auprc={m.get('auprc', float('nan')):.4f} | f1={m.get('f1', float('nan')):.4f} | "
            f"n_pos_test={m.get('n_pos_test')}"
        )

    if BENCHMARK_CSV.exists():
        import pandas as pd
        df = pd.read_csv(BENCHMARK_CSV)
        if not df.empty:
            lines.append("\n=== benchmark.csv multi_varied summary ===")
            for _, row in df[['dataset', 'model', 'split', 'auc', 'auprc', 'f1', 'n_pos_test']].iterrows():
                lines.append(
                    f"{row['dataset']} | {row['model']} | {row['split']} | auc={row['auc']:.4f} | "
                    f"auprc={row['auprc']:.4f} | f1={row['f1']:.4f} | n_pos_test={int(row['n_pos_test'])}"
                )

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
