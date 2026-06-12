# EPC Compliance Experiments — Training & Evaluation Pipeline

This directory contains the complete model training and evaluation pipeline for the thesis benchmark.

## Pipeline (run in this order)

```bash
# 1. Data split + evaluation helpers (imported by all scripts)
#    eval_framework.py, data_loader.py — not run directly

# 2. Hyperparameter tuning (optional — results already in best_params.json)
python tune_hyperparams.py

# 3. Main benchmark: TGN / TGAT / DyRep
python run_benchmark.py --dataset single --seeds 42        # single project (~45 min GPU)
python run_benchmark.py --dataset multi_varied --seeds 42  # 30 varied EPC families

# 4. ML feature-only baselines
python run_ml_baseline.py --dataset single   # single project baseline
python run_ml_baseline.py --dataset multi    # multi-project baseline (100 projects)

# 5. Static KG baselines
python run_static_baseline.py --model all --dataset all    # ComplEx + TNTComplEx
python run_static_gnn.py --dataset single                  # StaticGNN

# 6. Open results notebook
jupyter lab ../../notebooks/UseCase4/08_model_benchmark_final.ipynb
```

## Key files

| File | Role |
|------|------|
| `eval_framework.py` | `split_dataset`, `compute_metrics`, `find_best_threshold` — shared by all scripts |
| `data_loader.py` | `load_single_project`, `load_multi_project` |
| `models/tgn.py` | TGN implementation (Rossi et al. 2020) |
| `models/tgat.py` | TGAT implementation (Xu et al. 2020) |
| `models/dyrep.py` | DyRep implementation (Zuo et al. 2018) |

## Results

All results land in `results/`. See [THESIS_GUIDE.md](../../THESIS_GUIDE.md) for the mapping to thesis sections.

### Multi-project summary

Recent multi-project results show that the RF feature-only baseline is competitive with TGN temporal on the multi dataset, but TGN still preserves the best predictive quality when temporal-graph structure is taken into account.

- `RandomForest` multi:
  - AUC = 0.9803
  - AUPRC = 0.0907
  - F1 = 0.0963
  - Training time = 112.8s
- `TGN` multi `temporal`:
  - AUC = 0.9811
  - AUPRC = 0.0940
  - F1 = 0.0975
  - Training time = 349.7s

### Interpretation

- `RandomForest` is a strong baseline on multi, showing that feature engineering already captures much of the signal.
- `TGN temporal` remains the best model for the task because it explicitly models temporal graph dynamics and therefore slightly outperforms RF on the key ranking metrics.

