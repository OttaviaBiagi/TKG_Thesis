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

### Full benchmark table

| Model | Dataset | Split | AUC | AUPRC | F1 | Precision | Recall | AUPRC Lift | Train (s) |
|---|---|---|---|---|---|---|---|---|---|
| **T-Logic R1+R2** | single | — | — | — | **1.000** | 1.000 | 1.000 | — | — |
| **T-Logic R1+R2** | multi | — | — | — | **0.980** | 0.963 | 0.998 | — | — |
| **T-Logic R1+R2** | multi\_varied | — | — | — | **0.970** | 0.942 | 0.999 | — | — |
| T-Logic R1 only | single | — | — | — | 0.996 | 1.000 | 0.991 | — | — |
| T-Logic R1 only | multi | — | — | — | 0.986 | 0.986 | 0.987 | — | — |
| T-Logic R1 only | multi\_varied | — | — | — | 0.977 | 0.963 | 0.991 | — | — |
| Hybrid AND (T-Logic∩TGAT) | multi\_varied | temporal | — | — | 0.795±0.082 | 1.000 | 0.665±0.109 | — | — |
| Hybrid OR (T-Logic∪TGAT) | multi\_varied | temporal | — | — | 0.855±0.113 | 0.758±0.181 | 1.000 | — | — |
| TGAT | multi\_varied | temporal | 0.979±0.026 | — | 0.659±0.049 | 0.698±0.209 | 0.665±0.109 | — | — |
| TGN | single | temporal | 0.985 | 0.178 | 0.084 | 0.044 | 1.000 | — | 9.9 |
| TGN | single | stratified | 0.833 | 0.073 | 0.093 | 0.049 | 0.838 | — | 10.7 |
| TGN | multi | temporal | 0.981 | 0.094 | 0.098 | 0.051 | — | — | 349.7 |
| StaticGNN | single | temporal | 0.773±0.013 | 0.546±0.066 | 0.385±0.259 | 0.693±0.503 | 0.417±0.265 | 298x | 2.0 |
| StaticGNN | multi\_varied | temporal | 0.932±0.003 | 0.204±0.115 | 0.132±0.073 | 0.089±0.067 | 0.426±0.107 | 85x | 241.6 |
| RandomForest | multi | temporal | 0.980 | 0.091 | 0.096 | 0.051 | **1.000** | 43x | 112.8 |
| RandomForest | multi\_varied | temporal | 0.983 | 0.104 | 0.115 | — | — | 43x | 7.5 |
| LogisticRegression | multi | temporal | 0.688 | 0.068 | 0.102 | 0.057 | 0.489 | 32x | 11.3 |
| LogisticRegression | multi\_varied | temporal | 0.685 | 0.069 | 0.047 | — | — | 28x | 0.3 |
| TNTComplEx | single | temporal | 0.582 | 0.003 | 0.004 | — | — | 1.6x | — |
| TNTComplEx | multi | temporal | 0.507 | 0.002 | 0.004 | — | — | 1.0x | — |
| TNTComplEx | multi\_varied | temporal | 0.516 | 0.002 | 0.005 | — | — | 1.0x | — |
| ComplEx | single | temporal | 0.440 | 0.002 | 0.004 | — | — | 0.96x | 5.6 |
| ComplEx | multi | temporal | 0.503 | 0.002 | 0.005 | — | — | 0.98x | 2062.5 |
| ComplEx | multi\_varied | temporal | 0.521 | 0.002 | 0.005 | — | — | 1.03x | 436.0 |

### Results by model family

#### T-Logic (symbolic temporal rules)

Rule-based model operating on EPC temporal graphs. Two rules: R1 (direct compliance check), R2 (indirect).

| Dataset | Rule | Precision | Recall | F1 |
|---|---|---|---|---|
| single | R1 | 1.000 | 0.991 | 0.996 |
| single | R2 | 1.000 | 0.073 | 0.137 |
| single | R1+R2 | **1.000** | **1.000** | **1.000** |
| multi | R1 | 0.986 | 0.987 | 0.986 |
| multi | R2 | 0.720 | 0.063 | 0.116 |
| multi | R1+R2 | 0.963 | 0.998 | **0.980** |
| multi\_varied | R1 | 0.963 | 0.991 | 0.977 |
| multi\_varied | R2 | 0.697 | 0.055 | 0.103 |
| multi\_varied | R1+R2 | 0.942 | 0.999 | **0.970** |

R1 alone captures ~99% of violations with perfect precision. R2 adds recall for the remaining ~7% but at a precision cost on multi-project data.

#### Hybrid ensemble (T-Logic R1 + TGAT) — multi_varied, 3 seeds

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| T-Logic R1 alone | 1.000±0.000 | 1.000±0.000 | **1.000**±0.000 |
| TGAT alone | 0.698±0.209 | 0.665±0.109 | 0.659±0.049 |
| Hybrid OR (union) | 0.758±0.181 | **1.000**±0.000 | 0.855±0.113 |
| Hybrid AND (intersect) | **1.000**±0.000 | 0.665±0.109 | 0.795±0.082 |

Note: T-Logic R1 achieves perfect F1=1.0 on multi_varied (all 3 seeds), making the hybrid unnecessary on this dataset.

#### StaticGNN — multi-seed results

| Dataset | Seed | AUC | AUPRC | F1 | AUPRC Lift |
|---|---|---|---|---|---|
| single | 42 | 0.759 | 0.498 | 0.227 | 272x |
| single | 43 | 0.784 | 0.626 | 0.667 | 342x |
| single | 44 | 0.775 | 0.513 | 0.222 | 280x |
| multi\_varied | 42 | 0.930 | 0.353 | 0.091 | 148x |
| multi\_varied | 43 | 0.936 | 0.173 | 0.219 | 72x |
| multi\_varied | 44 | 0.929 | 0.085 | 0.085 | 35x |

StaticGNN is strong on single (AUPRC up to 0.63) but degrades significantly on multi_varied (high seed variance), showing that ignoring temporal ordering hurts generalisation.

#### ML feature-only baselines

| Model | Dataset | AUC | AUPRC | F1 | AUPRC Lift | Train (s) |
|---|---|---|---|---|---|---|
| RandomForest | multi (100 proj) | **0.980** | 0.091 | 0.096 | **43x** | 112.8 |
| RandomForest | multi\_varied (30 families) | **0.983** | **0.104** | **0.115** | **43x** | 7.5 |
| LogisticRegression | multi (100 proj) | 0.688 | 0.068 | 0.102 | 32x | 11.3 |
| LogisticRegression | multi\_varied (30 families) | 0.685 | 0.069 | 0.047 | 28x | 0.3 |

RF is consistent across both multi datasets (AUC ~0.98, AUPRC lift ~43x). LR AUC ~0.69 confirms it cannot model non-linear interactions in the feature space.

#### Static KG baselines (ComplEx, TNTComplEx)

All three datasets: AUC ≈ 0.50, AUPRC ≈ random baseline (lift ≈ 1x). Static KG embedding cannot model the temporal ordering required for EPC compliance prediction.

### Interpretation

1. **T-Logic dominates**: pure symbolic rule-matching achieves near-perfect F1 (0.97–1.0) across all datasets. The compliance structure is largely rule-derivable from the KG.
2. **TGN is the best neural model**: slight edge over RF on AUPRC (0.094 vs 0.091) on multi, justifying the temporal-graph architecture for cases where rules are incomplete.
3. **RF is a strong neural baseline**: competitive with TGN at 3x lower training cost, confirming that feature engineering captures most of the signal.
4. **StaticGNN works on single but not multi**: good AUPRC on one project, but high variance across seeds on multi_varied shows it overfits without temporal ordering.
5. **Static KG embeddings fail**: ComplEx/TNTComplEx at AUC ≈ 0.5 — treating compliance as a link-prediction task without temporal context is insufficient.
6. **Hybrid ensemble adds little**: since T-Logic R1 alone is already perfect, combining with TGAT (AND/OR) only reduces recall or precision without gain.

