# Thesis Navigation Guide

**"EPC Compliance Monitoring with Temporal Knowledge Graphs"**  
Ottavia Biagi — MSc Thesis, Tecnicas Reunidas / UPM & PoliMi

This guide maps each thesis chapter to the relevant files and notebooks.

---

## Chapter 1 — Introduction

*Motivation: EPC projects are permit-constrained; manual compliance checking is expensive and error-prone. Can a TKG learn to predict violations before they happen?*

No code — background reading only.

---

## Chapter 2 — Background & Related Work

| Topic | Reference |
|-------|-----------|
| Temporal Graph Networks | Rossi et al. 2020 (TGN) |
| Temporal Graph Attention | Xu et al. 2020 (TGAT) |
| Temporal link prediction | Zuo et al. 2018 (DyRep) |
| Static KG embeddings | Trouillon et al. 2016 (ComplEx), Lacroix et al. 2020 (TNTComplEx) |
| Graph neural networks | Kipf & Welling 2017 (GCN) |
| T-Logic symbolic rules | Liu et al. 2022 |

---

## Chapter 3 — Dataset & TKG Construction

### 3a. Data sources and provenance
- [README.md § Data Provenance](README.md#data-provenance--tr-meram-dataset) — what is real vs synthetic
- `data/UseCase4/epc_dataset_real.json` — TKG built from TR Meram PCS data
- `data/UseCase4/epc_events.json` — real permit-denial records (labels)

### 3b. Dataset exploration
- [notebooks/UseCase4/01_explore_epc.ipynb](notebooks/UseCase4/01_explore_epc.ipynb) — entity stats, degree distribution, label exploration
- [notebooks/UseCase4/02_temporal_queries.ipynb](notebooks/UseCase4/02_temporal_queries.ipynb) — bitemporal Cypher queries

### 3c. Critical path & bottleneck analysis
- [notebooks/UseCase4/03_critical_path.ipynb](notebooks/UseCase4/03_critical_path.ipynb)

### 3d. Label sanity validation (T1–T5)
- [notebooks/UseCase4/08_model_benchmark_final.ipynb](notebooks/UseCase4/08_model_benchmark_final.ipynb) **§2b** — 5 empirical tests confirming permit-denial labels reflect real compliance failures
- Script: `tests/test_real_data.py` — 22 automated data quality checks (`python tests/test_real_data.py`)

### 3e. Multi-project dataset construction
- `data/UseCase4/projects/` — 100 homogeneous + 30 varied-topology project instances
- [notebooks/UseCase4/04_event_stream_analysis.ipynb](notebooks/UseCase4/04_event_stream_analysis.ipynb) — event stream characterisation

---

## Chapter 4 — Model Architecture

### 4a. Temporal GNN models (TGN / TGAT / DyRep)
- Implementations: [`experiments/UseCase4/models/tgn.py`](experiments/UseCase4/models/tgn.py), [`tgat.py`](experiments/UseCase4/models/tgat.py), [`dyrep.py`](experiments/UseCase4/models/dyrep.py)
- Training loop: [`experiments/UseCase4/run_benchmark.py`](experiments/UseCase4/run_benchmark.py)
- Evaluation framework: [`experiments/UseCase4/eval_framework.py`](experiments/UseCase4/eval_framework.py) — temporal split, threshold tuning, AUPRC/AUC/F1

### 4b. Static baselines
- ComplEx + TNTComplEx: [`experiments/UseCase4/run_static_baseline.py`](experiments/UseCase4/run_static_baseline.py)
- StaticGNN (GCN + MLP): [`experiments/UseCase4/run_static_gnn.py`](experiments/UseCase4/run_static_gnn.py)
- ML feature-only (LR + RF): [`experiments/UseCase4/run_ml_baseline.py`](experiments/UseCase4/run_ml_baseline.py)

### 4c. Hyperparameter tuning
- [`experiments/UseCase4/tune_hyperparams.py`](experiments/UseCase4/tune_hyperparams.py) — Optuna TPE, 50 trials, val-AUPRC objective
- Best params: `experiments/UseCase4/results/best_params.json`

### 4d. T-Logic symbolic rules
- [notebooks/UseCase4/07_four_layer_tlogic.ipynb](notebooks/UseCase4/07_four_layer_tlogic.ipynb) — R1/R2/R3 rules, P=R=F1=0.875

---

## Chapter 5 — Experiments & Results

### 5a. Single-project benchmark (primary result)
- **Notebook:** [08_model_benchmark_final.ipynb §4–§5](notebooks/UseCase4/08_model_benchmark_final.ipynb)
- **Results file:** `experiments/UseCase4/results/benchmark.json`
- Key: TGN AUC=0.985, lift=×98.9; TGAT AUC=0.822, lift=×25.6

### 5b. Static KG baselines
- **Notebook:** [08_model_benchmark_final.ipynb §5b–§5c](notebooks/UseCase4/08_model_benchmark_final.ipynb)
- **Results:** `experiments/UseCase4/results/static_baseline.json`, `static_gnn.json`
- Key: ComplEx/TNTComplEx = random at all scales; StaticGNN ×272 (single, unstable), ×147.6 (multi_varied)

### 5c. Multi-project generalisation (multi_varied — 30 diverse EPC families)
- **Notebook:** [08_model_benchmark_final.ipynb §6c](notebooks/UseCase4/08_model_benchmark_final.ipynb)
- **Results:** `experiments/UseCase4/results/benchmark.json` (dataset=multi_varied)
- Key: TGAT lift=×309.0; structural hierarchy ComplEx×1 < StaticGNN×147.6 < TGAT×309

### 5d. Scalability test (homogeneous multi — 100 identical instances)
- **Notebook:** [08_model_benchmark_final.ipynb §6b](notebooks/UseCase4/08_model_benchmark_final.ipynb)
- Key: TGAT AUPRC=0.955 (933 test violations); TGN degrades due to memory interference

### 5e. Ablation & robustness
- Multi-split stability: 6-slot temporal drift analysis (§6a, notebook 08)
- Inductive generalisation: new-worker nodes withheld (§6, notebook 08)
- Multi-seed: `experiments/UseCase4/results/benchmark.json` (seeds 42–44)

---

## Chapter 6 — Conclusions

Summary of structural hierarchy, architectural insight, and limitations:
- [README.md § Key findings](README.md#violation-detection--temporal-split-primary-benchmark)
- [08_model_benchmark_final.ipynb §8](notebooks/UseCase4/08_model_benchmark_final.ipynb)

---

## Quick-start: reproduce the main result

```bash
conda activate tkg-env
cd TKG_Thesis

# 1. Validate data quality
python tests/test_real_data.py

# 2. Run single-project benchmark (TGN/TGAT/DyRep, ~45 min GPU)
python experiments/UseCase4/run_benchmark.py --dataset single --seeds 42

# 3. Run static baselines
python experiments/UseCase4/run_ml_baseline.py
python experiments/UseCase4/run_static_baseline.py --model all --dataset all
python experiments/UseCase4/run_static_gnn.py --dataset single

# 4. Open results notebook
jupyter lab notebooks/UseCase4/08_model_benchmark_final.ipynb
```

---

## Results files at a glance

| File | Contents |
|------|----------|
| `experiments/UseCase4/results/benchmark.json` | TGN / TGAT / DyRep — all splits × datasets × seeds |
| `experiments/UseCase4/results/ml_baseline.json` | LR + RF feature-only results |
| `experiments/UseCase4/results/static_baseline.json` | ComplEx + TNTComplEx (all 3 scales) |
| `experiments/UseCase4/results/static_gnn.json` | StaticGNN (single + multi_varied) |
| `experiments/UseCase4/results/best_params.json` | Optuna-tuned hyperparameters |
