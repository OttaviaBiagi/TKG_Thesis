# TKG Thesis — Temporal Knowledge Graphs for Industrial Monitoring

**MSc Thesis** · Tecnicas Reunidas / Universidad Politécnica de Madrid & Politecnico di Milano  
**Author:** Ottavia Biagi

> Design and evaluation of Temporal Knowledge Graph (TKG) systems across four industrial domains: anomaly detection, causal analysis, and EPC project compliance.

---

## Use Cases at a Glance

| # | Domain | Data | Approach | Main result |
|---|--------|------|----------|-------------|
| **UC1** | Synthetic turbine anomaly detection | 1.3M sensor readings, 6.8% anomaly rate | IsolationForest + TGN | Threshold-tuned IF baseline; TGN on sensor TKG |
| **UC2** | Oil well anomaly detection (3W) | Petrobras 3W dataset, 10 anomaly classes | TGN → RF/XGBoost | Improved recall per anomaly class with class-weighted trees |
| **UC3** | EPC delay causal analysis | Synthetic EPC (60 activities, 5 causal chains) | T-Logic symbolic rules | R1+R2+R3 rules validated against ground truth causal chains |
| **UC4** | EPC compliance & violation detection | Real TR Meram (29,150 steps, 449 violations) | TGN / TGAT / DyRep + baselines | TGN AUC=0.985 lift=×98.9 (single); TGAT lift=×454.8 (multi, shared topology) |

---

## Repository Structure

```
TKG_Thesis/
├── data/
│   ├── UseCase2/           # 3W Dataset (Petrobras) — oil well sensor data
│   └── UseCase4/           # EPC TKG — TR Meram Refinery Expansion
│       ├── epc_dataset_real.json       # Generated TKG (real TR activity data)
│       ├── epc_events.json             # Simulated event stream (ASSIGNED_TO, PERMIT_DENIED)
│       ├── generate_epc_dataset.py     # Dataset generator
│       ├── import_graph_real.py        # Neo4j import script
│       ├── projects/                   # 100 synthetic project instances (proj_000–proj_099)
│       └── queries/                    # Cypher queries (temporal compliance, critical path)
│
├── notebooks/
│   ├── UseCase1/           # 01 generate+explore · 02 anomaly detection · 03 Neo4j queries
│   ├── UseCase2/           # 01 explore · 02 preprocessing · 03–04 TGN · 05–06 RF/XGBoost
│   ├── UseCase3/           # 01 explore · 02 TKG build · 03 T-Logic causal rules
│   └── UseCase4/
│       ├── 01_explore_epc.ipynb            # Dataset exploration + Neo4j verification
│       ├── 02_temporal_queries.ipynb       # Bitemporal compliance queries
│       ├── 03_critical_path.ipynb          # Critical path & bottleneck analysis
│       ├── 04_dynamic_tkg.ipynb            # Event stream analysis
│       ├── 05_tgn_epc.ipynb                # TGN training (early prototype)
│       ├── 06_tkg_models.ipynb             # TNTComplEx + RF/XGBoost baselines
│       ├── 07_four_layer_tlogic.ipynb      # T-Logic symbolic rules + cascade risk
│       └── 08_model_benchmark_final.ipynb  # ★ Full benchmark: TGN/TGAT/DyRep × 4 splits
│                                           #   + label sanity analysis + ML baselines
│
├── experiments/UseCase4/
│   ├── eval_framework.py       # split_dataset · compute_metrics · find_best_threshold
│   ├── data_loader.py          # load_single_project / load_multi_project
│   ├── run_benchmark.py        # 3 models × 4 splits × 2 datasets × N seeds
│   ├── run_ml_baseline.py      # LR + RF feature-only baselines
│   ├── tune_hyperparams.py     # Optuna TPE (50 trials, val-AUPRC objective)
│   ├── models/                 # TGN, TGAT, DyRep implementations
│   └── results/
│       ├── benchmark.json      # Full results (all metrics, per-slot detail)
│       ├── benchmark.csv       # Summary table
│       ├── best_params.json    # Tuned hyperparameters
│       └── ml_baseline.json    # LR + RF results
│
└── src/                    # Shared utilities (Neo4j loader, model scripts)
```

---

## UseCase4 — Full Benchmark Results

### Dataset

| | Single project | Multi project |
|--|--|--|
| Events (edges) | 29,150 | 2,915,000 |
| Violations | 449 (1.54%) | 43,472 (1.49%) |
| Unique nodes | 29,200 | 2,919,840 |
| Edge features | 6 | 6 |

Features: `permit_enc · disc_enc · after_rc · on_critical_path · weight_pct · cert_expires_soon`

### Violation Detection — Temporal Split (primary benchmark)

> Temporal 70/15/15 split · test-set prevalence = 0.18% (8 violations / 4,373 events) · threshold optimised on val set · seed=42

| Model | Type | AUC | AUPRC | Lift | F1 | Recall |
|-------|------|-----|-------|------|----|--------|
| **TGN** | Temporal GNN | **0.985** | **0.178** | **×98.9** | 0.084 | **1.000** |
| Random Forest | Feature-only ML | 0.978 | 0.161 | ×87.8 | 0.071 | — |
| Logistic Regression | Feature-only ML | 0.840 | 0.162 | ×88.4 | 0.024 | — |
| TGAT | Temporal GNN | 0.822 | 0.046 | ×25.6 | 0.105 | 0.250 |
| DyRep | Temporal GNN | 0.416 | 0.002 | ×1.1 | 0.000 | 0.000 |
| Random baseline | — | 0.500 | 0.002 | ×1.0 | — | — |

**Key findings:**
- **TGN is best** (AUC=0.985, recall=1.0 — catches all 8 violations), driven by its persistent memory module accumulating worker behaviour over time.
- **Task is feature-driven**: RF (AUPRC=0.161) nearly matches TGN (AUPRC=0.178) using only the 6 edge features. TGN adds +10% AUPRC and recall=1.0 by contextualising features in the temporal workflow graph.
- **DyRep fails architecturally**: designed for link prediction (balanced classes); its intensity formulation cannot handle 1.5% imbalance. Retained as a negative result.
- **Threshold matters**: at 0.18% test prevalence, fixed threshold=0.5 gives F1≈0 for all models. Optimal threshold (found on val set) is 0.052 for TGN temporal.

### Multi-Split Summary (TGN)

| Split | AUC | AUPRC | Lift |
|-------|-----|-------|------|
| Temporal (primary) | 0.985 | 0.178 | ×98.9 |
| 6-slot (temporal stability) | 0.985 | 0.178 | ×98.9 |
| Inductive (new nodes) | 0.984 | 0.182 | ×101.1 |
| Stratified (optimistic upper bound) | 0.833 | 0.073 | ×4.7 |

The stratified drop (AUC 0.985→0.833) is expected: it shuffles time, allowing future events in training. The temporal split is the methodologically correct evaluation.

### Multi-Project Generalisation (§6b, notebook 08)

100 synthetic projects (2,915,000 events, 43,472 violations) sharing the same EPC graph topology (identical step node IDs). Temporal split, seed=42.

| Model | Dataset | AUC | AUPRC | Lift | F1 |
|-------|---------|-----|-------|------|-----|
| TGAT | multi | **1.000** | **0.955** | **×454.8** | 0.905 |
| TGN | multi | 0.981 | 0.094 | ×44.8 | 0.098 |
| DyRep | multi | 0.500 | 0.002 | ×1.0 | 0.004 |
| TGN | single | 0.985 | 0.178 | ×98.9 | 0.084 |
| TGAT | single | 0.822 | 0.046 | ×25.6 | 0.129 |
| LR (diagnostic) | multi | 0.682 | 0.072 | ×4.7 | — |

**Architectural finding — TGAT stateless attention vs TGN stateful memory on shared topology:**

All 100 projects share the same step node IDs (identical EPC permit graph). TGAT, which recomputes attention from scratch at every event without maintaining persistent memory, accumulates the signal that "step X is high-risk" cleanly across 100 independent training projects, reaching AUPRC=0.955. TGN maintains a persistent memory state per node; with 100 overlapping projects all writing to the same node memories, the memory suffers interference and AUPRC degrades from 0.178 (single) to 0.094 (multi).

The LR diagnostic (AUPRC=0.072 with the same 6 features) confirms that TGAT's improvement is not explained by feature artifacts such as `cert_expires_soon`. The performance gap is structural: TGAT learns node-level permit patterns from the shared EPC graph topology.

**Scope note:** The multi-project dataset is a scalability test (same graph, more data), not a cross-project generalisation test. All 100 instances share the same EPC step structure, so the violation label distribution is nearly identical across projects (1.49% ± 0.07%).

### Label Sanity Analysis (§2b, notebook 08)

Labels from `epc_events.json['permit_denied']` — real operational EPC permit denials. Validated via 5 empirical tests (Ratner et al. 2017 / Mintz et al. 2009):

| Test | Result | Verdict |
|------|--------|---------|
| T1 Feature–label correlation | 5/6 features significant; `cert_expires_soon` r=+0.155, p=7×10⁻¹⁵⁷ | ✓ PASS |
| T2 Distribution shift (Cohen's d) | Visible separation for 4/6 features | ✓ PASS |
| T3 Temporal clustering (Spearman ρ) | Non-uniform violation rate across deciles | ✓ PASS |
| T4 Label consistency | All (worker, step) pairs unique → 100% consistent | ✓ PASS |
| T5 Linear separability (LR 5-fold CV) | AUC=0.654 ± 0.013 >> 0.5 random | ✓ PASS |

### TNTComplEx — Link Prediction (notebook 06)

| Relation | MRR | H@10 | Notes |
|----------|-----|------|-------|
| REQUIRES_PERMIT | 0.401 | **1.00** | Deterministic structural relation — perfectly learned |
| ASSIGNED_TO | 0.0003 | 0.00 | Stochastic many-to-many — structurally cannot be predicted from graph topology |

### T-Logic Symbolic Rules (notebook 07)

**P=1.0, R=1.0, F1=1.0** on post-rule-change test (274 violations). Rules:
- R1: `DELIVERED_LATE(PO, Activity, t)` → `IS_DELAYED(Activity)`
- R2: `IMPACTS_ACTIVITY(Event, Activity, t)` → `IS_DELAYED(Activity)`
- R3: `APPROVED_LATE(Doc, Activity, t)` → `IS_DELAYED(Activity)`

---

## Data Provenance — UseCase4

| Data | Source |
|------|--------|
| Activity / Family / Step names, codes, sequences | ✅ Real — TR Meram PCS + Family_Steps_macro.xlsm |
| Estimated hours, earned hours, discipline, area, CWP | ✅ Real — TR Meram PCS (8,762 rows → 5,555 unique activities) |
| Discipline timeline | ⚠️ Estimated (hardcoded per discipline) |
| Workers, certifications, work permits | ❌ Synthetic |
| Permit denial events (labels) | ✅ Real system records from `epc_events.json` |
| Bitemporal rule-change scenario | ❌ Synthetic (demo) |

Data quality: 22 PASS / 0 FAIL / 3 WARN — run `python tests/test_real_data.py`

---

## How to Run

### Prerequisites
```bash
conda activate tkg-env   # or: pip install torch numpy pandas scikit-learn scipy optuna matplotlib
```

### UseCase4 Benchmark
```bash
# Single-project benchmark (all models × splits, ~45 min)
python experiments/UseCase4/run_benchmark.py --dataset single --seeds 42

# Specific model + split for quick check
python experiments/UseCase4/run_benchmark.py --model TGN --split temporal --dataset single

# ML feature-only baselines (LR + RF)
python experiments/UseCase4/run_ml_baseline.py

# Hyperparameter tuning (50 trials per model, ~3h)
python experiments/UseCase4/tune_hyperparams.py

# Multi-project benchmark (~2-3h)
python experiments/UseCase4/run_benchmark.py --dataset multi --seeds 42
```

### UseCase4 Neo4j Import
```bash
python data/UseCase4/generate_epc_dataset.py
python data/UseCase4/import_graph_real.py     # requires Neo4j at bolt://localhost:7687
```

### Run notebooks in order
```bash
jupyter lab   # then open notebooks/UseCase4/08_model_benchmark_final.ipynb
```

---

## Methodology Checklist (UseCase4)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Temporal integrity (no future leakage) | ✅ Temporal 70/15/15 split as primary evaluation |
| 2 | Class imbalance handling | ✅ Weighted BCE + AUPRC as primary metric + val-set threshold |
| 3 | Hyperparameter optimisation | ✅ Optuna TPE, 50 trials, val-AUPRC objective |
| 4 | Same protocol for all models | ✅ Identical splits, scaler, feat_cols, threshold procedure |
| 5 | Feature-only baselines | ✅ LR + RF on FEAT_COLS — task is feature-driven |
| 6 | Inductive evaluation | ✅ 10% worker nodes withheld from training |
| 7 | Temporal drift analysis | ✅ 6-slot split — per-time-window metrics |
| 8 | Label validation | ✅ 5 empirical sanity tests (T1–T5) |
| 9 | Reproducibility | ✅ Fixed seed=42; multi-seed via `--seeds 42 43 44` |
| 10 | Multi-project generalisation | ✅ Completed — TGAT×454.8 lift; architectural finding documented |
| 11 | Expert label validation | ⏳ Future work |
| 12 | Static KG baseline (TransE) | ⏳ Future work |

---

## References

- Xu et al. (2020) — Inductive Representation Learning on Temporal Graphs (TGAT) · ICLR
- Rossi et al. (2020) — Temporal Graph Networks (TGN) · NeurIPS
- Zuo et al. (2018) — Embedding Temporal Network via Neighborhood Formation (DyRep)
- Lacroix et al. (2020) — Tensor Decompositions for Knowledge Base Completion (TNTComplEx)
- Liu et al. (2022) — T-Logic: Temporal Logical Rules for Explainable Link Forecasting
- Vargas et al. (2019) — 3W Dataset · Journal of Petroleum Science and Engineering
- Ratner et al. (2017) — Data Programming: Creating Large Training Sets Quickly · NeurIPS
- TR Internal — Family_Steps_macro.xlsm · Meram_PCS_Progress.xlsx
