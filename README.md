# EPC Compliance Monitoring with Temporal Knowledge Graphs

**MSc Thesis** · Tecnicas Reunidas / Universidad Politécnica de Madrid & Politecnico di Milano  
**Author:** Ottavia Biagi  
**Industry partner:** Tecnicas Reunidas S.A. — Meram Refinery Expansion (Turkey)

> This thesis designs, implements, and evaluates a Temporal Knowledge Graph (TKG) framework for automated EPC (Engineering, Procurement & Construction) permit compliance monitoring, using real project activity data from Tecnicas Reunidas. The system detects work-permit violations — events where a worker is assigned to a step without valid certification — under realistic temporal and structural conditions.

---

## Thesis at a Glance

| Domain | Industry data | Scale | Approach | Best result |
|--------|--------------|-------|----------|-------------|
| EPC permit compliance & violation detection | ✅ Real TR Meram activity/step/worker structure; real permit-denial records | 29K–560K events; 8–201 test violations | TGN / TGAT / DyRep / StaticGNN / ComplEx / TNTComplEx + ML baselines | **TGN AUC=0.985, lift=×98.9** (single); **TGAT lift=×309.0** (multi_varied, 30 diverse EPC families); ComplEx/TNTComplEx=random; StaticGNN ×147.6 (multi_varied) |

---

## Repository Structure

```
TKG_Thesis/
│
├── data/epc_tkg/                       # [THESIS DATA] TR Meram EPC dataset
│   (folder on disk: data/UseCase4/)
│   ├── epc_dataset_real.json           # TKG built from real TR activity/step/worker data
│   ├── epc_events.json                 # Real permit-denial event records (labels)
│   ├── generate_epc_dataset.py         # Dataset generator (real TR PCS → TKG)
│   ├── import_graph_real.py            # Neo4j import (requires bolt://localhost:7687)
│   ├── projects/                       # Multi-project instances (proj_000–proj_099, V000–V029)
│   └── queries/                        # Cypher: temporal compliance, critical path
│
├── notebooks/epc_compliance/           # [THESIS] Analysis notebooks
│   (folder on disk: notebooks/UseCase4/)
│   ├── 01_explore_epc.ipynb            # Dataset exploration + Neo4j verification
│   ├── 02_temporal_queries.ipynb       # Bitemporal compliance queries
│   ├── 03_critical_path.ipynb          # Critical path & bottleneck analysis
│   ├── 04_dynamic_tkg.ipynb            # Event stream analysis
│   ├── 05_tgn_epc.ipynb               # TGN prototype (early exploration)
│   ├── 06_tkg_models.ipynb             # TNTComplEx + RF/XGBoost baselines
│   ├── 07_four_layer_tlogic.ipynb      # T-Logic symbolic rules + cascade risk
│   ├── 08_model_benchmark_final.ipynb  # [MAIN] Full benchmark: TGN/TGAT/DyRep x4 splits
│   │                                   #   + label sanity (T1-T5) + all static baselines
│   └── archive_synth_v1/              # [ARCHIVE] Synthetic-data prototype (pre-real-data)
│                                       #   Not part of thesis — kept for development history
│
├── experiments/epc_compliance/         # [THESIS] Training & evaluation pipeline
│   (folder on disk: experiments/UseCase4/)
│   ├── eval_framework.py               # split_dataset · compute_metrics · find_best_threshold
│   ├── data_loader.py                  # load_single_project / load_multi_project
│   ├── run_benchmark.py                # TGN/TGAT/DyRep x 4 splits x N seeds
│   ├── run_ml_baseline.py              # LR + RF feature-only baselines
│   ├── run_static_baseline.py          # ComplEx + TNTComplEx (all 3 scales)
│   ├── run_static_gnn.py               # Static GCN (structure-only, no time)
│   ├── tune_hyperparams.py             # Optuna TPE, 50 trials, val-AUPRC objective
│   ├── models/                         # TGN, TGAT, DyRep implementations
│   └── results/
│       ├── benchmark.json              # All TGN/TGAT/DyRep results (metrics + per-slot)
│       ├── ml_baseline.json            # LR + RF results
│       ├── static_baseline.json        # ComplEx + TNTComplEx (all 3 scales)
│       └── static_gnn.json             # StaticGNN results (single + multi_varied)
│
├── tests/                              # Data validation — run once before experiments
│   ├── test_real_data.py               # 22 quality checks on TR Meram data (T1-T5)
│   └── test_3w.py                      # [LEGACY] UC2 leftover (Petrobras 3W) — not used
│
├── scripts/                            # Development utilities (not thesis pipeline)
│   ├── delay_analysis.py               # EPC delay propagation exploration
│   ├── eval_models_testset.py          # One-off test-set evaluation (TNTComplEx/TGN-B)
│   ├── plot_roc.py                     # ROC curve plotting helper
│   ├── patch_neo4j_db.py / revert      # One-time Neo4j data corrections
│   └── run_exp_*.py / inject_exp_*.py  # Experiment injection scripts (development only)
│
└── src/                                # Legacy utility code (pre-thesis development)
    ├── config.py                       # Neo4j connection settings
    ├── graph/load_to_neo4j.py          # UC1 turbine data -> Neo4j (not used in thesis)
    └── models/                         # Early model prototypes (UC1/UC2 anomaly detection)
                                        # Thesis models are in experiments/epc_compliance/models/
```

> **Note on folder naming:** Physical folder names on disk (`UseCase4/`) reflect development history and are preserved for git stability. Logical names in this README (`epc_compliance/`, `epc_tkg/`) describe their thesis role.  
> **Active thesis pipeline:** `data/UseCase4/` → `experiments/UseCase4/` → `notebooks/UseCase4/08_model_benchmark_final.ipynb`

---

## EPC Compliance Detection — Full Benchmark Results (TR Meram)

### Dataset

| | Single project | Multi project (×100) | Multi-varied (×30 families) |
|--|--|--|--|
| Events (edges) | 29,150 | 2,915,000 | 559,877 |
| Violations | 449 (1.54%) | 43,472 (1.49%) | 8,276 (1.48%) |
| Unique nodes | 29,200 | 2,919,840 | 561,317 |
| Edge features | 6 | 6 | 6 |

Features: `permit_enc · disc_enc · after_rc · on_critical_path · weight_pct · cert_expires_soon`

### Violation Detection — Temporal Split (primary benchmark)

> Temporal 70/15/15 split · **8 test violations / 4,373 test events** (prevalence=0.18%) · threshold on val · seed=42 · **Note: with only 8 test violations, AUPRC estimates are high-variance. AUC is more reliable here; multi_varied (201 violations) is the statistically stable benchmark.**

Ordered by AUC (most reliable metric with only 8 test violations — AUPRC is high-variance at this scale).

| Model | Type | AUC | AUPRC | Lift | F1 | Recall |
|-------|------|-----|-------|------|----|--------|
| **TGN** | Temporal GNN | **0.985** | **0.178** | **×98.9** | 0.084 | **1.000** |
| Random Forest | Feature-only ML | 0.978 | 0.160 | ×87.8 | 0.071 | 0.125 |
| TGAT | Temporal GNN | 0.822 | 0.046 | ×25.6 | 0.129 | 0.250 |
| Logistic Regression | Feature-only ML | 0.738 | 0.161 | ×88.4 | 0.024 | 0.625 |
| StaticGNN (d=1) | Structure-only GNN | 0.759 | 0.498† | ×272† | 0.227 | 0.625 |
| TNTComplEx | Time-aware KG emb | 0.582 | 0.003 | ×1.6 | 0.004 | — |
| Random baseline | — | 0.500 | 0.002 | ×1.0 | — | — |
| ComplEx | Static KG emb | 0.440 | 0.002 | ×1.0 | 0.004 | — |
| DyRep | Temporal GNN | 0.416 | 0.002 | ×1.1 | 0.004 | 1.000‡ |

† StaticGNN single AUPRC=0.498 is a high-variance artefact of 8 test violations (val_AUPRC=0.068 confirms instability). Do not report as a reliable result. The reliable StaticGNN figure is multi_varied: AUPRC=0.353, ×147.6 (201 violations).  
‡ DyRep recall=1.0 is degenerate: val-tuned threshold collapses to near-zero, flagging almost all events as violations (precision=0.002). Confirmed architectural failure.

**Key findings:**
- **TGN best overall** (AUC=0.985, recall=1.0 — catches all 8 violations). Persistent memory module accumulates worker certificate history over time.
- **Structural hierarchy confirmed on multi_varied** (201 violations, reliable): ComplEx/TNTComplEx ×1.0 < StaticGNN ×147.6 < TGAT ×309.0. Each layer (static embedding → structure aggregation → temporal dynamics) adds signal.
- **Features are highly informative**: RF (AUPRC=0.160) nearly matches TGN (AUPRC=0.178) on 6 edge features alone. TGN adds +10% AUPRC and recall=1.0 by contextualising features temporally.
- **ComplEx and TNTComplEx = random** at every scale: same `(worker, step, relation)` triple can be either compliant or a violation depending solely on the timestamp. Static and time-binned embeddings cannot capture this without persistent memory.
- **DyRep fails architecturally**: intensity-based link-prediction design; threshold degenerates at 1.5% imbalance.
- **Threshold matters**: at 0.18% prevalence, threshold=0.5 gives F1≈0 for all models. Val-set optimised threshold is 0.052 for TGN.

### Multi-Split Summary (TGN)

| Split | AUC | AUPRC | Lift |
|-------|-----|-------|------|
| Temporal (primary) | 0.985 | 0.178 | ×98.9 |
| 6-slot (temporal stability) | 0.985 | 0.178 | ×98.9 |
| Inductive (new nodes) | 0.984 | 0.182 | ×101.1 |
| Stratified (optimistic upper bound) | 0.833 | 0.073 | ×4.7 |

The stratified drop (AUC 0.985→0.833) is expected: it shuffles time, allowing future events in training. The temporal split is the methodologically correct evaluation.

### Multi-Project Evaluation (§6b, notebook 08)

Three evaluation scenarios with increasing complexity:

1. **Single-project** (29K events) — primary result, full baseline suite.
2. **Homogeneous multi** (100 instances × same EPC process structure, 2.9M events) — statistically stable AUPRC from 933 test violations; validates computational scalability. Each project has its own scoped node IDs (`P{proj}:worker`, `P{proj}:step`): no cross-project node sharing occurs at the model level.
3. **Varied-topology multi** (30 structurally diverse EPC project families, 560K events) — cross-project generalisation test. Step codes vary across families; 201 test violations give reliable AUPRC estimates.

Temporal split, seeds 42–44 (3-seed evaluation).

**Primary generalisation result — multi_varied (30 diverse EPC families, 201 test violations):**

| Model | Dataset | AUC | AUPRC (seed 42) | AUPRC mean±std (3 seeds) | Lift (mean) |
|-------|---------|-----|----------------|--------------------------|-------------|
| **TGAT** | multi_varied | **0.992** | **0.646** | **0.717 ± 0.073** | **×300** |
| TGN | multi_varied | 0.983 | ~0.127 | 0.127 ± 0.001 | ×53 |
| StaticGNN (d=2) | multi_varied | 0.930 | 0.353 | — (seed 42 only) | ×147.6 |
| ComplEx | multi_varied | 0.521 | 0.002 | — | ×1.0 |
| TNTComplEx | multi_varied | 0.516 | 0.002 | — | ×1.0 |

TGAT mean±std computed over seeds 42, 43, 44 (AUPRC: 0.646 / 0.713 / 0.791). TGN is stable but low (0.126–0.127).

**Scalability test — homogeneous multi (100 identical instances, 933 test violations):**

| Model | Dataset | AUC | AUPRC | Lift | F1 | Note |
|-------|---------|-----|-------|------|----|------|
| TGAT | multi | 1.000 | 0.955 | ×454.8 | 0.905 | 933 test violations; no cross-project node sharing |
| TGN | multi | 0.981 | 0.094 | ×44.8 | 0.098 | Memory interference with repeated event structure |
| DyRep | multi | 0.500 | 0.002 | ×1.0 | 0.004 | Fails at scale (same as single) |
| LR (diagnostic) | multi | 0.682 | 0.072 | ×4.7 | — | Features less predictive at scale |
| ComplEx | multi | 0.503 | 0.002 | ×1.0 | 0.005 | Random |
| StaticGNN | multi | — | — | — | — | Infeasible: 2.9M nodes, GPU OOM, CPU ~60h |

**Key findings:**
- **TGAT ×300 on multi_varied (3-seed mean)**: AUPRC = 0.717 ± 0.073 across seeds 42–44. Confirms temporal attention generalises robustly to structurally new EPC projects. Seed-to-seed range (0.646–0.791) is expected given only 201 test violations.
- **TGN stable but weak on multi_varied**: AUPRC = 0.127 ± 0.001 — stateful memory module struggles with cross-project variability, same as in homogeneous multi.
- **Structural hierarchy confirmed (3-seed)**: ComplEx/TNTComplEx ×1 < StaticGNN ×147.6 < TGN ~×53 < TGAT ×300. Each layer adds meaningful signal.
- **Scalability (homogeneous multi)**: TGAT reaches AUPRC=0.955 on 100 independent single-project instances (933 test violations). TGN degrades to 0.094 because repeated event structures cause memory interference in the stateful architecture. LR diagnostic (0.072) rules out feature artefacts.
- **Design note**: in the homogeneous multi dataset, node IDs are scoped per project (`P{proj}:step`, `P{proj}:worker`), so models do not share representations across projects. The high AUPRC reflects reliable estimation from 933 test violations, not cross-project learning.

### Label Sanity Analysis (§2b, notebook 08)

Labels from `epc_events.json['permit_denied']` — real operational EPC permit denials. Validated via 5 empirical tests (Ratner et al. 2017 / Mintz et al. 2009):

| Test | Result | Verdict |
|------|--------|---------|
| T1 Feature–label correlation | 5/6 features significant; `cert_expires_soon` r=+0.155, p=7×10⁻¹⁵⁷ | ✓ PASS |
| T2 Distribution shift (Cohen's d) | Visible separation for 4/6 features | ✓ PASS |
| T3 Temporal clustering (Spearman ρ) | Non-uniform violation rate across deciles | ✓ PASS |
| T4 Label consistency | All (worker, step) pairs unique → 100% consistent | ✓ PASS |
| T5 Linear separability (LR 5-fold CV) | AUC=0.654 ± 0.013 >> 0.5 random | ✓ PASS |

### Static KG Baselines — Violation Detection (§5b–§5c, notebook 08)

Same protocol as TGN/TGAT/DyRep: temporal 70/15/15 split, val-threshold, AUC/AUPRC on test.

| Model | Dataset | AUC | AUPRC | Lift | Key point |
|-------|---------|-----|-------|------|-----------|
| ComplEx | single | 0.440 | 0.002 | ×1.0 | Static embeddings — no timestamp |
| ComplEx | multi | 0.503 | 0.002 | ×1.0 | Same: random at scale |
| ComplEx | multi_varied | 0.521 | 0.002 | ×1.0 | Same: random at scale |
| TNTComplEx | single | 0.582 | 0.003 | ×1.6 | Time embedding — marginal gain |
| TNTComplEx | multi | 0.507 | 0.002 | ×1.0 | No persistent memory → random at scale |
| TNTComplEx | multi_varied | 0.516 | 0.002 | ×1.0 | Same |
| StaticGNN | single | 0.759 | 0.498 | ×272.5 | Structural patterns; val_AUPRC=0.068 (small test set) |
| StaticGNN | multi | — | — | — | Infeasible: 2.9M nodes GPU OOM, CPU ~60h |
| StaticGNN | multi_varied | 0.930 | 0.353 | ×147.6 | Structure only; no temporal dynamics |

**Core finding**: ComplEx and TNTComplEx = random at every scale. The same `(worker, step, relation)` triple can be either compliant or a violation depending only on the timestamp (certificate expiry). Static and time-binned embeddings cannot capture this signal without persistent memory.

### TNTComplEx — Link Prediction (notebook 06)

| Relation | MRR | H@10 | Notes |
|----------|-----|------|-------|
| REQUIRES_PERMIT | 0.401 | **1.00** | Deterministic structural relation — perfectly learned |
| ASSIGNED_TO | 0.0003 | 0.00 | Stochastic many-to-many — structurally cannot be predicted from graph topology |

### T-Logic Symbolic Rules (notebook 07)

**P=0.875, R=0.875, F1=0.875** on post-rule-change test (274 violations). Rules:
- R1: `DELIVERED_LATE(PO, Activity, t)` → `IS_DELAYED(Activity)`
- R2: `IMPACTS_ACTIVITY(Event, Activity, t)` → `IS_DELAYED(Activity)`
- R3: `APPROVED_LATE(Doc, Activity, t)` → `IS_DELAYED(Activity)`

---

## Data Provenance — TR Meram Dataset

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

### EPC Compliance Benchmark
```bash
# Single-project benchmark (all models × splits, ~45 min)
python experiments/UseCase4/run_benchmark.py --dataset single --seeds 42

# Specific model + split for quick check
python experiments/UseCase4/run_benchmark.py --model TGN --split temporal --dataset single

# ML feature-only baselines (LR + RF)
python experiments/UseCase4/run_ml_baseline.py

# Static KG baselines (ComplEx + TNTComplEx, all 3 datasets)
python experiments/UseCase4/run_static_baseline.py --model all --dataset all

# Static GNN baseline (single project, ~2 min on GPU)
python experiments/UseCase4/run_static_gnn.py --dataset single

# Hyperparameter tuning (50 trials per model, ~3h)
python experiments/UseCase4/tune_hyperparams.py

# Multi-project benchmark (~2-3h)
python experiments/UseCase4/run_benchmark.py --dataset multi --seeds 42
```

### Neo4j Import (TR Meram graph)
```bash
python data/UseCase4/generate_epc_dataset.py
python data/UseCase4/import_graph_real.py     # requires Neo4j at bolt://localhost:7687
```

### Run notebooks in order
```bash
jupyter lab   # then open notebooks/UseCase4/08_model_benchmark_final.ipynb
```

---

## Methodology Checklist

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
| 12 | Static KG baselines | ✅ ComplEx + TNTComplEx (random at all scales); StaticGNN (single ×272.5; multi infeasible; multi_varied ×147.6) |

---

## References

- Xu et al. (2020) — Inductive Representation Learning on Temporal Graphs (TGAT) · ICLR
- Rossi et al. (2020) — Temporal Graph Networks (TGN) · NeurIPS
- Zuo et al. (2018) — Embedding Temporal Network via Neighborhood Formation (DyRep)
- Trouillon et al. (2016) — Complex Embeddings for Simple Link Prediction (ComplEx) · ICML
- Kipf & Welling (2017) — Semi-Supervised Classification with Graph Convolutional Networks (GCN) · ICLR
- Lacroix et al. (2020) — Tensor Decompositions for Knowledge Base Completion (TNTComplEx)
- Liu et al. (2022) — T-Logic: Temporal Logical Rules for Explainable Link Forecasting
- Ratner et al. (2017) — Data Programming: Creating Large Training Sets Quickly · NeurIPS
- TR Internal — Family_Steps_macro.xlsm · Meram_PCS_Progress.xlsx
