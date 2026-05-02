# TKG_Thesis
**Temporal Knowledge Graph Systems for Industrial Monitoring, Prediction, and Decision Support**

MSc Thesis — Tecnicas Reunidas / Universidad Politécnica de Madrid & Politecnico di Milano
Author: Ottavia Biagi

---

## Objective
Design and implement Temporal Knowledge Graph (TKG) systems across four industrial domains:
1. **UseCase1** — Anomaly detection on synthetic turbine sensor data (baseline + TGN)
2. **UseCase2** — Anomaly detection on real oil well sensor data (3W Dataset, Petrobras)
3. **UseCase3** — Causal analysis of EPC project delays (T-Logic symbolic rules)
4. **UseCase4** — EPC project compliance and scheduling (bitemporal TKG + full ML pipeline)

---

## Repository Structure

```
TKG_Thesis/
├── data/
│   ├── UseCase2/           # 3W Dataset (Petrobras) — oil well sensor data
│   └── UseCase4/           # EPC TKG — TR Refinery Expansion Project
│       ├── epc_dataset_real.json        # Generated TKG dataset
│       ├── generate_epc_dataset.py      # Dataset generator (real TR Family Steps)
│       ├── import_graph_real.py         # Neo4j import script
│       └── queries/
│           ├── temporal_queries.cypher  # Bitemporal compliance queries
│           └── critical_path.cypher     # Critical path & bottleneck queries
│
├── notebooks/
│   ├── UseCase1/
│   │   ├── 01_generate_explore.ipynb    # Synthetic turbine data generation + EDA
│   │   ├── 02_anomaly_detection.ipynb   # IsolationForest + TGN on sensor TKG
│   │   └── 03_neo4j_queries.ipynb       # TKG import + 4 monitoring queries
│   ├── UseCase2/
│   │   ├── 01_explore_3w.ipynb          # 3W dataset exploration
│   │   ├── 02_preprocessing.ipynb       # Feature engineering
│   │   ├── 03_load_neo4j.ipynb          # Load 3W TKG into Neo4j
│   │   ├── 04_tgn_training.ipynb        # TGN model training (AUC-ROC 0.61)
│   │   ├── 05_results_analysis.ipynb    # Results analysis
│   │   └── 06_improved_models.ipynb     # RF/XGBoost with class imbalance handling
│   ├── UseCase3/
│   │   ├── 01_explore_epc.ipynb         # EPC delay dataset exploration + Gantt
│   │   ├── 02_tkg_build.ipynb           # TKG quadruples + Neo4j import
│   │   └── 03_tlogic_causal.ipynb       # T-Logic causal rules vs ground truth chains
│   └── UseCase4/
│       ├── 01_explore_epc.ipynb         # EPC dataset exploration + Neo4j verification
│       ├── 02_temporal_queries.ipynb    # Bitemporal compliance queries
│       ├── 03_critical_path.ipynb       # Critical path & bottleneck analysis
│       ├── 04_dynamic_tkg.ipynb         # Event stream analysis (ASSIGNED_TO, COMPLETED, PERMIT_DENIED)
│       ├── 05_tgn_epc.ipynb             # TGN training on EPC events (delay + violation prediction)
│       ├── 06_tkg_models.ipynb          # TNTComplEx vs TGN cert-aware embedding models
│       └── 07_four_layer_tlogic.ipynb   # Four-layer T-Logic symbolic rule mining + cascade risk
│
├── experiments/
│   ├── UseCase4/                        # Plots and outputs from UseCase4 notebooks
│   └── ...
│
└── src/                                 # Shared utilities
    ├── graph/
    │   └── load_to_neo4j.py             # Neo4j loader (UseCase1)
    └── models/
        ├── IsolationForest-anomaly_detection.py
        └── TGN-anomaly_detection.py
```

---

## Use Cases

### UseCase1 — Synthetic Turbine Anomaly Detection
- **Data:** Synthetic turbine (5 sensors, 30 days, ~1.3M readings, 6.8% anomaly rate)
- **Anomalies:** Spike (VIB_001 day 7), gradual degradation (TEMP_001 days 15-25), cyclic (PRES_001 day 20)
- **Approach:** IsolationForest (trained on normal-only + F1 threshold tuning) + TGN from scratch
- **TKG schema:** Component → HAS_SENSOR → Sensor → MADE_OBSERVATION → Observation → DETECTED_ANOMALY → AnomalyEvent
- **Neo4j:** 4 monitoring Cypher queries (anomaly window, causal chain, degradation trend, predictive alert)

### UseCase2 — Anomaly Detection (3W Dataset)
- **Data:** 3W Dataset (Petrobras, Vargas et al. 2019) — 2,228 oil well instances, 8 sensors, 10 anomaly classes
- **Challenge:** Severe class imbalance (normal vs rarest anomaly ~42:1)
- **Approach:** TGN (AUC-ROC 0.61) + improved RF/XGBoost with `class_weight='balanced'`, threshold tuning, per-class recall analysis
- **Key fix:** Tree-based models load directly from parquet (no Neo4j dependency), contamination estimated from data
- **Neo4j:** 9 wells, 8 sensors, temporal observations

### UseCase3 — EPC Causal Delay Analysis
- **Data:** Synthetic EPC project (60 activities, 25 events, 18 POs, 30 documents) + 5 verified causal chains
- **Approach:** T-Logic symbolic rules on TKG quadruples, validated against causal ground truth
- **Rules:**
  - R1: `DELIVERED_LATE(PO, Activity, t)` → `IS_DELAYED(Activity)` — late procurement causes delay
  - R2: `IMPACTS_ACTIVITY(Event, Activity, t) ∧ type∈{NCR,ChangeOrder}` → `IS_DELAYED(Activity)`
  - R3: `APPROVED_LATE(Doc, Activity, t)` → `IS_DELAYED(Activity)` — document blocking
- **Validation:** Precision / Recall / F1 per rule vs 5 ground truth causal chains (CC-01..CC-05)
- **TKG quadruples:** BELONGS_TO_WP, IS_DELAYED, IMPACTS_ACTIVITY, LINKED_TO_ACTIVITY, DELIVERED_LATE, APPROVED_LATE

### UseCase4 — EPC Compliance, Scheduling & Dynamic TKG
- **Data:** Real TR Meram project data (5,555 activities, 29,150 steps, 17 disciplines) + synthetic HSE layer (workers, certifications, work permits)
- **Source:** Family_Steps_macro.xlsm + Meram PCS Excel (8,762 rows deduplicated to 5,555 unique activities)
- **Approach:** Bitemporal TKG (valid_time + transaction_time) + simulated dynamic event stream
- **Results:**
  - Bitemporal compliance: 8 workers qualified before rule change → 3 after (5 lost compliance)
  - Critical path: 18 steps (ME.CT — Cooling Tower Erection)
  - Main bottleneck: CI discipline (563 steps), BU.BR.AR blocks 57,341 downstream steps
  - Dynamic events: 29,150 ASSIGNED_TO, 449 PERMIT_DENIED (1.5% violation rate), delay cascade propagation
- **Neo4j:** 1 Project, 5,555 Activities, 29,150 Steps, 8 WorkPermits, 33 Certifications, 50 Workers
- **ML Models (notebooks 05–07):**

**TNTComplEx — Link Prediction (notebook 06, real Meram data: 34,794 entities, 29,150 steps)**

| Relation | Task | MRR | H@10 | Notes |
|----------|------|-----|------|-------|
| REQUIRES_PERMIT | Structural (deterministic) | **0.431** | **1.00** | H@10=1.0 across 8 permit types; MRR improved vs synthetic (0.30→0.43) with 29k real steps |
| ASSIGNED_TO | Stochastic (many-to-many) | 0.0002 | 0.00 | Expected near-zero: 29,150 candidate steps, assignment driven by schedule not graph structure |

  MRR≈0 on ASSIGNED_TO is structurally expected: embedding models excel at deterministic, rule-like relations; stochastic many-to-many assignments require dynamic context (availability, specialisation) not encoded in the quadruple structure.

**Violation Detection — all models (real Meram data, stratified 70/30 split, 134 violations in test set)**

| Model | Type | Precision | Recall | macro-F1 | AUC-ROC | Notes |
|-------|------|-----------|--------|----------|---------|-------|
| **T-Logic R1+R2** | **Symbolic** | **1.00** | **1.00** | **1.00** | — | Perfect P/R on post-rule-change test (274 violations) — notebook 07 |
| Random Forest | Feature ML | 0.053 | 0.141 | 0.525 | **0.933** | Best AUC; temporal split — notebook 06 |
| TGN cert-aware | Neural GNN | 0.060 | 0.259 | 0.529 | 0.859 | Stratified split, cert-aware features — notebook 06 |
| Logistic Regression | Feature ML | 0.050 | 0.193 | 0.522 | 0.857 | Temporal split — notebook 06 |
| TGN-B (focal γ=2 + balanced batching) | Neural GNN | 0.060 | 0.207 | 0.530 | 0.767 | Stratified split, feature-aware — scripts/eval_models_testset.py |
| TNTComplEx (violation scoring) | KG Embedding | — | — | — | 0.447 | **Below random**: PERMIT_DENIED scoring not suited for event classification |

  **Key result:** T-Logic R1+R2 achieves **P=1.0, R=1.0** on the temporal test set — perfect recall with full interpretability. Among probabilistic models, **Random Forest (AUC=0.933)** and **TGN cert-aware (AUC=0.859)** are the strongest. TNTComplEx is not suited for violation *detection* (AUC<0.5) — its value is structural link prediction (see table above).

  **Why TNTComplEx fails for violation detection:** The model predicts graph structure (which triples exist), not event outcomes. PERMIT_DENIED is driven by cert-state and domain rules, not graph topology. TNTComplEx excels at deterministic structural relations (REQUIRES_PERMIT H@10=1.0) but cannot generalise to stochastic assignment violations.

  **Precision ceiling:** with 449 violations in 29,150 events (1.5% rate) all models show low precision at high recall — this is structurally expected at this imbalance ratio. AUC-ROC is the correct comparison metric; T-Logic bypasses this by using domain rules directly.

  **Bitemporal rule change detection:** TNTComplEx score for `(hot_work, REQUIRES_CERT, Advanced_Fire_Watch, τ)` peaks at **month 6** confirming the model correctly captures the compliance rule update in its structural embeddings, even though it cannot classify individual violation events.

#### What is real vs synthetic in UseCase4
| Data | Source |
|---|---|
| Activity / Family / Step names and codes | ✅ Real TR Meram PCS + Family_Steps_macro.xlsm (8,762 activities) |
| Step sequences (PRECEDES) | ✅ Real TR Family Steps templates |
| Estimated hours, earned hours, discipline, area, CWP | ✅ Real TR Meram PCS data |
| Discipline timeline (months) | ⚠️ Estimated (hardcoded) |
| Workers, certifications, work permits | ❌ Synthetic |
| Bitemporal rule change scenario | ❌ Synthetic (for demo) |

---

## Key Findings

| | UseCase1 | UseCase2 | UseCase3 | UseCase4 |
|---|---|---|---|---|
| **Domain** | Synthetic turbine | Oil well anomaly detection | EPC delay causality | EPC project compliance |
| **Graph type** | Sensor TKG (dynamic) | Sensor TKG (dynamic) | Delay causal TKG | Planning TKG (bitemporal) |
| **ML approach** | IsolationForest + TGN | TGN → RF/XGBoost (improved) | T-Logic symbolic rules | TNTComplEx, TGN, T-Logic |
| **Main result** | Threshold-tuned IF baseline | Improved recall per anomaly class | R1+R2+R3 causal validation | T-Logic P=1.0 R=1.0; RF AUC=0.933; TNTComplEx REQUIRES_PERMIT MRR=0.431 H@10=1.0 (link prediction) |

**Thesis contribution:** TKG as a unifying framework for heterogeneous industrial domains — anomaly detection, causal analysis, and compliance tracking all benefit from the temporal graph structure, with symbolic rules (T-Logic) providing interpretability critical for safety-critical applications.

---

## How to Run

### Prerequisites
```bash
pip install neo4j pandas numpy matplotlib networkx matplotlib-venn scikit-learn xgboost torch
```
Neo4j running at `bolt://172.22.43.151:7687`

### UseCase1 — Synthetic Turbine
```bash
# Run notebooks in order
# notebooks/UseCase1/01_generate_explore.ipynb  (generates synthetic_turbine.csv)
# notebooks/UseCase1/02_anomaly_detection.ipynb
# notebooks/UseCase1/03_neo4j_queries.ipynb
```

### UseCase2 — 3W Dataset
```bash
# Run notebooks in order
# notebooks/UseCase2/01_explore_3w.ipynb
# notebooks/UseCase2/02_preprocessing.ipynb
# notebooks/UseCase2/03_load_neo4j.ipynb
# notebooks/UseCase2/04_tgn_training.ipynb
# notebooks/UseCase2/05_results_analysis.ipynb
# notebooks/UseCase2/06_improved_models.ipynb   (RF/XGBoost with class imbalance)
```

### UseCase3 — EPC Causal Analysis
```bash
# Run notebooks in order
# notebooks/UseCase3/01_explore_epc.ipynb
# notebooks/UseCase3/02_tkg_build.ipynb
# notebooks/UseCase3/03_tlogic_causal.ipynb
```

### UseCase4 — Import and Run
```bash
# Generate dataset
python3 data/UseCase4/generate_epc_dataset.py

# Import into Neo4j
python3 data/UseCase4/import_graph_real.py

# Run notebooks in order
# notebooks/UseCase4/01_explore_epc.ipynb through 07_four_layer_tlogic.ipynb
```

---

## References
- Vargas et al. (2019) — 3W Dataset, Journal of Petroleum Science and Engineering
- Rossi et al. (2020) — Temporal Graph Networks (TGN)
- Lacroix et al. (2020) — Tensor Decompositions for Temporal Knowledge Bases (TNTComplEx)
- Liu et al. (2022) — T-Logic: Temporal Logical Rules for Explainable Link Forecasting
- TR Internal — Family_Steps_macro.xlsm (EPC planning data)
