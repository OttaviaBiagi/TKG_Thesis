# TKG_Thesis
**Temporal Knowledge Graph Systems for Industrial Monitoring, Prediction, and Decision Support**

MSc Thesis — Tecnicas Reunidas / Universidad Politécnica de Madrid  
Author: Ottavia Biagi

---

## Objective
Design and implement Temporal Knowledge Graph (TKG) systems for two industrial domains:
1. **Anomaly detection** in oil well sensor data (UseCase2 — 3W Dataset)
2. **EPC project compliance and scheduling** (UseCase4 — TR Family Steps data)

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
│   ├── UseCase2/
│   │   ├── 01_explore_3w.ipynb          # 3W dataset exploration
│   │   ├── 02_preprocessing.ipynb       # Feature engineering
│   │   ├── 03_load_neo4j.ipynb          # Load 3W TKG into Neo4j
│   │   ├── 04_tgn_training.ipynb        # TGN model training (AUC-ROC 0.61)
│   │   └── 05_results_analysis.ipynb    # Results analysis
│   └── UseCase4/
│       ├── 01_explore_epc.ipynb         # EPC dataset exploration + Neo4j verification
│       ├── 02_temporal_queries.ipynb    # Bitemporal compliance queries
│       ├── 03_critical_path.ipynb       # Critical path & bottleneck analysis
│       ├── 04_dynamic_tkg.ipynb         # Event stream analysis (ASSIGNED_TO, COMPLETED, PERMIT_DENIED)
│       └── 05_tgn_epc.ipynb             # TGN training on EPC events (delay + violation prediction)
│
├── experiments/
│   ├── UseCase4/                        # Plots and outputs from UseCase4 notebooks
│   └── ...
│
└── src/                                 # Shared utilities
```

---

## Use Cases

### UseCase2 — Anomaly Detection (3W Dataset)
- **Data:** 3W Dataset (Petrobras, Vargas et al. 2019) — 2,228 oil well instances, 8 sensors, 8 anomaly types
- **Approach:** Temporal Graph Network (TGN) on sensor TKG
- **Result:** AUC-ROC 0.61 — limited by severe class imbalance (597 normal vs 14 rarest anomaly)
- **Neo4j:** 9 wells, 8 sensors, temporal observations

### UseCase4 — EPC Compliance, Scheduling & Dynamic TKG
- **Data:** Real TR Family Steps (276 activities, 1,518 steps, 17 disciplines) + synthetic HSE layer
- **Approach:** Bitemporal TKG (valid_time + transaction_time) + simulated dynamic event stream
- **Results:**
  - Bitemporal compliance: 8 workers qualified before rule change → 3 after (5 lost compliance)
  - Critical path: 18 steps (ME.CT — Cooling Tower Erection)
  - Main bottleneck: CI discipline (563 steps), BU.BR.AR blocks 57,341 downstream steps
  - Dynamic events: ASSIGNED_TO, COMPLETED (with delay cascade), PERMIT_DENIED (~5% violation rate)
- **Neo4j:** 1 Project, 276 Activities, 1,518 Steps, 8 WorkPermits, 33 Certifications, 50 Workers
- **Next:** TGN on EPC event stream — predict delay_days and compliance violations

#### What is real vs synthetic in UseCase4
| Data | Source |
|---|---|
| Activity / Family / Step names and codes | ✅ Real TR Family_Steps_macro.xlsm |
| Step sequences (PRECEDES) | ✅ Real TR data |
| Discipline timeline (months) | ⚠️ Estimated (hardcoded) |
| Workers, certifications, work permits | ❌ Synthetic |
| Bitemporal rule change scenario | ❌ Synthetic (for demo) |

---

## How to Run

### Prerequisites
```bash
pip install neo4j pandas numpy matplotlib networkx matplotlib-venn
```
Neo4j running at `bolt://172.22.43.151:7687`

### UseCase4 — Import and Run
```bash
# Generate dataset
python3 data/UseCase4/generate_epc_dataset.py

# Import into Neo4j
python3 data/UseCase4/import_graph_real.py

# Run notebooks in order
# notebooks/UseCase4/01_explore_epc.ipynb
# notebooks/UseCase4/02_temporal_queries.ipynb
# notebooks/UseCase4/03_critical_path.ipynb

# Simulate dynamic events
python3 data/UseCase4/simulate_events.py

# Run dynamic analysis and TGN
# notebooks/UseCase4/04_dynamic_tkg.ipynb
# notebooks/UseCase4/05_tgn_epc.ipynb
```

---

## Key Findings

| | UseCase2 | UseCase4 |
|---|---|---|
| **Domain** | Oil well anomaly detection | EPC project compliance |
| **Graph type** | Sensor TKG (dynamic) | Planning TKG (bitemporal) |
| **ML approach** | TGN (temporal GNN) | Graph queries (Cypher) |
| **Main result** | AUC-ROC 0.61 | 5 workers lost compliance after rule change |
| **Limitation** | Class imbalance | Synthetic HSE data |

**Thesis contribution:** TKG as a unifying framework for heterogeneous industrial domains — both anomaly detection and compliance tracking benefit from the temporal graph structure.

---

## References
- Vargas et al. (2019) — 3W Dataset, Journal of Petroleum Science and Engineering
- Rossi et al. (2020) — Temporal Graph Networks (TGN)
- TR Internal — Family_Steps_macro.xlsm (EPC planning data)
