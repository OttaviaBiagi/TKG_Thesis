# EPC Compliance Monitoring with Temporal Knowledge Graphs

**MSc Thesis** · Universidad Politécnica de Madrid & Politecnico di Milano  
**Author:** Ottavia Biagi  
**Industry partner:** Tecnicas Reunidas S.A. — Meram Refinery Expansion (Turkey)

> This thesis designs, implements, and evaluates a Temporal Knowledge Graph (TKG) framework
> for automated EPC permit compliance monitoring, using real project activity data from Tecnicas
> Reunidas. The system detects work-permit violations — events where a worker is assigned to a
> step without valid certification — under realistic temporal and structural conditions.

---

## Research Framework

**Objective.** Investigate whether a TKG system designed for the EPC domain produces measurable
improvements in temporal query expressiveness, causal violation auditability, and delay cascade
characterization compared to approaches that do not employ graph-based temporal modelling.

| Sub-objective | Scope | Chapter |
|---|---|---|
| SO1 — Architecture | Hybrid TKG: OWL-2 ontological layer + Neo4j bitemporal property graph | 3–4 |
| SO2 — Query expressiveness | Expressiveness gap: standard Cypher vs Allen interval algebra patterns | 4.4 |
| SO3 — Causal rule extraction | T-Logic rules for delay cascade characterization + compliance violation auditability | 4.8 |
| SO4 — Operational feasibility | Query overhead: temporal vs atemporal queries at EPC scale | 5 |

| Hypothesis | Claim | Verdict |
|---|---|---|
| **H1** | Temporal path queries not expressible as single-construct Cypher; gap requires ≥ 3 steps | ✅ SUPPORTED |
| **H2** | T-Logic rules: precision ≥ 0.60, recall ≥ 0.70 | ✅ SUPPORTED — P=R=1.0 single project; P=0.963 R=0.991 F1=0.977 multi_varied (313 FP = manual overrides) |
| **H3** | Temporal query overhead < 50% relative to atemporal equivalents | ✅ SUPPORTED — max +43.7% (Neo4j P2); all P1–P5 PASS |

---

## SO1 — Architecture (Chapter 3–4)

Three-layer hybrid TKG architecture:

| Layer | Technology | Contents |
|---|---|---|
| Layer 1 — Conceptual | OWL-2 DL (`epc_tkg.ttl`) + rdflib 7.x | Classes, reification, bitemporal properties, EVM Module 3; SPARQL Q1–Q7 |
| Layer 2 — Property Graph | Neo4j (`import_graph_real.py`) | Activity, Step, Worker, WorkPermit, Certification nodes; ASSIGNED_TO, HAS_CERT, REQUIRES_CERT, PRECEDES edges |
| Layer 3 — Event Stream | TGN / TGAT / T-Logic | Timestamped ASSIGNED_TO events; 6 edge features; permit-denial labels |

**Loaded graph statistics (TR Meram compliance graph):**

| Element | Count | Notes |
|---|---|---|
| Project nodes | 1 | TR Meram Refinery Expansion |
| Activity nodes | 276 | After family-step expansion |
| Step nodes | 1,419 | Real TR Family Steps sequences |
| WorkPermit types | 8 | hot_work, confined_space, etc. |
| Certification types | 33 | Including AFW added Jun 2024 |
| Worker nodes | 50 | Synthetic, discipline-matched |
| REQUIRES_PERMIT edges | 1,477 | Step → permit type |
| REQUIRES_CERT edges | 24 | Permit type → certification (bitemporal) |
| HAS_CERT edges | 248 | Worker → certification (bitemporal) |
| PRECEDES edges | 1,208 | Step precedence constraints |
| ASSIGNED_TO edges (Neo4j) | 55 | Synthetic bitemporal (Worker→Step, hot_work post-RC) |
| ASSIGNED_TO events (stream) | 29,150 | Event stream in epc_events.json (Layer 3 — not in Neo4j) |
| PERMIT_DENIED events | 449 | Violation rate: 1.54% |

Data quality: **22 PASS / 0 FAIL / 3 WARN** — run `python tests/test_real_data.py`

**Bitemporal model.** Both layers encode two temporal dimensions:
- `valid_from / valid_to` — valid time (when the fact holds in the real world)
- `tx_time` — transaction time (when it was recorded in the system)

Concrete scenario: on 29 June 2024 the hot_work permit was updated to require Advanced Fire Watch (AFW).
The pre-change REQUIRES_CERT edge (without AFW) retains `valid_to = 2024-06-28`; the post-change edge
starts `valid_from = 2024-06-29`. Both coexist — neither overwrites the other. Result: 8 workers
qualified on 28 June; only 3 qualified on 30 June without any action on their part.

---

## SO2 / RQ1 / H1 — Temporal Query Expressiveness (Chapter 4.4, 5)

Standard Cypher treats time as a data attribute, not a structural dimension. Four EPC monitoring query
classes document the expressiveness gap, using extended Cypher approximations of Allen interval algebra.

| Query class | T-GQL operator | Standard Cypher | SPARQL witness |
|---|---|---|---|
| **Class 1** — Snapshot | `SNAPSHOT(t)` | Single `WHERE valid_from <= t AND valid_to >= t` | Q1, Q2 |
| **Class 2** — Interval overlap | `BETWEEN(t1, t2)` | Two comparisons joined by `AND` | Q4 |
| **Class 3** — Simultaneous validity | `WHEN(A, B)` | **3 separate queries** + application-layer intersection | Q3, Q5 |
| **Class 4** — Bitemporal forensic | (both axes) | Requires `tx_time` attribute; impossible in uni-temporal graph | Q5 |

**H1 witness — Class 3** is the critical query class: "which activities were on the critical path
while their required work package was simultaneously in a delayed procurement state?" Standard Cypher
requires three separate constructs with no formal correctness guarantee. T-GQL's `WHEN(A, B)` is a
single operator implementing Allen's `overlaps`. H1 is supported.

**SPARQL Q1–Q7 — OWL-2 / rdflib layer (6,217 triples):**

| Query | File | Results | Capability |
|---|---|---|---|
| Q1 | `Q1_workers_before_rule_change.sparql` | 12 workers | Valid-time slice: state 2024-06-28 |
| Q2 | `Q2_workers_after_rule_change.sparql` | 12 workers | Valid-time slice: state 2024-07-01 |
| Q3 | `Q3_delta_non_compliant.sparql` | 21 rows | `FILTER NOT EXISTS` delta — 7 workers lost compliance; requires 6 Cypher `WITH`-steps |
| Q4 | `Q4_audit_trail.sparql` | 7 rows | Full bitemporal audit: validFrom/validTo + txTime per version |
| Q5 | `Q5_bitemporal_asof.sparql` | 143 rows | True bitemporal as-of: both axes filtered simultaneously |
| Q6 | `Q6_violation_inference.sparql` | 180 triples | CONSTRUCT materialising `ComplianceViolation` via 4-hop chain |
| Q7 | `Q7_evm_spi_atrisk.sparql` | 100 rows | Module 3 EVM: activities with SPI < 0.9 (critical schedule slippage) |

Q3, Q5, Q6 are the primary H1 witnesses: each demonstrates a SPARQL capability that requires
multi-step procedural workarounds in standard Cypher.

**Ontology schema:**  
Classes: `Project · Activity · Family · Step · WorkPermit · Certification · Worker`  
Reification: `CertificationHolding · PermitCertRequirement · WorkerAssignment · ComplianceViolation`  
Key axiom: `epc:precedes` is `owl:TransitiveProperty` (transitive closure over step sequences)  
EVM properties: `epc:plannedValue · epc:earnedValue · epc:progressPct · epc:SPI · epc:CPI`

---

## SO3 / RQ2 / H2 — Causal Rule Extraction + Violation Auditability (Chapter 4.5–4.8, 5)

### T-Logic Symbolic Rules (notebook 07)

Three compliance rules specified as domain-driven T-Logic patterns (confidence = 1.0 on training partition):

| Rule | Body | Head | Semantics |
|---|---|---|---|
| **R1** | `ASSIGNED_TO(W,S,t) ∧ ¬HAS_CERT(W, req_cert, t)` | `PERMIT_DENIED(W,S,t)` | Worker assigned without valid certification |
| **R2** | `ASSIGNED_TO(W,S,t) ∧ t≥RC ∧ permit=hot_work ∧ ¬HAS_CERT(W, Advanced_Fire_Watch, t)` | `PERMIT_DENIED(W,S,t)` | Post rule-change hot_work without new cert |
| **R3** | `PERMIT_DENIED(W,S₁,t) ∧ PRECEDES(S₁,S₂)` | `cascade_risk(S₂, t+Δ)` | Upstream denial propagates to downstream steps |

**Note on rule origin:** R1 and R2 are manually specified domain rules, not mined by TLogic's random walk algorithm. This is a principled design choice: the compliance discriminant is negation-based (`¬HAS_CERT`) and TLogic's positive-pattern walk mining cannot recover negation signals. Section 12 of notebook 07 validates this by running the actual walk sampler and demonstrating it cannot rediscover R1 — confirming that domain-expert specification is the correct approach for this class of compliance problem.

**Results on single-project test set (post rule-change split, 20,040 events, 274 violations):**

| Model | P | R | F1 | Note |
|---|---|---|---|---|
| R1 (missing cert) | 1.000 | 1.000 | **1.000** | Deterministic domain rule |
| R2 (post-RC hot_work only) | 1.000 | 0.073 | 0.137 | Subset of R1 violations |
| R1+R2 combined | 1.000 | 1.000 | **1.000** | Evaluated on post-RC split; neural models use temporal 70/15/15 with 8 test violations — different protocols, not directly comparable |

**H2 verdict:** P=R=F1=1.0 on single project — well above thresholds (P≥0.60, R≥0.70). **H2 SUPPORTED.**

**Cross-project generalisation (§15 notebook 07):** T-Logic R1 evaluated on `multi_varied` dataset using `tlogic_all_datasets.json`:

| Model | Dataset | P | R | F1 | Note |
|---|---|---|---|---|---|
| T-Logic R1 | single | 1.000 | 0.991 | 0.996 | 0 FP — closed single-project environment |
| T-Logic R1 | multi_varied | **0.963** | **0.991** | **0.977** | 313 FP = manual manager overrides across projects |
| T-Logic R1+R2 | multi_varied | 0.942 | 0.999 | 0.970 | adds post-rule-change hot_work requirement |

R1 generalises without retraining: the compliance framework (cert requirements) is identical across all projects. The 313 false positives are cases where a worker lacked a required cert at assignment time but no PERMIT_DENIED was recorded — attributable to manager discretionary approvals.

**§14 fair comparison — same task + same test split (nb08 temporal split, 8 violations / 4,373 test events):**

| Model | AUC | P | R | F1 | Note |
|---|---|---|---|---|---|
| T-Logic R1 (no training) | — | 1.000 | 1.000 | **1.000** | Exact rule = performance ceiling |
| TGN seed=42 (nb08) | 0.985 | 0.044 | 1.000 | 0.084 | All 8 violations caught; 165 false alerts |

T-Logic R1 is the performance ceiling because it encodes the exact rule that generates the labels. TGN learns inductively and achieves AUC=0.985 (strong discrimination) but low precision under class imbalance (8/4373=0.18%).

### Violation Auditability (causal traceability)

The TKG provides complete causal auditability for all detected violations:

| Metric | Value | Source |
|---|---|---|
| Traceability completeness | **449/449 (100%)** violations traceable to specific missing cert via R1 | notebook 07 §5 |
| Causal chain depth | **4 hops** — Worker → ASSIGNED_TO → Step → REQUIRES_PERMIT → WorkPermit → REQUIRES_CERT → Certification | SPARQL Q6 |
| Violations materialised | **180 ComplianceViolation triples** constructed via Q6 SPARQL CONSTRUCT | `Q6_violation_inference.sparql` |
| Bitemporal audit trail | **7 versioned states** per compliance rule with valid-time + tx-time | SPARQL Q4 |

### Delay Cascade Characterization (R3)

R3 models how upstream permit denials propagate downstream through the `PRECEDES` dependency graph.
This is a structural analysis (deterministic graph traversal), not a learned prediction —
the causal mechanism (critical-path precedence) is fully known a priori.
Risk score: `risk(step at depth d) = 0.5^d` (geometric decay). Hours-at-risk = estimated_hours × risk_score.

**Single project (notebook 07 §13):**

| Metric | Value |
|---|---|
| Direct violations (R1+R2) | 449 |
| Downstream steps at cascade risk | **1,037** (depth 1–3 via PRECEDES transitive closure) |
| Cascade amplification factor | **2.3× steps** per direct violation |
| Project-wide exposure | **73% of all 1,419 steps** reachable from at least one violation |
| Hours-at-risk | ~517,000 h (weighted by risk score × estimated hours per step) |

**Cross-project generalisation — multi_varied (notebook 07 §15.3, 100 projects):**

| Metric | Multi-varied mean | Range |
|---|---|---|
| Violations per project | 435 | 390–481 |
| Cascade steps per project | 1,001 | 886–1,120 |
| **Amplification factor** | **2.30× ± 0.04** | 2.19–2.37 |
| Hours-at-risk per project | 573,380 h | — |
| Total hours-at-risk (100 projects) | **57,338,044 h** | — |

**Key finding:** The 2.30× cascade amplification is structurally stable across all project topologies (std=0.04). EPC project DAGs consistently produce ~2.3× downstream step exposure per direct violation — R3 generalises without re-calibration, confirming it is a structural property of EPC scheduling rather than a single-project artefact.

---

### Violation Detection Benchmark (notebook 08)

**Prediction task:** Binary edge classification on ASSIGNED_TO events.
Given a `(worker, step, timestamp)` triple, predict whether the assignment results
in a `PERMIT_DENIED` compliance violation (label = 1) or not (label = 0).
Input: 6 edge features (`FEAT_COLS`). Class imbalance: 0.18% single-project, 0.24% multi_varied.

**What each model type learns:**

| Type | What it uses | What it learns |
|---|---|---|
| Temporal GNN (TGN, TGAT, DyRep) | Graph structure + event timestamps + temporal neighbourhoods | Sequential patterns in ASSIGNED_TO history — e.g. worker repeatedly assigned without cert |
| Structure-only GNN (StaticGNN) | Graph topology only, no timestamps | Which workers/steps are structurally at risk, ignoring when |
| KG embedding (ComplEx, TNTComplEx) | Entity/relation co-occurrence (TNT adds time embedding) | Whether a `(worker, ASSIGNED_TO, step)` triple is plausible — a link prediction task, not violation classification |
| Feature-only ML (LR, RF) | 6 edge features directly, no graph | Correlation between features (e.g. `after_rc`, `cert_expires_soon`) and violation label |

7 models evaluated on 4 splits across 3 dataset scales. Primary metric: AUPRC (class imbalance 0.18–0.24%).

**Edge features (`FEAT_COLS`):** `permit_enc · disc_enc · after_rc · on_critical_path · weight_pct · cert_expires_soon`

**Label sanity (5 tests, all PASS):**

| Test | Result | Verdict |
|---|---|---|
| T1 Feature–label correlation | 5/6 significant; `cert_expires_soon` r=+0.155 | ✓ PASS |
| T2 Distribution shift (Cohen's d) | Visible separation for 4/6 features | ✓ PASS |
| T3 Temporal clustering | Non-uniform rate; step change at rule-change date | ✓ PASS |
| T4 Label consistency | All (worker, step) pairs unique → 100% consistent | ✓ PASS |
| T5 Linear separability (LR 5-fold) | AUC = 0.654 ± 0.013 >> 0.5 | ✓ PASS |

#### Single-project (primary)

Temporal 70/15/15 split · 8 test violations / 4,373 events · threshold optimised on val set.

| Model | Type | AUC (3 seeds) | AUPRC (3 seeds) | Lift | Recall |
|---|---|---|---|---|---|
| **TGN** | Temporal GNN | **0.984 ± 0.001** | **0.177 ± 0.002** | **×98.9** | **1.000** |
| Random Forest | Feature-only ML | 0.978 | 0.160 | ×87.8 | 0.125 |
| TGAT | Temporal GNN | 0.822 | 0.046 | ×25.6 | 0.250 |
| Logistic Regression | Feature-only ML | 0.738 | 0.161 | ×88.4 | 0.625 |
| StaticGNN | Structure-only GNN | 0.773 ± 0.010 | 0.546 ± 0.057 † | — | — |
| TNTComplEx | Time-aware KG emb | 0.582 | 0.003 | ×1.6 | — |
| ComplEx | Static KG emb | 0.440 | 0.002 | ×1.0 | — |
| DyRep | Temporal GNN | 0.416 | 0.002 | ×1.1 | 1.000 ‡ |
| Random baseline | — | 0.500 | 0.002 | ×1.0 | — |

† StaticGNN single AUPRC is unreliable: val_AUPRC≈0.065 across all seeds; test_AUPRC (0.498/0.626/0.513)
is noise at 8 violations. AUC=0.759–0.784 confirms weak structural signal. Reliable StaticGNN result: multi_varied.  
‡ DyRep recall=1.0 is degenerate: threshold collapses near zero, flagging almost all events (precision=0.002).

#### Cross-project generalisation — multi_varied (diverse EPC families)

Temporal split · 201 test violations / 83,982 events · seeds 42, 43, 44 · lift vs test base rate (0.24%).

| Model | AUC (3 seeds) | AUPRC (3 seeds) | Lift (mean) | Note |
|---|---|---|---|---|
| T-Logic R1 (no training) | — | — | F1=**0.977** | P=0.963 R=0.991; full-dataset eval, no train/test split |
| **TGAT** | **0.979 ± 0.025** | **0.717 ± 0.073** | **×300 ± 31** | — |
| StaticGNN (d=2) | 0.932 ± 0.004 | 0.204 ± 0.112 | ×85 ± 47 | — |
| TGN | 0.983 ± 0.000 | 0.127 ± 0.001 | ×53 | — |
| ComplEx | 0.521 | 0.002 | ×1.0 | — |
| TNTComplEx | 0.516 | 0.002 | ×1.0 | — |

**Operational summary — what works and when:**

| Model | Works for | Reason |
|---|---|---|
| **TGN** | Single-project violation detection | Temporal memory captures worker cert history; recall=1.0 catches all 8 test violations |
| **TGAT** | Cross-project generalisation | Temporal attention adapts to unseen topologies; ×300±31 on multi_varied |
| **Random Forest** | Fast feature-only baseline | Strong AUC=0.978 but recall=0.125 — finds easy cases, misses rare violations |
| **KG embeddings** | Nothing — near-random | Wrong task: optimised for link prediction, not edge classification |
| **DyRep** | Nothing — degenerate | Threshold collapses near zero; flags almost everything (precision=0.002) |

**Architectural hierarchy confirmed across 3 seeds on 201 violations:**

```
ComplEx / TNTComplEx  ×1.0       (no structure, no time → random)
         StaticGNN   ×85 ± 47   (graph structure, no time → unstable)
               TGN   ×53 ± 0.5  (temporal memory, single-project optimised)
              TGAT   ×300 ± 31  (temporal attention, generalises cross-project)
```

Each architectural layer adds a measurable, reproducible capability — the central theoretical
contribution of the thesis.

---

## SO4 / RQ3 / H3 — Operational Feasibility (Chapter 5)

**H3 claim:** temporal query overhead < 50% relative to atemporal equivalents at prototype scale.

**Neo4j benchmark** (`data/UseCase4/run_cypher_benchmark.py`) — 5 pairs × 100 timed runs on 34,964-node graph:

| Pair | Query | Atemporal | Temporal | Overhead | H3 |
|---|---|---|---|---|---|
| P1 | 1-hop cert lookup | 1.9 ms | 2.1 ms | +9.4% | ✅ PASS |
| P2 | 3-hop compliance chain (Step→Permit→Cert→Worker) | 18.0 ms | 25.8 ms | +43.7% | ✅ PASS |
| P3 | Non-compliance detection | 1.0 ms | 1.0 ms | −4.1% | ✅ PASS |
| P4 | Bitemporal as-of (valid-time + tx-time) | 2.2 ms | 2.4 ms | +7.1% | ✅ PASS |
| P5 | 4-hop ASSIGNED_TO chain (Worker→Step→Permit→Cert) | 3.2 ms | 3.3 ms | +0.4% | ✅ PASS | † |

† P5 uses `check_date = 2025-07-01` (project end) rather than `DATE_POST = 2024-07-01`. ASSIGNED_TO edges have `valid_to = NULL` (ongoing), so only `valid_from ≤ check_date` filters; using project-end ensures QT5 returns the same 220 rows as QA5 for a fair overhead comparison.

**H3 OVERALL: SUPPORTED (P1–P5).** Max measured overhead +43.7% on the 3-hop compliance chain (P2); all five pairs well below 50% threshold.

**rdflib SPARQL benchmark** (`ontology/run_query_benchmark.py`) — 200 runs, in-memory, 6,217 triples:

| Pair | Query | Overhead | H3 |
|---|---|---|---|
| S1 | 1-hop cert holding | +199% | — (no property index; Neo4j with index: +9.4%) |
| S2 | 3-hop compliance chain | +44.7% | ✅ PASS |
| S3 | Single-axis vs bitemporal as-of | +32.5% | ✅ PASS |

S1 artefact is expected: rdflib has no property index; the Neo4j equivalent (P1) is +9.4% with index.

---

## Dataset (Chapter 4.2)

### Data Provenance — TR Meram

| Data element | Source | Status |
|---|---|---|
| Activity / Family / Step names, codes, sequences | TR Family_Steps_macro.xlsm | ✅ Real TR data |
| Estimated hours, earned hours, discipline, area, CWP | TR Meram PCS (8,762 rows → 5,555 unique) | ✅ Real TR data |
| Discipline timeline (months) | Hardcoded per discipline | ⚠️ Estimated |
| Workers, certifications, work permits | `simulate_events.py` | ❌ Synthetic |
| Permit denial events (labels) | Deterministic label generation | ❌ Synthetic |
| Bitemporal rule-change Jun 2024 | Demonstration scenario | ❌ Synthetic |

The project activity and step structure is real TR Meram data. The compliance layer is synthetic
due to the unavailability of real HSE personnel records (future work: expert validation
with TR project controller).

### Evaluation Scales

| Scale | Events | Test violations | Purpose |
|---|---|---|---|
| Single project | 29,150 | 8 (0.18%) | Primary — real TR Meram topology |
| Multi-varied (×30 varied) | 559,877 | 201 in test (0.24%) | Generalisation robustness — 30 structurally distinct EPC topologies |

---

## Repository Structure

```
TKG_Thesis/
│
├── data/UseCase4/                      # TR Meram EPC dataset
│   ├── epc_dataset_real.json           # TKG from real TR activity/step/worker data
│   ├── generate_epc_dataset.py         # Real TR PCS → TKG
│   ├── import_graph_real.py            # Neo4j import (Layer 2 — includes ASSIGNED_TO)
│   ├── run_cypher_benchmark.py         # SO4/H3 Neo4j overhead benchmark (5 pairs)
│   ├── projects/                       # Multi-project instances
│   └── queries/                        # Cypher: temporal compliance, critical path
│
├── notebooks/UseCase4/
│   ├── 01_explore_epc.ipynb               # Dataset exploration + Neo4j verification
│   ├── 02_temporal_queries.ipynb          # Bitemporal Cypher (SO2/RQ1) — 4 query classes
│   ├── 03_critical_path_analysis.ipynb    # Critical path & bottleneck analysis
│   ├── 04_event_stream_analysis.ipynb     # Event stream analysis + feature engineering
│   ├── 05_tgn_epc.ipynb                   # Early TGN prototype (exploratory)
│   ├── 06_tkg_model_development.ipynb     # TNTComplEx + RF/XGBoost baselines
│   ├── 07_tlogic_symbolic_reasoning.ipynb # T-Logic rules + cascade + walk mining (SO3/RQ2)
│   ├── 08_model_benchmark_final.ipynb     # [MAIN] Full benchmark: all models × all splits
│   └── 09_tkg_visualization.ipynb         # TKG graph visualisation
│
├── experiments/UseCase4/
│   ├── eval_framework.py               # split_dataset · compute_metrics · find_best_threshold
│   ├── data_loader.py                  # load_single_project / load_multi_project
│   ├── run_benchmark.py                # TGN/TGAT/DyRep × 4 splits × N seeds
│   ├── run_ml_baseline.py              # LR + RF feature-only baselines
│   ├── run_static_baseline.py          # ComplEx + TNTComplEx (all 3 scales)
│   ├── run_static_gnn.py               # StaticGNN (structure-only)
│   ├── tune_hyperparams.py             # Optuna TPE, 50 trials
│   ├── models/                         # TGN, TGAT, DyRep implementations
│   └── results/
│       ├── benchmark.json              # TGN/TGAT/DyRep results
│       ├── ml_baseline.json            # LR + RF
│       ├── static_baseline.json        # ComplEx + TNTComplEx
│       ├── static_gnn.json             # StaticGNN
│       ├── query_benchmark.json        # H3 Neo4j timing results
│       └── sparql_benchmark.json       # H3 rdflib timing results
│
├── ontology/                           # OWL-2 ontology + SPARQL layer (SO1/SO2)
│   ├── epc_tkg.ttl                     # OWL-2 DL schema
│   ├── epc_instance_data.ttl           # Populated individuals (6,217 triples)
│   ├── populate_onto.py                # Load dataset → rdflib graph
│   ├── run_sparql.py                   # Execute Q1–Q7
│   ├── run_query_benchmark.py          # SO4/H3 rdflib overhead benchmark
│   └── sparql/
│       ├── Q1_workers_before_rule_change.sparql
│       ├── Q2_workers_after_rule_change.sparql
│       ├── Q3_delta_non_compliant.sparql   # H1 witness: FILTER NOT EXISTS
│       ├── Q4_audit_trail.sparql           # Bitemporal audit
│       ├── Q5_bitemporal_asof.sparql       # H1 witness: dual time-axis
│       ├── Q6_violation_inference.sparql   # H1 witness: CONSTRUCT inference
│       └── Q7_evm_spi_atrisk.sparql        # Module 3 EVM: SPI < 0.9
│
└── tests/
    └── test_real_data.py               # 22 data quality checks
```

---

## How to Run

### Prerequisites
```bash
conda activate tkg-env
# or: pip install torch numpy pandas scikit-learn scipy optuna matplotlib rdflib neo4j
```

### SO1 — OWL-2 Ontology + SPARQL (Layer 1)
```bash
python ontology/populate_onto.py          # → epc_instance_data.ttl (6,217 triples)
python ontology/run_sparql.py             # Q1–Q7 results + summary table
```

### SO1 — Neo4j Import (Layer 2)
```bash
python data/UseCase4/generate_epc_dataset.py
python data/UseCase4/import_graph_real.py  # requires Neo4j at bolt://localhost:7687
```

### SO2 / RQ1 — Temporal Queries
```bash
jupyter lab   # open notebooks/UseCase4/02_temporal_queries.ipynb
```

### SO3 / RQ2 — T-Logic + Violation Detection
```bash
# T-Logic rules + cascade risk
jupyter lab   # open notebooks/UseCase4/07_tlogic_symbolic_reasoning.ipynb

# Full benchmark: all models × splits × seeds
python experiments/UseCase4/run_benchmark.py --dataset single --seeds 42 43 44
python experiments/UseCase4/run_benchmark.py --dataset multi_varied --seeds 42 43 44
python experiments/UseCase4/run_ml_baseline.py
python experiments/UseCase4/run_static_baseline.py --model all --dataset all
python experiments/UseCase4/run_static_gnn.py --dataset single
```

### SO4 / H3 — Overhead Benchmark
```bash
python ontology/run_query_benchmark.py            # rdflib benchmark (200 runs)
python data/UseCase4/run_cypher_benchmark.py      # Neo4j benchmark (100 runs, requires Neo4j)
```

---

## Methodology Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Temporal integrity (no future leakage) | ✅ Temporal 70/15/15 split as primary evaluation |
| 2 | Class imbalance handling | ✅ Weighted BCE + AUPRC as primary metric + val-set threshold |
| 3 | Hyperparameter optimisation | ✅ Optuna TPE, 50 trials, val-AUPRC objective |
| 4 | Same protocol for all models | ✅ Identical splits, scaler, feat_cols, threshold procedure |
| 5 | Feature-only baselines | ✅ LR + RF on FEAT_COLS |
| 6 | Inductive evaluation | ✅ 10% worker nodes withheld from training |
| 7 | Temporal drift analysis | ✅ 6-slot split — per-time-window metrics |
| 8 | Label validation | ✅ 5 empirical sanity tests T1–T5 (all PASS) |
| 9 | Reproducibility | ✅ Fixed seed=42; multi-seed via `--seeds 42 43 44` |
| 10 | Multi-project generalisation | ✅ T-Logic R1 F1=0.977 (no training); TGAT ×300±31 AUPRC (multi_varied, 3 seeds); architectural hierarchy confirmed |
| 11 | Expert label validation | ⏳ Future work — requires TR HSE records |
| 12 | Static KG baselines | ✅ ComplEx + TNTComplEx (random at all scales); StaticGNN (AUC=0.773±0.010 single; ×85±47 multi_varied) |
| 13 | OWL-2 ontology + SPARQL (SO1) | ✅ epc_tkg.ttl (OWL-2 DL); Q1–Q7 verified; 6,217 triples; Module 3 EVM |
| 14 | Temporal query overhead (SO4/H3) | ✅ H3 SUPPORTED — Neo4j max +43.7% (P2, all P1–P5 PASS); rdflib S2 +44.7%, S3 +32.5% |

---

## Future Work

| Priority | Item | Rationale |
|---|---|---|
| **Data** | Expert validation with TR HSE records (checklist item 11) | Compliance layer is synthetic; real permit-denial logs would validate R1/R2 rules and retrain neural models on actual violations |
| **Data** | Extend to multi-project federation with shared ontology | Current `multi_varied` uses independent projects; a shared TKG across concurrent TR projects enables cross-project compliance queries |
| **Modelling** | EvolveGCN-O as DTDG baseline | §2.3.5 positions EvolveGCN as the discrete-time counterpart to TGN/TGAT; adding it closes the CTDG vs DTDG comparison gap with daily snapshots |
| **Modelling** | Hybrid symbolic-neural ensemble (Experiment H) | T-Logic R1+R2 (recall=1.0) + TGN-B (precision=1.0) complement each other; notebook 07 §10 outlines the design; needs formal benchmark |
| **Modelling** | Inductive learning across permit types | Current inductive split withholds worker nodes; extending to unseen permit types tests whether temporal patterns generalise to new regulatory contexts |
| **Query layer** | Native T-GQL / GQL implementation | Current Allen-algebra approximations require multi-step Cypher; ISO GQL (standardised 2024) or a T-GQL layer would make Class 3 simultaneous-validity queries first-class |
| **Compliance** | Continuous / streaming compliance monitoring | Current system runs batch queries; a streaming layer (Neo4j CDC or Kafka) would enable real-time permit-denial alerts as ASSIGNED_TO events arrive |
| **Compliance** | Cert expiry forecasting | `cert_expires_soon` is a binary feature; a time-to-expiry regression model on HAS_CERT `valid_to` dates would enable proactive compliance management |
| **Evaluation** | Delay cascade validation | R3 cascade risk scores have not been validated against actual project delay data; if TR PCS provides schedule-actual data, cascade predictions can be evaluated against observed delays |

---



- Rossi et al. (2020) — Temporal Graph Networks (TGN) · NeurIPS
- Xu et al. (2020) — Inductive Representation Learning on Temporal Graphs (TGAT) · ICLR
- Zuo et al. (2018) — Embedding Temporal Network via Neighbourhood Formation (DyRep)
- Trouillon et al. (2016) — Complex Embeddings for Simple Link Prediction (ComplEx) · ICML
- Lacroix et al. (2020) — Tensor Decompositions for Knowledge Base Completion (TNTComplEx)
- Kipf & Welling (2017) — Semi-Supervised Classification with GCN · ICLR
- Liu et al. (2022) — T-Logic: Temporal Logical Rules for Explainable Link Forecasting
- Debrouvier et al. (2021) — T-GQL: Temporal Graph Query Language
- Hogan et al. (2021) — Knowledge Graphs · ACM Computing Surveys
- Jensen & Snodgrass (1999) — Temporal Data Management · VLDB Journal
- Ratner et al. (2017) — Data Programming: Creating Large Training Sets Quickly · NeurIPS
- PMI (2021) — A Guide to the Project Management Body of Knowledge (PMBOK 7th ed.)
- W3C OWL-2 Recommendation (2012) · W3C Time Ontology (OWL-Time, 2022)
- TR Internal — Family_Steps_macro.xlsm · Meram_PCS_Progress.xlsx
