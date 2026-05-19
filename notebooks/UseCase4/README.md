# EPC Compliance Notebooks — TR Meram Thesis

These notebooks document the full analysis pipeline for the MSc thesis on Temporal Knowledge Graph-based EPC permit compliance monitoring.

For chapter-to-notebook mapping, see [THESIS_GUIDE.md](../../THESIS_GUIDE.md).

## Notebooks (run in order)

| Notebook | Section | Description |
|----------|---------|-------------|
| `01_explore_epc.ipynb` | §3b | Dataset exploration, entity stats, Neo4j verification |
| `02_temporal_queries.ipynb` | §3b | Bitemporal Cypher compliance queries |
| `03_critical_path.ipynb` | §3c | Critical path & bottleneck analysis |
| `04_event_stream_analysis.ipynb` | §3e | Event stream characterisation |
| `05_tgn_epc.ipynb` | — | TGN prototype (early exploration, superseded by 08) |
| `06_tkg_models.ipynb` | — | TNTComplEx + RF/XGBoost development notebook |
| `07_four_layer_tlogic.ipynb` | §4d | T-Logic symbolic rules, R1/R2/R3, P=R=F1=0.875 |
| **`08_model_benchmark_final.ipynb`** | **§5** | **★ MAIN: full benchmark, all models, all results** |

## Archive

`archive_synth_v1/` contains the **synthetic-data prototype** (notebooks 01–07 built before real TR Meram data was available). These are kept for development history and are **not part of the thesis results**.
