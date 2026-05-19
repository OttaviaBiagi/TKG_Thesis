# scripts/ — Development Utilities

These scripts were used during development and **are not part of the thesis pipeline**.

The active thesis pipeline is in `experiments/UseCase4/`.

| Script | Was used for |
|--------|-------------|
| `run_exp_h/i/j.py` | One-off experiments extracted from notebook 06 |
| `inject_exp_*.py` | Injecting results back into notebooks during development |
| `debug_exp_j.py`, `save_exp_*.py` | Debug and output capture during development |
| `patch_neo4j_db.py / revert` | One-time Neo4j data corrections |
| `eval_models_testset.py` | Ad-hoc test-set evaluation for TNTComplEx/TGN-B |
| `delay_analysis.py` | EPC delay propagation exploration |
| `plot_roc.py`, `check_weights.py` | Plotting and weight inspection helpers |
