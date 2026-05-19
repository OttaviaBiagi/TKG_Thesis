# tests/ — Data Validation

These are pytest-style data quality checks that validate the TR Meram dataset before running experiments.

**Run once before training:**
```bash
python tests/test_real_data.py
```

Expected output: 22 PASS / 0 FAIL / 3 WARN

| File | Tests |
|------|-------|
| `test_real_data.py` | TR Meram EPC data — 22 quality checks (T1-T5 label sanity + structural checks) |
| `test_3w.py` | [LEGACY] UC2 leftover (Petrobras 3W dataset) — not relevant to thesis |

**Difference from `experiments/`:**  
`tests/` = verify the data is correct before you start.  
`experiments/` = train models and produce results.
