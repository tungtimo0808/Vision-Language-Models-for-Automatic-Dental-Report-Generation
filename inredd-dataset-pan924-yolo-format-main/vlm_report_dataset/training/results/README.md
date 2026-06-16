# Results

This folder collects the final results from the RTX 3090 scripts. Each `train_<model>.py`
writes here automatically after training:

- `<model>_results.txt` — a human-readable report: the best checkpoint, the validation
  macro-F1 of every checkpoint, and the full test-set metrics (per-condition precision /
  recall / F1, FDI detection, hallucination & miss rates, exact-report match, ROUGE-L).
- `metrics_<model>_test.json` — the same test metrics as machine-readable JSON.

Compare all models once they are done:
```bash
python ../tools/compare_models.py metrics_*_test.json
```
