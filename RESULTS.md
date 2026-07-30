# Results

Produced by a single reproducible command (no external data — five 16-bit depth
maps are synthesized with fixed seeds into `results/synthetic_data/`):

```bash
python run_all.py
```

Artifacts under [`results/`](results/): per-experiment logs and a parsed
[`results/mse_summary.json`](results/mse_summary.json). MSE is averaged over the
five synthetic images below; values are on the 16-bit depth scale.

## Project 1 — clean round-trip (MSE, mean over 5 images)

| Method | MSE | Notes |
| --- | --- | --- |
| 1 — byte split (upper byte only) | ~21,650 | lossy by design (drops the low byte) |
| 2 — spectral color-wheel | ~638 | smooth, mostly recoverable |
| 3 — 6/6/4 bit-field | **0.0** | exactly lossless when clean |

Full log: [`results/01_mapping_clean.log`](results/01_mapping_clean.log).

## Project 2 — degraded round-trip (Gaussian noise + lossy JPEG)

| Method | MSE (mean) |
| --- | --- |
| 1 — byte split (QP=90) | ~2.4e5 |
| 2 — spectral color-wheel | ~6.1e8 |
| 3 — 6/6/4 bit-field | ~7.9e7 |

Full log: [`results/02_mapping_degraded.log`](results/02_mapping_degraded.log).

The key finding: **Method 3 flips from best (lossless) to catastrophic under
degradation.** Bit-field packing has no redundancy, so noise/JPEG corrupting the
least-significant bits produces huge depth errors — whereas the spectral map,
though never lossless, degrades more gracefully in the clean case. This is the
precision-vs-robustness trade-off the project set out to study.

## Original notebook figures

The forward-mapping visualizations and reconstruction figures embedded in the
original notebooks are preserved under
[`results/notebook_reference/`](results/notebook_reference/) for provenance.
