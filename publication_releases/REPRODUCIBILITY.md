# Reproducibility instructions

## 1. Create the environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Create a local data directory

```bash
mkdir data
```

Place the two externally obtained source files there:

- `data/processed.cleveland.data`
- `data/grid3_nga_-_health_facilities_-1.xlsx`

Verify the SHA-256 hashes in `DATA_SOURCES.md`.

## 3. Run the frozen experiment

```bash
python run_article1_v11_geographic_replication.py \
  --uci data/processed.cleveland.data \
  --grid3 data/grid3_nga_-_health_facilities_-1.xlsx \
  --runs 30 \
  --geo-reps 100 \
  --bootstrap 1000 \
  --seed-base 42 \
  --speed 40 \
  --outdir reproduced_results
```

## 4. Design checks

The run should use:
- 30 seeds beginning at 42.
- 100 geography realizations per referral-eligible case.
- 40 km/h assumed speed.
- 30/45/60/90-minute thresholds.
- uniform facility sampling with replacement.
- the same geographic draws across model variants within each seed.

The publication design generated 126,000 case-location realizations across the 30 seeds. These are Monte Carlo case-location realizations, not 126,000 patients.

## 5. Publication-level checks

The publication summaries should be close to the frozen results in `results/`. Key baseline values include:

| Threshold | Overall C | Overall J |
|---|---:|---:|
| 30 min | 0.145611 | 0.032730 |
| 45 min | 0.052214 | 0.011794 |
| 60 min | 0.022270 | 0.004881 |
| 90 min | 0.003579 | 0.000754 |

Baseline subgroup disparity in joint burden, ΔJ = J0 − J1:

| Threshold | ΔJ |
|---|---:|
| 30 min | 0.010552 |
| 45 min | 0.004204 |
| 60 min | 0.001449 |
| 90 min | -0.000235 |

Paired reweighing minus baseline change in ΔJ:

| Threshold | ΔΔJ |
|---|---:|
| 30 min | -0.009784 |
| 45 min | -0.003847 |
| 60 min | -0.001530 |
| 90 min | -0.000182 |

Small last-decimal variation may occur if dependency implementations change. Exact archival reproduction should use the frozen environment and source-data hashes.

## 6. Statistical unit

The bootstrap resamples model seeds. Do not treat the 100 geographic replicas per case as independent clinical observations.

## 7. What this release does not establish

This archive does not establish:
- observed patient travel time,
- road-network travel time,
- population-weighted Nigerian access prevalence,
- cardiovascular-specific facility readiness,
- or causal policy effects.
