# GAGS-HE Paper 1 — Reproducibility Release v1.0.0

## Associated manuscript

**Whether model fairness is visible at pathway level depends on how often access is constrained**

Thomas Enesi Baidoo  
Veritas University Abuja, Abuja, Nigeria  
baidoot@veritas.edu.ng

## Purpose

This repository snapshot contains the frozen code and publication outputs for the Article 1 v11 geographic-replication experiment in the Healthcare Equity module of the Global AI Governance Sandbox (GAGS).

The study is a **simulation and audit testbed**. It is not a deployed clinical decision-support system and does not estimate Nigerian population prevalence or individual travel time.

## What is included

- `02_Healthcare_Equity_Article1_v10.py` — frozen healthcare-equity implementation containing the functions used by the publication runner.
- `run_article1_v11_geographic_replication.py` — publication runner.
- `requirements.txt` — core Python dependencies.
- `DATA_SOURCES.md` — data provenance, expected filenames and cryptographic hashes.
- `REPRODUCIBILITY.md` — exact reproduction commands and interpretation notes.
- `RELEASE_METADATA.json` — frozen design metadata.
- `results/` — publication CSV outputs.
- `figures/` — publication figures.
- `CITATION.cff` — citation metadata.
- `.zenodo.json` — Zenodo metadata.

## Data are intentionally not redistributed here

The release does **not** include the UCI Cleveland file or the GRID3 workbook. Users should obtain the data from the official providers and place the files locally using the filenames documented in `DATA_SOURCES.md`.

This avoids implying redistribution rights for third-party datasets and keeps the software release separate from source-data licensing.

## Main design

- 30 model seeds: 42–71.
- UCI Cleveland complete-case benchmark.
- Referral-eligible population defined by the prespecified research screen-to-referral policy.
- 100 geographic origin realizations per referral-eligible clinical case.
- Geographic origins sampled uniformly with replacement from functional primary facilities.
- Same geographic draws reused across model variants within each seed.
- Accessibility proxy: geodesic distance divided by an assumed 40 km/h.
- Thresholds: 30, 45, 60 and 90 minutes.
- Seed-level bootstrap uncertainty.
- Geographic realizations are Monte Carlo location replicas, **not additional patients**.

## Important interpretation constraints

- UCI Cleveland is an external benchmark, not Nigerian patient data.
- Functional secondary/tertiary facilities are not assumed to be cardiology-capable.
- The geodesic accessibility calculation is an optimistic proxy, not observed road-network travel time.
- Facility-weighted simulation results are not Nigerian population estimates.
- `J` is a joint burden measure, not an interaction term.
- In the primary uncoupled design, the near-zero excess-beyond-independence diagnostic is expected by design.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

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

See `REPRODUCIBILITY.md` for validation checks and expected hashes.

## Versioning

This release should be tagged **`paper1-v1.0.0`** on GitHub before archival in Zenodo.

After Zenodo creates the DOI, add the DOI badge/link to the main GitHub README and insert the DOI in the manuscript Code Availability statement. Do not modify the archived files for this release after DOI publication; make a new version if a substantive correction is needed.
