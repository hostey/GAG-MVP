# Data sources

## 1. UCI Heart Disease — Cleveland

Official dataset: UCI Machine Learning Repository, Heart Disease.

For this release, the expected local filename is:

`data/processed.cleveland.data`

Expected SHA-256:

`a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`

The publication run used the Cleveland benchmark and complete-case analysis, yielding 297 complete rows.

UCI dataset DOI: `10.24432/C52P4X`

The dataset is third-party data and is not redistributed in this software archive.

## 2. GRID3 Nigeria Health Facilities

Publication source used for the nationwide facility backbone:

**GRID3 NGA - Health Facilities v2.0 (2024)**

DOI: `10.7916/kv1n-0743`

For this release, the expected local filename is:

`data/grid3_nga_-_health_facilities_-1.xlsx`

Expected SHA-256:

`c438adbeec7577515a945b5dffebe6b97af52a4a291c5f110b9088feb36d60a2`

The frozen analysis identified:
- 32,228 functional primary-facility origins.
- 2,024 functional secondary- or tertiary-level eligible referral facilities.

The GRID3 workbook is not redistributed in this software archive. Obtain the data from the official GRID3 source and observe the provider's current licensing/redistribution terms.

## Hash checking

### Windows PowerShell

```powershell
Get-FileHash .\data\processed.cleveland.data -Algorithm SHA256
Get-FileHash .\data\grid3_nga_-_health_facilities_-1.xlsx -Algorithm SHA256
```

### macOS/Linux

```bash
sha256sum data/processed.cleveland.data
sha256sum data/grid3_nga_-_health_facilities_-1.xlsx
```

Do not proceed with a claimed exact reproduction if the hashes do not match.
