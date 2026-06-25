# Evaluation of the Effect of Banco del Bienestar on Private Financial Infrastructure: A Dynamic Causal Study, 2019–2022

## Summary

This project evaluates whether the 2022 Banco del Bienestar branch expansion affected private financial infrastructure (corresponsales) in previously unserved Mexican municipalities, using a staggered Difference-in-Differences design (Callaway-Sant'Anna) on municipal-level CNBV panel data.

**Key finding:** No statistically significant effect was detected. The result is stable across three estimation specifications and robust to two different control group definitions. This null result is interpreted as informative rather than inconclusive — the identification strategy was tested explicitly, including a formal event-study check of the parallel trends assumption.

## Data

- **Source:** CNBV (Comisión Nacional Bancaria y de Valores), "Bases de Datos de Inclusión Financiera"
- **Coverage:** 16 quarters, March 2019–December 2022, ~2,463–2,472 municipalities per quarter
- **Unit of observation:** Municipality × quarter

## Method

- Treatment: first 0→1 transition in development-bank branches, restricted to municipalities with zero such branches at 2019Q1 baseline
- Estimator: Callaway-Sant'Anna (staggered-adoption-robust), outcome in log(1+corresponsales)
- Identification threats addressed: baseline placement selection (quantified via a comparability table), treatment misclassification risk, concurrent private-sector expansion

## Repository structure

├── notebooks/

│   ├── 00-data-loading.ipynb       # Raw data cleaning and panel construction

│   └── 02-did-analysis.ipynb       # Full causal analysis: identification, results, limitations

├── data/

│   ├── raw/                         # CNBV quarterly source files (not tracked — see .gitignore)

│   └── processed/

│       └── cnbv_panel.csv           # Cleaned municipal panel (39,459 rows × 21 columns)

└── README.md

## Key limitations

- Short post-treatment window (up to 4 quarters for the earliest treated cohort)
- Treatment variable aggregates all development banks; cannot isolate Banco del Bienestar with full certainty (though available evidence rules out comparable expansion by other institutions in this period)
- Cannot fully separate private-infrastructure response to treatment from independent private expansion

Full methodology, identification strategy, and disclosed limitations are documented in `02-did-analysis.ipynb`.

## Stack

Python · pandas · linearmodels · csdid · matplotlib/seaborn

## Author

Mathias Gomez Chan
[GitHub](https://github.com/Mathias70473) · [LinkedIn](https://linkedin.com/in/mathiasgomez-ds)
