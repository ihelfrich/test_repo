# Project Status Report
**Topological Trade Dynamics Platform**

**Date**: 2026-01-15
**Status**: Prototype platform with data-backed visualizations
**Scope**: Gravity explorer + topology signals + research summary

---

## Current Capabilities (Implemented)

- **Gravity Explorer (Three.js)**
  - Multi-model selection (AvW PPML, Head-Mayer PPML, Year-FE PPML, AvW OLS).
  - Counterfactual sliders with model-consistent coefficients.
  - Dynamic model insight and dyad explanations.

- **Topology Signals (Field View)**
  - Data-driven residual fields by year (2005–2021, top-20 dyads).
  - Real diagnostics in-browser (energy, variance, autocorrelation, Betti proxies).
  - Shock controls and display modes.

- **Research Lab Summary**
  - Wasserstein-1 shift across years.
  - Residual dispersion and concentration (HHI).
  - Field variance + Betti proxies (if topology fields available).

---

## Data Coverage (Visualization Subset)

- **Years:** 2005–2021 (17 years)
- **Coverage:** Top 20 exporters × top 20 importers
- **Rows:** 5,848
- **Source:** BACI bilateral totals + CEPII Gravity v202211

---

## Scripts and Outputs

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/04_prepare_viz_data.py` | Build viz dataset + model predictions | `docs/data/baci_gravity_viz.*` |
| `scripts/09_build_topology_fields.py` | Grid residual fields for topology view | `docs/data/topology_fields.json` |
| `scripts/11_unified_analysis_pipeline.py` | Research summary metrics | `docs/data/research_summary.json` |

---

## Known Gaps (Planned)

- **Full dataset pipeline** (`data/processed/baci_gravity_full.parquet`) not built yet.
- **Advanced methods** (GNN, causal DAG, RL, Hodge, stochastic dynamics) are research agenda items, not yet implemented or validated in this repository.
- **Formal statistical validation** of topology signals is pending.

---

## Next Actions

1. Expand dataset to full country coverage and confirm memory/runtime constraints.
2. Validate topology signals against known shocks (COVID, tariff episodes).
3. Add model comparison dashboard with formal robustness checks.
4. Decide which advanced method to implement first (OT or graph-based metrics).

---

## Project Links

- **Gravity Explorer:** `docs/index.html`
- **Topology Signals:** `docs/topology.html`
- **Research Lab:** `docs/advanced_topology.html`
- **Methodology:** `docs/methodology.html`
- **Technical Spec (roadmap):** `docs/TECHNICAL_SPEC.md`
