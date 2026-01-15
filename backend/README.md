# Gravity Trade API (FastAPI + DuckDB)

This backend serves full‑sample queries and GE counterfactuals for the Policy Lab.
It is optional for GitHub Pages, but required for large‑N runs and GE simulations.

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Default uses docs/data/baci_gravity_viz.parquet
uvicorn backend.app:app --reload --port 8000
```

## Full‑sample data

Set `FLOW_PARQUET` to point at your full dataset:

```bash
export FLOW_PARQUET=/Users/ian/trade_data_warehouse/baci/baci_bilateral_totals.parquet
uvicorn backend.app:app --reload --port 8000
```

## Endpoints

- `GET /health`
- `GET /years`
- `GET /countries`
- `GET /flows?year=2021&limit=500`
- `POST /ge_counterfactual`

### Example GE request

```json
{
  "year": 2021,
  "theta": 5,
  "shocks": [
    { "type": "global_tariff", "rate": 0.10 },
    { "type": "bilateral_tariff", "target": "USA", "partner": "CHN", "rate": 0.60 }
  ]
}
```

The response includes baseline and counterfactual flows, exports/imports, and welfare ratios.
