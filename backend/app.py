import os
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ge import build_tau_hat, solve_ge


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "docs" / "data" / "baci_gravity_viz.parquet"
FLOW_PARQUET = Path(os.getenv("FLOW_PARQUET", DEFAULT_PARQUET))

app = FastAPI(title="Gravity Trade API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONN = duckdb.connect()
FLOW_CACHE = {}


class Shock(BaseModel):
    type: str = Field(..., description="Shock type: global_tariff, import_tariff, bilateral_tariff, rta")
    rate: float = Field(..., description="Ad valorem rate (e.g., 0.1 for 10%)")
    target: Optional[str] = Field(None, description="Target country ISO3")
    partner: Optional[str] = Field(None, description="Partner country ISO3 (for bilateral/RTA)")


class GERequest(BaseModel):
    year: int
    theta: float = 5.0
    shocks: List[Shock] = Field(default_factory=list)


def _read_flows(year: int):
    if year in FLOW_CACHE:
        return FLOW_CACHE[year]

    if not FLOW_PARQUET.exists():
        raise FileNotFoundError(f"Flow parquet not found: {FLOW_PARQUET}")

    try:
        df = CONN.execute(
            "SELECT year, iso_o, iso_d, trade_pred AS trade FROM read_parquet(?) WHERE year = ?",
            [str(FLOW_PARQUET), year],
        ).df()
    except Exception:
        df = CONN.execute(
            "SELECT year, iso_o, iso_d, trade_value_usd_millions AS trade FROM read_parquet(?) WHERE year = ?",
            [str(FLOW_PARQUET), year],
        ).df()

    countries = sorted(set(df["iso_o"]).union(set(df["iso_d"])))
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)
    flows = np.zeros((n, n), dtype=float)
    for row in df.itertuples(index=False):
        i = idx[row.iso_o]
        j = idx[row.iso_d]
        flows[i, j] = float(row.trade)

    FLOW_CACHE[year] = (countries, flows)
    return countries, flows


@app.get("/health")
def health():
    return {"status": "ok", "flow_parquet": str(FLOW_PARQUET)}


@app.get("/years")
def years():
    rows = CONN.execute(
        "SELECT DISTINCT year FROM read_parquet(?) ORDER BY year",
        [str(FLOW_PARQUET)],
    ).fetchall()
    return {"years": [int(r[0]) for r in rows]}


@app.get("/countries")
def countries():
    rows = CONN.execute(
        """
        SELECT DISTINCT iso_o AS iso FROM read_parquet(?)
        UNION DISTINCT
        SELECT DISTINCT iso_d AS iso FROM read_parquet(?)
        ORDER BY iso
        """,
        [str(FLOW_PARQUET), str(FLOW_PARQUET)],
    ).fetchall()
    return {"countries": [r[0] for r in rows]}


@app.get("/flows")
def flows(
    year: int = Query(..., description="Year to filter"),
    limit: int = Query(500, description="Max rows to return"),
):
    try:
        df = CONN.execute(
            """
            SELECT year, iso_o, iso_d, trade_value_usd_millions
            FROM read_parquet(?)
            WHERE year = ?
            ORDER BY trade_value_usd_millions DESC
            LIMIT ?
            """,
            [str(FLOW_PARQUET), year, limit],
        ).df()
    except Exception:
        df = CONN.execute(
            """
            SELECT year, iso_o, iso_d, trade_pred AS trade_value_usd_millions
            FROM read_parquet(?)
            WHERE year = ?
            ORDER BY trade_value_usd_millions DESC
            LIMIT ?
            """,
            [str(FLOW_PARQUET), year, limit],
        ).df()
    return {"rows": df.to_dict(orient="records")}


@app.post("/ge_counterfactual")
def ge_counterfactual(payload: GERequest = Body(...)):
    countries, flows = _read_flows(payload.year)
    shocks = [shock.dict() for shock in payload.shocks]
    tau_hat = build_tau_hat(countries, shocks)
    result = solve_ge(flows, theta=payload.theta, tau_hat=tau_hat)

    response = {
        "year": payload.year,
        "theta": payload.theta,
        "countries": countries,
        "exports_base": result["exports_base"].tolist(),
        "imports_base": result["imports_base"].tolist(),
        "exports_cf": result["exports_cf"].tolist(),
        "imports_cf": result["imports_cf"].tolist(),
        "flows_base": result["flows_base"].tolist(),
        "flows_cf": result["flows_cf"].tolist(),
        "welfare": result["welfare"].tolist(),
        "converged": result["converged"],
        "iterations": result["iterations"],
        "max_diff": result["max_diff"],
    }
    return response
