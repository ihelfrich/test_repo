import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import statsmodels.api as sm
import statsmodels.formula.api as smf

MODEL_SPECS = [
    {
        "key": "avw_ppml",
        "label": "Anderson-van Wincoop (2003) PPML with exporter/importer FE",
        "kind": "glm",
        "link": "log",
        "formula": (
            "trade_value_usd_millions ~ ln_dist + contig + comlang_off + comcol "
            "+ ln_gdp_o + ln_gdp_d + ln_pop_o + ln_pop_d + rta_coverage "
            "+ C(year) + C(iso_o) + C(iso_d)"
        ),
        "notes": "Baseline structural gravity with multilateral resistance (exporter/importer FE).",
    },
    {
        "key": "hm_ppml",
        "label": "Head-Mayer (2014) PPML with exporter-year & importer-year FE",
        "kind": "glm",
        "link": "log",
        "formula": (
            "trade_value_usd_millions ~ ln_dist + contig + comlang_off + comcol "
            "+ rta_coverage + C(iso_o):C(year) + C(iso_d):C(year)"
        ),
        "notes": "Time-varying multilateral resistance via exporter-year/importer-year FE.",
    },
    {
        "key": "year_fe_ppml",
        "label": "Reduced-form PPML with year FE only",
        "kind": "glm",
        "link": "log",
        "formula": (
            "trade_value_usd_millions ~ ln_dist + contig + comlang_off + comcol "
            "+ ln_gdp_o + ln_gdp_d + ln_pop_o + ln_pop_d + rta_coverage + C(year)"
        ),
        "notes": "Parsimonious specification for quick comparisons (no country FE).",
    },
    {
        "key": "avw_ols",
        "label": "AvW OLS on log1p trade with exporter/importer FE",
        "kind": "ols",
        "link": "log1p",
        "formula": (
            "log_trade_value ~ ln_dist + contig + comlang_off + comcol "
            "+ ln_gdp_o + ln_gdp_d + ln_pop_o + ln_pop_d + rta_coverage "
            "+ C(year) + C(iso_o) + C(iso_d)"
        ),
        "notes": "Log-linear benchmark (OLS on log1p trade).",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a BACI + gravity subsample for interactive visualization."
    )
    parser.add_argument(
        "--baci",
        default="/Users/ian/trade_data_warehouse/baci/baci_bilateral_totals.parquet",
        help="Path to BACI bilateral totals parquet.",
    )
    parser.add_argument(
        "--gravity",
        default="/Users/ian/trade_data_warehouse/gravity/gravity_v202211.parquet",
        help="Path to CEPII gravity parquet.",
    )
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys to include (default: all).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=120,
        help="Maximum iterations for GLM estimation.",
    )
    parser.add_argument(
        "--out-parquet",
        default="docs/data/baci_gravity_viz.parquet",
        help="Output parquet path (columnar).",
    )
    parser.add_argument(
        "--out-json",
        default="docs/data/baci_gravity_viz.json",
        help="Output JSON path for web visualization.",
    )
    parser.add_argument(
        "--out-arrow",
        default="docs/data/baci_gravity_viz.arrow",
        help="Output Arrow IPC path for browser-native loading.",
    )
    return parser.parse_args()


def _safe_float(value: float) -> float:
    try:
        if value is None or not np.isfinite(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _predict_linear(result, df: pd.DataFrame) -> np.ndarray:
    try:
        return result.predict(df, which="linear")
    except TypeError:
        try:
            return result.predict(df, linear=True)
        except TypeError:
            return result.predict(df)


def _fit_model(spec: dict, df: pd.DataFrame, max_iter: int) -> tuple:
    if spec["kind"] == "ols":
        model = smf.ols(formula=spec["formula"], data=df)
        result = model.fit()
        base_eta = result.predict(df)
        trade_pred = np.expm1(base_eta)
        trade_pred = np.where(np.isfinite(trade_pred), trade_pred, 0.0)
        trade_pred = np.clip(trade_pred, 0, None)
        return result, trade_pred, base_eta

    model = smf.glm(formula=spec["formula"], data=df, family=sm.families.Poisson())
    result = model.fit(maxiter=max_iter, disp=False)
    trade_pred = result.predict(df)
    base_eta = _predict_linear(result, df)
    return result, trade_pred, base_eta


def main() -> None:
    args = parse_args()
    out_parquet = Path(args.out_parquet)
    out_json = Path(args.out_json)
    out_arrow = Path(args.out_arrow)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_arrow.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    query = f"""
        with base as (
            select year, iso_o, iso_d, trade_value_usd_millions
            from read_parquet('{args.baci}')
            where year between {args.start_year} and {args.end_year}
        ),
        top_exporters as (
            select iso_o, sum(trade_value_usd_millions) as total_trade
            from base
            group by iso_o
            order by total_trade desc
            limit {args.top_n}
        ),
        top_importers as (
            select iso_d, sum(trade_value_usd_millions) as total_trade
            from base
            group by iso_d
            order by total_trade desc
            limit {args.top_n}
        ),
        filtered as (
            select *
            from base
            where iso_o in (select iso_o from top_exporters)
              and iso_d in (select iso_d from top_importers)
        )
        select
            f.year,
            f.iso_o,
            f.iso_d,
            f.trade_value_usd_millions,
            g.dist,
            g.contig,
            g.comlang_off,
            g.comcol,
            g.rta_coverage,
            g.gdp_o,
            g.gdp_d,
            g.pop_o,
            g.pop_d
        from filtered f
        left join read_parquet('{args.gravity}') g
          on f.year = g.year
         and f.iso_o = g.iso3_o
         and f.iso_d = g.iso3_d
    """
    df = con.execute(query).df()

    required = [
        "trade_value_usd_millions",
        "dist",
        "gdp_o",
        "gdp_d",
        "pop_o",
        "pop_d",
    ]
    df = df.dropna(subset=required)
    for col in ["dist", "gdp_o", "gdp_d", "pop_o", "pop_d"]:
        df = df[df[col] > 0]

    df["ln_dist"] = np.log(df["dist"])
    df["ln_gdp_o"] = np.log(df["gdp_o"])
    df["ln_gdp_d"] = np.log(df["gdp_d"])
    df["ln_pop_o"] = np.log(df["pop_o"])
    df["ln_pop_d"] = np.log(df["pop_d"])
    df["ln_gdp_prod"] = df["ln_gdp_o"] + df["ln_gdp_d"]
    df["log_trade_value"] = np.log1p(df["trade_value_usd_millions"])

    for col in ["contig", "comlang_off", "comcol", "rta_coverage"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    model_results = {}
    failed_models = []
    selected_models = {spec["key"] for spec in MODEL_SPECS}
    if args.models != "all":
        selected_models = {key.strip() for key in args.models.split(",") if key.strip()}

    for spec in MODEL_SPECS:
        if spec["key"] not in selected_models:
            continue
        try:
            result, trade_pred, base_eta = _fit_model(spec, df, args.max_iter)
            model_results[spec["key"]] = {
                "result": result,
                "trade_pred": trade_pred,
                "base_eta": base_eta,
                "label": spec["label"],
                "notes": spec["notes"],
                "link": spec["link"],
            }
        except Exception as exc:
            failed_models.append((spec["key"], str(exc)))

    if not model_results:
        raise RuntimeError(f"All model fits failed: {failed_models}")

    default_model_key = "avw_ppml"
    if default_model_key not in model_results:
        default_model_key = list(model_results.keys())[0]

    df["log_trade"] = np.log1p(df["trade_value_usd_millions"])

    for key, info in model_results.items():
        df[f"trade_pred_{key}"] = info["trade_pred"]
        df[f"base_eta_{key}"] = info["base_eta"]

    df["trade_pred"] = df[f"trade_pred_{default_model_key}"]
    df["base_eta"] = df[f"base_eta_{default_model_key}"]
    df["log_trade_pred"] = np.log1p(df["trade_pred"])
    df["log_trade_gap"] = df["log_trade"] - df["log_trade_pred"]

    keep_cols = [
        "year",
        "iso_o",
        "iso_d",
        "trade_value_usd_millions",
        "trade_pred",
        "base_eta",
        "log_trade",
        "log_trade_pred",
        "log_trade_gap",
        "ln_dist",
        "ln_gdp_o",
        "ln_gdp_d",
        "ln_gdp_prod",
        "contig",
        "comlang_off",
        "comcol",
        "rta_coverage",
    ]
    for key in model_results:
        keep_cols.extend([f"trade_pred_{key}", f"base_eta_{key}"])

    df_out = df[keep_cols].copy()

    df_out.to_parquet(out_parquet, index=False)
    table = pa.Table.from_pandas(df_out)
    with ipc.new_file(out_arrow, table.schema) as writer:
        writer.write(table)

    coef_terms = ["ln_dist", "contig", "comlang_off", "comcol", "rta_coverage"]
    model_meta = {}
    for key, info in model_results.items():
        result = info["result"]
        coefficients = {term: _safe_float(result.params.get(term, 0.0)) for term in coef_terms}
        model_meta[key] = {
            "label": info["label"],
            "notes": info["notes"],
            "link": info["link"],
            "coefficients": coefficients,
        }

    payload = {
        "meta": {
            "source": "BACI bilateral totals + CEPII gravity v202211",
            "years": sorted(df_out["year"].unique().tolist()),
            "top_n": args.top_n,
            "model": model_meta[default_model_key]["label"],
            "default_model": default_model_key,
            "models": model_meta,
            "notes": "Predictions from multiple gravity specifications; log variables are log(1+x).",
            "coefficients": model_meta[default_model_key]["coefficients"],
        },
        "rows": df_out.to_dict(orient="records"),
    }
    with out_json.open("w") as f:
        json.dump(payload, f)

    print(f"Wrote parquet to {out_parquet}")
    print(f"Wrote Arrow IPC to {out_arrow}")
    print(f"Wrote JSON to {out_json}")
    print(f"Rows: {len(df_out)}")


if __name__ == "__main__":
    main()
