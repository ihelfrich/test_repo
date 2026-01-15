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

    for col in ["contig", "comlang_off", "comcol", "rta_coverage"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Anderson-van Wincoop structure with exporter/importer FE (multilateral resistance).
    formula = (
        "trade_value_usd_millions ~ ln_dist + contig + comlang_off + comcol "
        "+ ln_gdp_o + ln_gdp_d + ln_pop_o + ln_pop_d + rta_coverage "
        "+ C(year) + C(iso_o) + C(iso_d)"
    )

    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    result = model.fit()

    df["trade_pred"] = result.predict(df)
    try:
        df["base_eta"] = result.predict(df, which="linear")
    except TypeError:
        df["base_eta"] = result.predict(df, linear=True)
    df["log_trade"] = np.log1p(df["trade_value_usd_millions"])
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
    df_out = df[keep_cols].copy()

    df_out.to_parquet(out_parquet, index=False)
    table = pa.Table.from_pandas(df_out)
    with ipc.new_file(out_arrow, table.schema) as writer:
        writer.write(table)

    coef_terms = [
        "ln_dist",
        "contig",
        "comlang_off",
        "comcol",
        "rta_coverage",
        "ln_gdp_o",
        "ln_gdp_d",
        "ln_pop_o",
        "ln_pop_d",
    ]
    coefficients = {term: float(result.params.get(term, 0.0)) for term in coef_terms}

    payload = {
        "meta": {
            "source": "BACI bilateral totals + CEPII gravity v202211",
            "years": sorted(df_out["year"].unique().tolist()),
            "top_n": args.top_n,
            "model": "Anderson-van Wincoop (2003) gravity with exporter/importer FE (PPML)",
            "notes": "Predictions from Poisson GLM with fixed effects; log variables are log(1+x).",
            "coefficients": coefficients,
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
