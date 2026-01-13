import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a gravity PPML model using BACI and CEPII covariates."
    )
    parser.add_argument(
        "--baci-sample",
        default="data/processed/baci_sample.parquet",
        help="Input BACI sample parquet.",
    )
    parser.add_argument(
        "--gravity",
        default="/Users/ian/trade_data_warehouse/gravity/gravity_v202211.parquet",
        help="CEPII gravity covariates parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Base output directory for tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    query = f"""
        select
            b.year,
            b.iso_o,
            b.iso_d,
            b.trade_value_usd_millions,
            g.dist,
            g.contig,
            g.comlang_off,
            g.comcol,
            g.rta_coverage,
            g.gdp_o,
            g.gdp_d,
            g.pop_o,
            g.pop_d
        from read_parquet('{args.baci_sample}') b
        left join read_parquet('{args.gravity}') g
          on b.year = g.year
         and b.iso_o = g.iso3_o
         and b.iso_d = g.iso3_d
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

    for col in ["contig", "comlang_off", "comcol", "rta_coverage"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    formula = (
        "trade_value_usd_millions ~ ln_dist + contig + comlang_off + comcol "
        "+ ln_gdp_o + ln_gdp_d + ln_pop_o + ln_pop_d + rta_coverage "
        "+ C(year) + C(iso_o) + C(iso_d)"
    )

    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    result = model.fit(cov_type="HC1")

    summary_path = tables_dir / "ppml_summary.txt"
    with summary_path.open("w") as f:
        f.write(result.summary().as_text())

    coef_table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "p_value": result.pvalues.values,
        }
    )
    coef_table.to_csv(tables_dir / "ppml_coefficients.csv", index=False)

    print(f"Wrote PPML summary to {summary_path}")
    print(f"Wrote coefficients to {tables_dir / 'ppml_coefficients.csv'}")
    print(f"Rows used: {len(df)}")


if __name__ == "__main__":
    main()
