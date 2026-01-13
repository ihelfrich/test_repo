import argparse
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a limited BACI bilateral totals subsample."
    )
    parser.add_argument(
        "--source",
        default="/Users/ian/trade_data_warehouse/baci/baci_bilateral_totals.parquet",
        help="Path to BACI bilateral totals parquet.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/baci_sample.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    query = f"""
        select year, iso_o, iso_d, trade_value_usd_millions
        from read_parquet('{args.source}')
        where year between {args.start_year} and {args.end_year}
    """
    df = con.execute(query).df()
    df.to_parquet(out_path, index=False)

    n_rows = len(df)
    n_exporters = df["iso_o"].nunique()
    n_importers = df["iso_d"].nunique()
    years = sorted(df["year"].unique().tolist())

    print(f"Wrote sample to {out_path}")
    print(f"Rows: {n_rows}, exporters: {n_exporters}, importers: {n_importers}")
    print(f"Years: {years}")


if __name__ == "__main__":
    main()
