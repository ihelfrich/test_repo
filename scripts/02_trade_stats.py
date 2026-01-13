import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate general trade statistics from a BACI sample."
    )
    parser.add_argument(
        "--input",
        default="data/processed/baci_sample.parquet",
        help="Input BACI sample parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Base output directory for tables/figures.",
    )
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    value_col = "trade_value_usd_millions"

    summary = pd.DataFrame(
        {
            "rows": [len(df)],
            "years": [df["year"].nunique()],
            "exporters": [df["iso_o"].nunique()],
            "importers": [df["iso_d"].nunique()],
            "total_trade_value": [df[value_col].sum()],
            "mean_trade_value": [df[value_col].mean()],
            "median_trade_value": [df[value_col].median()],
            "p10_trade_value": [df[value_col].quantile(0.10)],
            "p90_trade_value": [df[value_col].quantile(0.90)],
            "zero_share": [(df[value_col] == 0).mean()],
        }
    )
    summary.to_csv(tables_dir / "trade_summary_stats.csv", index=False)

    trade_by_year = df.groupby("year", as_index=False)[value_col].sum()
    trade_by_year.to_csv(tables_dir / "trade_by_year.csv", index=False)

    top_exporters = (
        df.groupby("iso_o", as_index=False)[value_col].sum()
        .sort_values(value_col, ascending=False)
        .head(args.top_n)
    )
    top_exporters.to_csv(tables_dir / "top_exporters.csv", index=False)

    top_importers = (
        df.groupby("iso_d", as_index=False)[value_col].sum()
        .sort_values(value_col, ascending=False)
        .head(args.top_n)
    )
    top_importers.to_csv(tables_dir / "top_importers.csv", index=False)

    top_pairs = (
        df.groupby(["iso_o", "iso_d"], as_index=False)[value_col].sum()
        .sort_values(value_col, ascending=False)
        .head(args.top_n)
    )
    top_pairs.to_csv(tables_dir / "top_pairs.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.plot(trade_by_year["year"], trade_by_year[value_col], marker="o")
    plt.title("Total Trade Value by Year")
    plt.xlabel("Year")
    plt.ylabel("Trade value (USD, millions)")
    plt.tight_layout()
    plt.savefig(figures_dir / "trade_by_year.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(top_exporters["iso_o"], top_exporters[value_col])
    plt.title(f"Top {args.top_n} Exporters by Trade Value")
    plt.xlabel("Exporter (ISO3)")
    plt.ylabel("Trade value (USD, millions)")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_exporters.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(top_importers["iso_d"], top_importers[value_col])
    plt.title(f"Top {args.top_n} Importers by Trade Value")
    plt.xlabel("Importer (ISO3)")
    plt.ylabel("Trade value (USD, millions)")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_importers.png", dpi=150)
    plt.close()

    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
