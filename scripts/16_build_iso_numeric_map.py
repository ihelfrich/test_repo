#!/usr/bin/env python3
"""
Build ISO numeric -> ISO3 mapping for client-side map centroids.

Reads BACI country codes and exports a compact JSON mapping so the
Leaflet map can join Natural Earth numeric IDs to ISO3 codes.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ISO numeric -> ISO3 mapping JSON."
    )
    parser.add_argument(
        "--input",
        default="/Users/ian/trade_data_warehouse/baci/country_codes.parquet",
        help="Input parquet with BACI country codes.",
    )
    parser.add_argument(
        "--out",
        default="docs/data/iso3_numeric_map.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df = df.dropna(subset=["country_code", "country_iso3"])

    mapping = {}
    for row in df.itertuples(index=False):
        try:
            code = int(row.country_code)
        except (TypeError, ValueError):
            continue
        mapping[code] = {
            "iso3": str(row.country_iso3),
            "iso2": str(row.country_iso2),
            "name": str(row.country_name),
        }

    payload = {
        "meta": {
            "source": str(input_path),
            "rows": len(mapping),
        },
        "codes": mapping,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote ISO numeric map to {out_path}")


if __name__ == "__main__":
    main()
