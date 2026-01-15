#!/usr/bin/env python3
"""
Build Policy Lab data payload for GE-style counterfactuals.

Creates a compact JSON with baseline trade flows, export/import shares,
and country metadata. The Policy Lab UI runs counterfactual scenarios
client-side for fast iteration on the current sample.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Policy Lab data payload."
    )
    parser.add_argument(
        "--input",
        default="docs/data/baci_gravity_viz.parquet",
        help="Input parquet with trade flows and predictions.",
    )
    parser.add_argument(
        "--country-codes",
        default="/Users/ian/trade_data_warehouse/baci/country_codes.parquet",
        help="BACI country codes parquet (for names).",
    )
    parser.add_argument(
        "--out",
        default="docs/data/policy_lab.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["trade_pred"] = df["trade_pred"].clip(lower=1e-9)
    df["trade_value_usd_millions"] = df["trade_value_usd_millions"].clip(lower=0)

    countries = sorted(set(df["iso_o"]) | set(df["iso_d"]))
    idx = {c: i for i, c in enumerate(countries)}

    # Load country names (optional)
    name_map = {c: c for c in countries}
    try:
        codes = pd.read_parquet(args.country_codes)
        for row in codes.itertuples(index=False):
            iso3 = str(row.country_iso3)
            if iso3 in name_map:
                name_map[iso3] = str(row.country_name)
    except Exception:
        pass

    years = sorted(df["year"].unique())
    payload = {
        "meta": {
            "source": str(input_path),
            "years": [int(y) for y in years],
            "countries": countries,
            "country_names": name_map,
            "note": "Baseline uses predicted trade from the default model; sample size depends on the input dataset.",
        },
        "by_year": {}
    }

    for year in years:
        subset = df[df["year"] == year]
        n = len(countries)
        flows = np.zeros((n, n))
        flows_obs = np.zeros((n, n))
        for row in subset.itertuples(index=False):
            i = idx[row.iso_o]
            j = idx[row.iso_d]
            flows[i, j] = row.trade_pred
            flows_obs[i, j] = row.trade_value_usd_millions

        exports = flows.sum(axis=1)
        imports = flows.sum(axis=0)
        total = flows.sum()

        # Export shares (rows)
        export_shares = np.zeros_like(flows)
        for i in range(n):
            denom = exports[i] if exports[i] > 0 else 1.0
            export_shares[i, :] = flows[i, :] / denom

        # Import shares (columns)
        import_shares = np.zeros_like(flows)
        for j in range(n):
            denom = imports[j] if imports[j] > 0 else 1.0
            import_shares[:, j] = flows[:, j] / denom

        payload["by_year"][str(int(year))] = {
            "total_trade_pred": float(total),
            "total_trade_obs": float(flows_obs.sum()),
            "exports_pred": exports.tolist(),
            "imports_pred": imports.tolist(),
            "flows_pred": flows.tolist(),
            "export_shares": export_shares.tolist(),
            "import_shares": import_shares.tolist(),
        }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote Policy Lab payload to {out_path}")


if __name__ == "__main__":
    main()
