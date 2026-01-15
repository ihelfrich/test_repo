#!/usr/bin/env python3
"""
Build data-driven topology fields for the browser demo.

Creates per-year 2D fields by gridding dyad residuals onto a fixed grid.
The output is a compact JSON payload for docs/topology.html.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build topology field payload from visualization parquet."
    )
    parser.add_argument(
        "--input",
        default="docs/data/baci_gravity_viz.parquet",
        help="Input parquet with gravity predictions.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Grid resolution for the field (default: 64).",
    )
    parser.add_argument(
        "--out",
        default="docs/data/topology_fields.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def classical_mds(dist_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    n = dist_matrix.shape[0]
    dist_sq = dist_matrix ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ dist_sq @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals[:n_components], 0)
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals)
    return coords


def build_distance_matrix(df: pd.DataFrame, countries: list) -> np.ndarray:
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)
    mat = np.full((n, n), np.nan)

    grouped = df.groupby(["iso_o", "iso_d"])["ln_dist"].median().reset_index()
    for row in grouped.itertuples(index=False):
        i = idx[row.iso_o]
        j = idx[row.iso_d]
        mat[i, j] = row.ln_dist
        mat[j, i] = row.ln_dist

    np.fill_diagonal(mat, 0.0)
    mean_val = np.nanmean(mat)
    mat = np.where(np.isnan(mat), mean_val, mat)
    return mat


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    span = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return (coords - min_vals) / span


def blur_grid(grid: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(grid, 1, mode="wrap")
    out = np.zeros_like(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            window = padded[i : i + 3, j : j + 3]
            out[i, j] = np.sum(window * kernel)
    return out


def build_field(df_year: pd.DataFrame, coords: np.ndarray, countries: list, grid_size: int) -> np.ndarray:
    idx = {c: i for i, c in enumerate(countries)}
    grid = np.zeros((grid_size, grid_size), dtype=float)

    for row in df_year.itertuples(index=False):
        i = idx[row.iso_o]
        j = idx[row.iso_d]
        mid = (coords[i] + coords[j]) / 2
        gx = mid[0] * (grid_size - 1)
        gy = mid[1] * (grid_size - 1)
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, grid_size - 1)
        y1 = min(y0 + 1, grid_size - 1)
        wx = gx - x0
        wy = gy - y0
        val = row.log_trade_gap

        grid[y0, x0] += val * (1 - wx) * (1 - wy)
        grid[y0, x1] += val * wx * (1 - wy)
        grid[y1, x0] += val * (1 - wx) * wy
        grid[y1, x1] += val * wx * wy

    return blur_grid(grid)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if "log_trade_gap" not in df.columns:
        df["log_trade"] = np.log1p(df["trade_value_usd_millions"])
        df["log_trade_pred"] = np.log1p(df["trade_pred"])
        df["log_trade_gap"] = df["log_trade"] - df["log_trade_pred"]

    countries = sorted(set(df["iso_o"]).union(set(df["iso_d"])))
    dist_matrix = build_distance_matrix(df, countries)
    coords = classical_mds(dist_matrix, n_components=2)
    coords = normalize_coords(coords)

    years = sorted(df["year"].unique().tolist())
    fields = {}
    for year in years:
        df_year = df[df["year"] == year]
        grid = build_field(df_year, coords, countries, args.grid_size)
        fields[str(year)] = {
            "grid": np.round(grid, 6).tolist(),
            "min": float(grid.min()),
            "max": float(grid.max()),
            "mean": float(grid.mean()),
            "variance": float(grid.var()),
        }

    payload = {
        "meta": {
            "source": "BACI bilateral totals + CEPII gravity (visualization subset)",
            "grid_size": args.grid_size,
            "years": years,
            "countries": len(countries),
            "field": "log_trade_gap (actual - predicted, log1p)",
            "method": "Midpoint gridding with bilinear weights + Gaussian blur",
        },
        "fields": fields,
    }

    out_path.write_text(json.dumps(payload))
    print(f"Wrote topology fields to {out_path} ({len(years)} years)")


if __name__ == "__main__":
    main()
