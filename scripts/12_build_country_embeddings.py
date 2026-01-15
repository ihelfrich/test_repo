#!/usr/bin/env python3
"""
Build country embeddings from gravity distances for 3D trade views.

Outputs a JSON file with a 2D MDS embedding and a derived 3D sphere mapping.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build country embeddings for 3D trade views."
    )
    parser.add_argument(
        "--input",
        default="docs/data/baci_gravity_viz.parquet",
        help="Input parquet with ln_dist for country pairs.",
    )
    parser.add_argument(
        "--out",
        default="docs/data/country_embedding.json",
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


def normalize(coords: np.ndarray) -> np.ndarray:
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    span = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    return (coords - min_vals) / span


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    countries = sorted(set(df["iso_o"]) | set(df["iso_d"]))
    dist_matrix = build_distance_matrix(df, countries)

    coords = classical_mds(dist_matrix, n_components=2)
    coords = normalize(coords)

    embeddings = {}
    for iso, (x, y) in zip(countries, coords):
        lon = (x - 0.5) * 2 * np.pi
        lat = (y - 0.5) * np.pi
        radius = 60.0
        x3 = radius * np.cos(lat) * np.cos(lon)
        y3 = radius * np.sin(lat)
        z3 = radius * np.cos(lat) * np.sin(lon)
        embeddings[iso] = {
            "x2": float(x),
            "y2": float(y),
            "lon": float(lon),
            "lat": float(lat),
            "x": float(x3),
            "y": float(y3),
            "z": float(z3),
        }

    payload = {
        "meta": {
            "source": "BACI viz subset (distance-based MDS embedding)",
            "countries": len(countries),
            "radius": 60.0,
            "note": "This is a trade-distance embedding, not geographic coordinates.",
        },
        "embeddings": embeddings,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote country embedding to {out_path}")


if __name__ == "__main__":
    main()
