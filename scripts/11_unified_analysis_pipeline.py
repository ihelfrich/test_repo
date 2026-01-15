#!/usr/bin/env python3
"""
Unified Analysis Pipeline (Research Lab)

Builds a compact research summary for the web UI from the visualization dataset.
The pipeline is intentionally conservative: it computes metrics that are
transparent, reproducible, and explainable in the browser.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build research summary metrics for the Research Lab page."
    )
    parser.add_argument(
        "--input",
        default="docs/data/baci_gravity_viz.parquet",
        help="Input parquet for analysis.",
    )
    parser.add_argument(
        "--topology",
        default="docs/data/topology_fields.json",
        help="Topology fields JSON (optional).",
    )
    parser.add_argument(
        "--out",
        default="docs/data/research_summary.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def compute_hhi(series: pd.Series) -> float:
    total = series.sum()
    if total <= 0:
        return 0.0
    shares = series / total
    return float((shares ** 2).sum())


def count_components(binary: np.ndarray) -> int:
    visited = np.zeros(binary.shape, dtype=bool)
    count = 0
    height, width = binary.shape
    for i in range(height):
        for j in range(width):
            if not binary[i, j] or visited[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            visited[i, j] = True
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < height and 0 <= ny < width:
                        if binary[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
    return count


def compute_betti(grid: np.ndarray) -> dict:
    threshold = float(np.mean(grid))
    binary = grid > threshold
    beta0 = count_components(binary)
    inverse = ~binary
    background = count_components(inverse)
    beta1 = max(background - 1, 0)
    return {"beta0": int(beta0), "beta1": int(beta1)}


def load_topology_fields(path: Path) -> dict:
    if not path.exists():
        return {"meta": {}, "fields": {}}
    payload = json.loads(path.read_text())
    return payload


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if "trade_pred" not in df.columns:
        raise ValueError("Expected trade_pred column in visualization parquet.")

    df["log_trade"] = np.log1p(df["trade_value_usd_millions"])
    df["log_trade_pred"] = np.log1p(df["trade_pred"])
    df["log_trade_gap"] = df["log_trade"] - df["log_trade_pred"]

    years = sorted(df["year"].unique().tolist())
    metrics = {}
    prev_residuals = None

    topology_payload = load_topology_fields(Path(args.topology))
    fields_by_year = topology_payload.get("fields", {})

    for year in years:
        df_year = df[df["year"] == year]

        total_trade = float(df_year["trade_value_usd_millions"].sum())
        mean_ln_dist = float(df_year["ln_dist"].mean())
        contig_share = float(df_year["contig"].mean()) if "contig" in df_year else 0.0
        rta_share = float(df_year["rta_coverage"].mean()) if "rta_coverage" in df_year else 0.0

        residual_mean = float(df_year["log_trade_gap"].mean())
        residual_std = float(df_year["log_trade_gap"].std(ddof=0))

        exporter_totals = df_year.groupby("iso_o")["trade_value_usd_millions"].sum()
        importer_totals = df_year.groupby("iso_d")["trade_value_usd_millions"].sum()
        export_hhi = compute_hhi(exporter_totals)
        import_hhi = compute_hhi(importer_totals)

        residuals = df_year["log_trade_gap"].to_numpy()
        shift = 0.0
        if prev_residuals is not None:
            shift = float(wasserstein_distance(prev_residuals, residuals))
        prev_residuals = residuals

        field_variance = None
        beta0 = None
        beta1 = None
        field_payload = fields_by_year.get(str(year))
        if field_payload and "grid" in field_payload:
            grid = np.array(field_payload["grid"], dtype=float)
            field_variance = float(np.var(grid))
            betti = compute_betti(grid)
            beta0 = betti["beta0"]
            beta1 = betti["beta1"]

        metrics[str(year)] = {
            "total_trade": total_trade,
            "mean_ln_dist": mean_ln_dist,
            "contig_share": contig_share,
            "rta_share": rta_share,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "export_hhi": export_hhi,
            "import_hhi": import_hhi,
            "wasserstein_shift": shift,
            "field_variance": field_variance,
            "beta0": beta0,
            "beta1": beta1,
        }

    # Identify notable years
    def top_years(key: str, n: int = 3):
        series = [(int(year), data[key]) for year, data in metrics.items() if data[key] is not None]
        series = sorted(series, key=lambda item: item[1], reverse=True)
        return [year for year, _ in series[:n]]

    summary = {
        "largest_shifts": top_years("wasserstein_shift"),
        "highest_residual_dispersion": top_years("residual_std"),
        "highest_concentration": top_years("export_hhi"),
    }

    payload = {
        "meta": {
            "source": "BACI bilateral totals + CEPII gravity (visualization subset)",
            "years": years,
            "rows": int(len(df)),
            "topology_available": bool(fields_by_year),
            "notes": "Metrics computed on top-N dyads from visualization dataset; intended for exploratory use.",
        },
        "metrics": metrics,
        "summary": summary,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote research summary to {out_path}")


if __name__ == "__main__":
    main()
