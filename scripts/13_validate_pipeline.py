#!/usr/bin/env python3
"""
Pipeline Validation and Testing Suite

Validates all data outputs, checks consistency, and generates a validation report.
This ensures the entire pipeline is working correctly and data quality is maintained.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete data pipeline and outputs."
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory of the project (default: current directory).",
    )
    parser.add_argument(
        "--out",
        default="validation_report.json",
        help="Output validation report path.",
    )
    return parser.parse_args()


class PipelineValidator:
    """Comprehensive validation of the trade analysis pipeline."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.docs_data_dir = self.base_dir / "docs" / "data"
        self.results = {
            "status": "unknown",
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_total": 0,
            "errors": [],
            "warnings": [],
            "data_summary": {},
            "validation_details": {}
        }

    def run_test(self, name: str, func) -> bool:
        """Run a single validation test and track results."""
        self.results["tests_total"] += 1
        try:
            result = func()
            if result:
                self.results["tests_passed"] += 1
                print(f"✅ {name}")
                return True
            else:
                self.results["tests_failed"] += 1
                self.results["errors"].append(f"Test failed: {name}")
                print(f"❌ {name}")
                return False
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"{name}: {str(e)}")
            print(f"❌ {name}: {str(e)}")
            return False

    def validate_file_exists(self, path: Path, description: str) -> bool:
        """Check if a file exists."""
        if not path.exists():
            self.results["warnings"].append(f"Missing file: {description} at {path}")
            return False
        return True

    def test_viz_parquet(self) -> bool:
        """Validate visualization parquet file."""
        path = self.docs_data_dir / "baci_gravity_viz.parquet"
        if not self.validate_file_exists(path, "Visualization parquet"):
            return False

        df = pd.read_parquet(path)

        # Check required columns
        required_cols = ["iso_o", "iso_d", "year", "trade_value_usd_millions",
                        "ln_dist", "trade_pred"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.results["errors"].append(f"Missing columns in viz parquet: {missing}")
            return False

        # Check for valid data
        if df["trade_value_usd_millions"].isna().all():
            self.results["errors"].append("All trade values are NaN")
            return False

        if (df["trade_value_usd_millions"] < 0).any():
            self.results["warnings"].append("Negative trade values found")

        # Store summary
        self.results["data_summary"]["viz_parquet"] = {
            "rows": len(df),
            "years": sorted(df["year"].unique().tolist()),
            "countries": len(set(df["iso_o"]) | set(df["iso_d"])),
            "total_trade_usd_millions": float(df["trade_value_usd_millions"].sum()),
            "avg_residual": float((df["trade_value_usd_millions"] - df["trade_pred"]).mean())
        }

        return True

    def test_topology_fields(self) -> bool:
        """Validate topology fields JSON."""
        path = self.docs_data_dir / "topology_fields.json"
        if not self.validate_file_exists(path, "Topology fields"):
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        # Check structure
        if "meta" not in data:
            self.results["errors"].append("Topology fields missing 'meta'")
            return False

        if "fields" not in data:
            self.results["errors"].append("Topology fields missing 'fields'")
            return False

        # Check field data
        fields = data["fields"]
        if not fields:
            self.results["warnings"].append("No field data in topology_fields.json")
            return False

        # Validate a sample field
        sample_year = list(fields.keys())[0]
        sample_field = fields[sample_year]

        if "grid" not in sample_field:
            self.results["errors"].append(f"Field for {sample_year} missing 'grid'")
            return False

        grid = np.array(sample_field["grid"])
        if grid.ndim != 2:
            self.results["errors"].append(f"Field grid is not 2D: shape {grid.shape}")
            return False

        # Store summary
        self.results["data_summary"]["topology_fields"] = {
            "years": sorted([int(y) for y in fields.keys()]),
            "grid_size": grid.shape[0],
            "num_fields": len(fields),
            "sample_field_range": [float(grid.min()), float(grid.max())],
            "sample_field_mean": float(grid.mean())
        }

        return True

    def test_research_summary(self) -> bool:
        """Validate research summary JSON."""
        path = self.docs_data_dir / "research_summary.json"
        if not self.validate_file_exists(path, "Research summary"):
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        # Check structure
        required_keys = ["meta", "metrics", "summary"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            self.results["errors"].append(f"Research summary missing keys: {missing}")
            return False

        # Validate metrics
        metrics = data["metrics"]
        if not metrics:
            self.results["errors"].append("No metrics in research_summary")
            return False

        # Check a sample year
        sample_year = list(metrics.keys())[0]
        sample_metrics = metrics[sample_year]

        expected_metrics = ["total_trade", "residual_std", "wasserstein_shift"]
        missing_metrics = [m for m in expected_metrics if m not in sample_metrics]
        if missing_metrics:
            self.results["warnings"].append(f"Missing metrics for {sample_year}: {missing_metrics}")

        # Store summary
        self.results["data_summary"]["research_summary"] = {
            "years": sorted([int(y) for y in metrics.keys()]),
            "num_metrics_per_year": len(sample_metrics),
            "largest_shifts": data["summary"].get("largest_shifts", [])
        }

        return True

    def test_country_embedding(self) -> bool:
        """Validate country embedding JSON."""
        path = self.docs_data_dir / "country_embedding.json"
        if not self.validate_file_exists(path, "Country embedding"):
            return False

        with open(path, 'r') as f:
            data = json.load(f)

        countries = None
        embedding = None

        if "countries" in data and "embedding_2d" in data:
            countries = data["countries"]
            embedding = np.array(data["embedding_2d"])
        elif "embeddings" in data:
            countries = sorted(data["embeddings"].keys())
            embedding = np.array([
                [data["embeddings"][iso]["x2"], data["embeddings"][iso]["y2"]]
                for iso in countries
            ])
        else:
            self.results["errors"].append(
                "Country embedding missing expected keys ('countries' + 'embedding_2d' or 'embeddings')"
            )
            return False

        # Validate dimensions
        if len(countries) != embedding.shape[0]:
            self.results["errors"].append(
                f"Mismatch: {len(countries)} countries but {embedding.shape[0]} embeddings"
            )
            return False

        if embedding.shape[1] != 2:
            self.results["errors"].append(f"Embedding is not 2D: shape {embedding.shape}")
            return False

        # Store summary
        self.results["data_summary"]["country_embedding"] = {
            "num_countries": len(countries),
            "embedding_shape": list(embedding.shape),
            "embedding_range": [
                [float(embedding[:, 0].min()), float(embedding[:, 0].max())],
                [float(embedding[:, 1].min()), float(embedding[:, 1].max())]
            ]
        }

        return True

    def test_data_consistency(self) -> bool:
        """Test consistency across datasets."""
        # Load all data
        viz_path = self.docs_data_dir / "baci_gravity_viz.parquet"
        topo_path = self.docs_data_dir / "topology_fields.json"
        summary_path = self.docs_data_dir / "research_summary.json"

        if not all(p.exists() for p in [viz_path, topo_path, summary_path]):
            self.results["warnings"].append("Cannot test consistency: missing files")
            return True  # Don't fail, just warn

        df = pd.read_parquet(viz_path)
        with open(topo_path, 'r') as f:
            topo = json.load(f)
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        # Check year consistency
        viz_years = set(df["year"].unique())
        topo_years = set(int(y) for y in topo["fields"].keys())
        summary_years = set(int(y) for y in summary["metrics"].keys())

        if viz_years != summary_years:
            self.results["warnings"].append(
                f"Year mismatch: viz has {viz_years}, summary has {summary_years}"
            )

        # Topology may have fewer years (that's okay for a subset)
        if not topo_years.issubset(viz_years):
            self.results["warnings"].append(
                f"Topology years {topo_years} not subset of viz years {viz_years}"
            )

        # Check trade totals consistency
        for year in viz_years & summary_years:
            viz_total = df[df["year"] == year]["trade_value_usd_millions"].sum()
            summary_total = summary["metrics"][str(year)]["total_trade"]

            rel_diff = abs(viz_total - summary_total) / (viz_total + 1e-10)
            if rel_diff > 0.01:  # More than 1% difference
                self.results["warnings"].append(
                    f"Trade total mismatch for {year}: viz={viz_total:.1f}, summary={summary_total:.1f}"
                )

        return True

    def test_field_statistics(self) -> bool:
        """Validate field statistics are reasonable."""
        path = self.docs_data_dir / "topology_fields.json"
        if not path.exists():
            return True  # Skip if file doesn't exist

        with open(path, 'r') as f:
            data = json.load(f)

        fields = data["fields"]

        for year, field_data in fields.items():
            if "grid" not in field_data:
                continue

            grid = np.array(field_data["grid"])

            # Check for NaN/Inf
            if np.isnan(grid).any():
                self.results["warnings"].append(f"NaN values in {year} field")

            if np.isinf(grid).any():
                self.results["errors"].append(f"Inf values in {year} field")
                return False

            # Check variance (should have some variation)
            variance = np.var(grid)
            if variance < 1e-10:
                self.results["warnings"].append(f"Field for {year} has very low variance: {variance}")

        return True

    def test_network_metrics(self) -> bool:
        """Validate network metrics JSON if present."""
        path = self.docs_data_dir / "network_metrics.json"
        if not path.exists():
            self.results["warnings"].append("Network metrics not found")
            return True

        with open(path, 'r') as f:
            data = json.load(f)

        if "meta" not in data or "by_year" not in data:
            self.results["errors"].append("Network metrics missing 'meta' or 'by_year'")
            return False

        years = data["meta"].get("years", [])
        by_year = data["by_year"]
        if not years or not by_year:
            self.results["errors"].append("Network metrics has empty year coverage")
            return False

        sample_year = str(years[-1])
        sample = by_year.get(sample_year, {})
        if "network_stats" not in sample:
            self.results["warnings"].append(f"Network metrics missing stats for {sample_year}")
            return False

        return True

    def run_all_tests(self) -> Dict:
        """Run all validation tests."""
        print("\n" + "=" * 60)
        print("PIPELINE VALIDATION SUITE")
        print("=" * 60 + "\n")

        # File existence tests
        print("Testing data files...")
        self.run_test("Visualization parquet exists and valid", self.test_viz_parquet)
        self.run_test("Topology fields JSON exists and valid", self.test_topology_fields)
        self.run_test("Research summary JSON exists and valid", self.test_research_summary)
        self.run_test("Country embedding JSON exists and valid", self.test_country_embedding)

        # Data quality tests
        print("\nTesting data quality...")
        self.run_test("Field statistics are reasonable", self.test_field_statistics)
        self.run_test("Network metrics JSON exists and valid", self.test_network_metrics)

        # Consistency tests
        print("\nTesting data consistency...")
        self.run_test("Cross-dataset consistency", self.test_data_consistency)

        # Determine overall status
        if self.results["tests_failed"] == 0:
            if len(self.results["warnings"]) == 0:
                self.results["status"] = "PASS"
            else:
                self.results["status"] = "PASS_WITH_WARNINGS"
        else:
            self.results["status"] = "FAIL"

        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Status: {self.results['status']}")
        print(f"Tests passed: {self.results['tests_passed']}/{self.results['tests_total']}")
        print(f"Tests failed: {self.results['tests_failed']}")
        print(f"Warnings: {len(self.results['warnings'])}")
        print(f"Errors: {len(self.results['errors'])}")

        if self.results["errors"]:
            print("\nErrors:")
            for err in self.results["errors"]:
                print(f"  ❌ {err}")

        if self.results["warnings"]:
            print("\nWarnings:")
            for warn in self.results["warnings"]:
                print(f"  ⚠️  {warn}")

        print("\n" + "=" * 60)

        return self.results


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_path = Path(args.out)

    validator = PipelineValidator(str(base_dir))
    results = validator.run_all_tests()

    # Save report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation report saved to: {out_path}")

    # Exit with appropriate code
    if results["status"] == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
