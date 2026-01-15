#!/usr/bin/env python3
"""
Master Analysis Script - Comprehensive Trade Network Report

This script generates a complete analytical report combining:
1. Gravity model diagnostics
2. Topology field statistics
3. Research summary metrics
4. Temporal evolution analysis
5. Statistical tests and correlations

Output: A comprehensive JSON report with all insights in one place.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive trade network analysis report."
    )
    parser.add_argument(
        "--topology",
        default="docs/data/topology_fields.json",
        help="Topology fields JSON.",
    )
    parser.add_argument(
        "--summary",
        default="docs/data/research_summary.json",
        help="Research summary JSON.",
    )
    parser.add_argument(
        "--network",
        default="docs/data/network_metrics.json",
        help="Network metrics JSON (optional).",
    )
    parser.add_argument(
        "--out",
        default="docs/data/master_report.json",
        help="Output master report path.",
    )
    return parser.parse_args()


class MasterAnalyzer:
    """Generate comprehensive analysis report."""

    def __init__(self):
        self.topology_data = None
        self.summary_data = None
        self.network_data = None
        self.report = {
            "meta": {
                "title": "Comprehensive Trade Network Analysis Report",
                "generated": None
            },
            "data_overview": {},
            "temporal_evolution": {},
            "topology_insights": {},
            "network_insights": {},
            "statistical_tests": {},
            "key_findings": [],
            "recommendations": []
        }

    def load_data(self, topology_path: Path, summary_path: Path, network_path: Path) -> bool:
        """Load all data files."""
        try:
            with open(topology_path, 'r') as f:
                self.topology_data = json.load(f)
            print(f"✅ Loaded topology data: {len(self.topology_data['fields'])} years")

            with open(summary_path, 'r') as f:
                self.summary_data = json.load(f)
            print(f"✅ Loaded research summary: {len(self.summary_data['metrics'])} years")

            if network_path.exists():
                with open(network_path, 'r') as f:
                    self.network_data = json.load(f)
                print(f"✅ Loaded network metrics: {len(self.network_data['by_year'])} years")
            else:
                self.network_data = None
                print(f"⚠️  Network metrics not found: {network_path}")

            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def analyze_data_overview(self) -> Dict:
        """Analyze basic data characteristics."""
        print("\n📊 Analyzing data overview...")

        topo_meta = self.topology_data.get("meta", {})
        summary_meta = self.summary_data.get("meta", {})

        overview = {
            "coverage": {
                "years": topo_meta.get("years", []),
                "num_years": len(topo_meta.get("years", [])),
                "countries": topo_meta.get("countries", 0),
                "grid_size": topo_meta.get("grid_size", 0),
                "total_observations": summary_meta.get("rows", 0)
            },
            "data_sources": {
                "topology_source": topo_meta.get("source", ""),
                "summary_source": summary_meta.get("source", ""),
                "field_type": topo_meta.get("field", ""),
                "method": topo_meta.get("method", "")
            }
        }

        return overview

    def analyze_temporal_evolution(self) -> Dict:
        """Analyze how metrics evolve over time."""
        print("\n⏰ Analyzing temporal evolution...")

        years = self.topology_data["meta"]["years"]
        fields = self.topology_data["fields"]
        metrics = self.summary_data["metrics"]

        evolution = {
            "field_variance_trend": [],
            "trade_volume_trend": [],
            "residual_dispersion_trend": [],
            "wasserstein_shift_trend": [],
            "concentration_trend": []
        }

        for year in years:
            year_str = str(year)

            # Field statistics
            if year_str in fields and "variance" in fields[year_str]:
                evolution["field_variance_trend"].append({
                    "year": year,
                    "variance": fields[year_str]["variance"]
                })

            # Research metrics
            if year_str in metrics:
                m = metrics[year_str]

                if "total_trade" in m:
                    evolution["trade_volume_trend"].append({
                        "year": year,
                        "total_trade_usd_millions": m["total_trade"]
                    })

                if "residual_std" in m:
                    evolution["residual_dispersion_trend"].append({
                        "year": year,
                        "residual_std": m["residual_std"]
                    })

                if "wasserstein_shift" in m:
                    evolution["wasserstein_shift_trend"].append({
                        "year": year,
                        "shift": m["wasserstein_shift"]
                    })

                if "export_hhi" in m:
                    evolution["concentration_trend"].append({
                        "year": year,
                        "export_hhi": m["export_hhi"],
                        "import_hhi": m.get("import_hhi", 0)
                    })

        return evolution

    def analyze_network_insights(self) -> Dict:
        """Summarize network metrics if available."""
        if not self.network_data:
            return {}

        years = self.network_data.get("meta", {}).get("years", [])
        if not years:
            return {}

        latest_year = str(years[-1])
        latest = self.network_data["by_year"].get(latest_year, {})
        stats = latest.get("network_stats", {})

        trend = []
        for year in years:
            entry = self.network_data["by_year"].get(str(year), {})
            net_stats = entry.get("network_stats", {})
            if net_stats:
                trend.append({
                    "year": int(year),
                    "density": net_stats.get("density", 0),
                    "reciprocity": net_stats.get("reciprocity", 0),
                    "assortativity": net_stats.get("assortativity", 0),
                    "global_clustering": net_stats.get("global_clustering", 0),
                })

        return {
            "latest_year": int(latest_year),
            "network_stats": stats,
            "top_by_pagerank": latest.get("top_by_pagerank", [])[:5],
            "top_by_betweenness": latest.get("top_by_betweenness", [])[:5],
            "density_trend": trend
        }

    def analyze_topology_insights(self) -> Dict:
        """Extract topological insights from field data."""
        print("\n🔬 Analyzing topology insights...")

        fields = self.topology_data["fields"]
        years = sorted([int(y) for y in fields.keys()])

        insights = {
            "field_extremes": {
                "max_variance_year": None,
                "min_variance_year": None,
                "max_mean_year": None,
                "min_mean_year": None
            },
            "field_statistics": {}
        }

        # Collect statistics
        variances = []
        means = []

        for year in years:
            year_str = str(year)
            field_data = fields[year_str]

            variance = field_data.get("variance", 0)
            mean = field_data.get("mean", 0)

            variances.append((year, variance))
            means.append((year, mean))

            insights["field_statistics"][year] = {
                "variance": variance,
                "mean": mean,
                "min": field_data.get("min", 0),
                "max": field_data.get("max", 0),
                "range": field_data.get("max", 0) - field_data.get("min", 0)
            }

        # Find extremes
        if variances:
            max_var_year, max_var = max(variances, key=lambda x: x[1])
            min_var_year, min_var = min(variances, key=lambda x: x[1])
            insights["field_extremes"]["max_variance_year"] = {
                "year": max_var_year,
                "variance": max_var
            }
            insights["field_extremes"]["min_variance_year"] = {
                "year": min_var_year,
                "variance": min_var
            }

        if means:
            max_mean_year, max_mean = max(means, key=lambda x: x[1])
            min_mean_year, min_mean = min(means, key=lambda x: x[1])
            insights["field_extremes"]["max_mean_year"] = {
                "year": max_mean_year,
                "mean": max_mean
            }
            insights["field_extremes"]["min_mean_year"] = {
                "year": min_mean_year,
                "mean": min_mean
            }

        return insights

    def perform_statistical_tests(self) -> Dict:
        """Perform statistical tests on the data."""
        print("\n📈 Performing statistical tests...")

        metrics = self.summary_data["metrics"]
        years = sorted([int(y) for y in metrics.keys()])

        tests = {
            "trend_analysis": {},
            "correlations": {},
            "volatility": {}
        }

        # Extract time series
        residual_stds = []
        wasserstein_shifts = []
        trade_volumes = []

        for year in years:
            year_str = str(year)
            m = metrics[year_str]

            if "residual_std" in m:
                residual_stds.append(m["residual_std"])
            if "wasserstein_shift" in m:
                wasserstein_shifts.append(m["wasserstein_shift"])
            if "total_trade" in m:
                trade_volumes.append(m["total_trade"])

        # Trend analysis (linear regression)
        if len(residual_stds) > 2:
            x = np.arange(len(residual_stds))
            coeffs = np.polyfit(x, residual_stds, 1)
            tests["trend_analysis"]["residual_dispersion"] = {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "interpretation": "increasing" if coeffs[0] > 0 else "decreasing"
            }

        if len(trade_volumes) > 2:
            x = np.arange(len(trade_volumes))
            coeffs = np.polyfit(x, trade_volumes, 1)
            tests["trend_analysis"]["trade_volume"] = {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "interpretation": "growing" if coeffs[0] > 0 else "declining"
            }

        # Volatility analysis
        if len(residual_stds) > 1:
            tests["volatility"]["residual_std"] = {
                "mean": float(np.mean(residual_stds)),
                "std": float(np.std(residual_stds)),
                "coefficient_of_variation": float(np.std(residual_stds) / (np.mean(residual_stds) + 1e-10))
            }

        if len(wasserstein_shifts) > 1:
            tests["volatility"]["wasserstein_shift"] = {
                "mean": float(np.mean(wasserstein_shifts)),
                "std": float(np.std(wasserstein_shifts)),
                "max_shift": float(max(wasserstein_shifts))
            }

        # Correlation analysis
        if len(residual_stds) == len(trade_volumes) and len(residual_stds) > 2:
            corr = np.corrcoef(residual_stds, trade_volumes)[0, 1]
            tests["correlations"]["residual_vs_volume"] = {
                "correlation": float(corr),
                "interpretation": "positive" if corr > 0.3 else ("negative" if corr < -0.3 else "weak")
            }

        return tests

    def generate_key_findings(self) -> List[Dict]:
        """Generate key findings from the analysis."""
        print("\n🔍 Generating key findings...")

        findings = []

        # Finding 1: Coverage
        years = self.topology_data["meta"]["years"]
        findings.append({
            "category": "Data Coverage",
            "finding": f"Analysis covers {len(years)} years ({min(years)}-{max(years)}) with {self.topology_data['meta']['countries']} countries",
            "importance": "high",
            "implication": "Comprehensive temporal and spatial coverage enables robust trend analysis"
        })

        # Finding 2: Largest shifts
        summary = self.summary_data.get("summary", {})
        if "largest_shifts" in summary and summary["largest_shifts"]:
            shift_years = ", ".join(str(y) for y in summary["largest_shifts"][:3])
            findings.append({
                "category": "Structural Changes",
                "finding": f"Largest distribution shifts occurred in years: {shift_years}",
                "importance": "high",
                "implication": "These years likely correspond to major economic events or policy changes"
            })

        # Finding 3: Variance trend
        if "trend_analysis" in self.report["statistical_tests"]:
            if "residual_dispersion" in self.report["statistical_tests"]["trend_analysis"]:
                trend = self.report["statistical_tests"]["trend_analysis"]["residual_dispersion"]
                findings.append({
                    "category": "Residual Dispersion Trend",
                    "finding": f"Residual dispersion is {trend['interpretation']} over time (slope: {trend['slope']:.6f})",
                    "importance": "medium",
                    "implication": "Indicates whether gravity model fit is improving or deteriorating"
                })

        # Finding 4: Field variance extremes
        if "field_extremes" in self.report["topology_insights"]:
            extremes = self.report["topology_insights"]["field_extremes"]
            if extremes["max_variance_year"]:
                year = extremes["max_variance_year"]["year"]
                findings.append({
                    "category": "Topological Extremes",
                    "finding": f"Maximum field variance observed in {year}",
                    "importance": "medium",
                    "implication": "This year shows highest spatial heterogeneity in trade deviations"
                })

        return findings

    def generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations."""
        print("\n💡 Generating recommendations...")

        recommendations = []

        # Recommendation 1: Data quality
        recommendations.append({
            "category": "Data Quality",
            "recommendation": "Validate topology fields against known economic events (COVID-19, trade wars)",
            "priority": "high",
            "rationale": "Ensures field dynamics capture real-world shocks"
        })

        # Recommendation 2: Analysis extension
        if not self.network_data:
            recommendations.append({
                "category": "Analysis Extension",
                "recommendation": "Compute network centrality metrics (PageRank, betweenness) from trade flows",
                "priority": "medium",
                "rationale": "Would identify systemically important countries beyond simple trade volume"
            })
        else:
            recommendations.append({
                "category": "Analysis Extension",
                "recommendation": "Extend network metrics to sector-level flows (HS2) for product-specific centrality",
                "priority": "medium",
                "rationale": "Highlights sector-specific chokepoints and supply risk concentrations"
            })

        # Recommendation 3: Visualization
        recommendations.append({
            "category": "Visualization",
            "recommendation": "Create animated time series showing field evolution",
            "priority": "medium",
            "rationale": "Dynamic visualization reveals temporal patterns not visible in static plots"
        })

        # Recommendation 4: Modeling
        if "statistical_tests" in self.report and "correlations" in self.report["statistical_tests"]:
            recommendations.append({
                "category": "Statistical Modeling",
                "recommendation": "Develop predictive model for Wasserstein shifts using lagged residuals",
                "priority": "low",
                "rationale": "Could provide early warning of structural breaks"
            })

        return recommendations

    def generate_report(self) -> Dict:
        """Generate complete master report."""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 70)

        # Run all analyses
        self.report["data_overview"] = self.analyze_data_overview()
        self.report["temporal_evolution"] = self.analyze_temporal_evolution()
        self.report["topology_insights"] = self.analyze_topology_insights()
        self.report["network_insights"] = self.analyze_network_insights()
        self.report["statistical_tests"] = self.perform_statistical_tests()
        self.report["key_findings"] = self.generate_key_findings()
        self.report["recommendations"] = self.generate_recommendations()

        # Add metadata
        from datetime import datetime
        self.report["meta"]["generated"] = datetime.now().isoformat()

        print("\n✅ Report generation complete")
        return self.report


def main() -> None:
    args = parse_args()

    topology_path = Path(args.topology)
    summary_path = Path(args.summary)
    network_path = Path(args.network)
    out_path = Path(args.out)

    # Check files exist
    if not topology_path.exists():
        print(f"❌ Topology file not found: {topology_path}")
        sys.exit(1)

    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        sys.exit(1)

    # Create analyzer
    analyzer = MasterAnalyzer()

    # Load data
    if not analyzer.load_data(topology_path, summary_path, network_path):
        sys.exit(1)

    # Generate report
    report = analyzer.generate_report()

    # Save report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("REPORT SUMMARY")
    print("=" * 70)
    print(f"Coverage: {report['data_overview']['coverage']['num_years']} years, "
          f"{report['data_overview']['coverage']['countries']} countries")
    print(f"\nKey Findings: {len(report['key_findings'])}")
    for i, finding in enumerate(report['key_findings'], 1):
        print(f"  {i}. [{finding['category']}] {finding['finding']}")

    print(f"\nRecommendations: {len(report['recommendations'])}")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. [{rec['priority'].upper()}] {rec['recommendation']}")

    print(f"\n📄 Full report saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
