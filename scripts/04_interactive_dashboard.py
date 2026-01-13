"""
Interactive Dashboard Generator for Trade & Gravity Model Analysis
Generates standalone HTML visualizations deployable on GitHub Pages
"""
import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interactive visualizations for gravity model analysis."
    )
    parser.add_argument(
        "--baci-sample",
        default="data/processed/baci_sample.parquet",
        help="Input BACI sample parquet.",
    )
    parser.add_argument(
        "--ppml-coefs",
        default="outputs/tables/ppml_coefficients.csv",
        help="PPML coefficients CSV.",
    )
    parser.add_argument(
        "--gravity-coords",
        default="/Users/ian/trade_data_warehouse/gravity/gravity_countries.parquet",
        help="Country coordinates for geographic visualizations.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/dashboard",
        help="Output directory for HTML dashboards.",
    )
    return parser.parse_args()


def create_trade_globe_3d(df: pd.DataFrame, country_coords: pd.DataFrame, output_path: Path):
    """
    Creates an interactive 3D globe visualization with trade flows as arcs.
    """
    # Aggregate top trade flows
    top_flows = (
        df.groupby(["iso_o", "iso_d"], as_index=False)["trade_value_usd_millions"]
        .sum()
        .sort_values("trade_value_usd_millions", ascending=False)
        .head(100)
    )

    # Merge with coordinates
    coords_o = country_coords.rename(columns={"iso3": "iso_o", "lat": "lat_o", "lon": "lon_o"})
    coords_d = country_coords.rename(columns={"iso3": "iso_d", "lat": "lat_d", "lon": "lon_d"})

    flows_geo = top_flows.merge(coords_o[["iso_o", "lat_o", "lon_o"]], on="iso_o", how="left")
    flows_geo = flows_geo.merge(coords_d[["iso_d", "lat_d", "lon_d"]], on="iso_d", how="left")
    flows_geo = flows_geo.dropna()

    # Create 3D globe with flows
    fig = go.Figure()

    # Add country scatter points
    all_countries = pd.concat([
        country_coords[["iso3", "lat", "lon"]].rename(columns={"iso3": "country"}),
    ]).drop_duplicates()

    fig.add_trace(go.Scattergeo(
        lon=all_countries["lon"],
        lat=all_countries["lat"],
        mode="markers",
        marker=dict(size=3, color="lightblue", opacity=0.6),
        name="Countries",
        hovertext=all_countries["country"],
    ))

    # Add trade flow arcs
    for _, row in flows_geo.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row["lon_o"], row["lon_d"]],
            lat=[row["lat_o"], row["lat_d"]],
            mode="lines",
            line=dict(width=row["trade_value_usd_millions"] / 50000, color="red"),
            opacity=0.4,
            showlegend=False,
            hovertext=f"{row['iso_o']} → {row['iso_d']}: ${row['trade_value_usd_millions']:.0f}M",
        ))

    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        countrycolor="lightgray",
    )

    fig.update_layout(
        title="Global Trade Flow Network (Top 100 Bilateral Flows)",
        height=800,
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f"Created 3D trade globe: {output_path}")


def create_trade_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Creates an interactive bilateral trade heatmap.
    """
    # Aggregate by country pair
    trade_matrix = df.groupby(["iso_o", "iso_d"], as_index=False)[
        "trade_value_usd_millions"
    ].sum()

    # Get top exporters and importers
    top_exporters = (
        trade_matrix.groupby("iso_o")["trade_value_usd_millions"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    top_importers = (
        trade_matrix.groupby("iso_d")["trade_value_usd_millions"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )

    # Filter to top countries
    trade_subset = trade_matrix[
        trade_matrix["iso_o"].isin(top_exporters) & trade_matrix["iso_d"].isin(top_importers)
    ]

    # Pivot to matrix
    matrix = trade_subset.pivot(index="iso_o", columns="iso_d", values="trade_value_usd_millions")
    matrix = matrix.fillna(0)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="YlOrRd",
        hovertemplate="Exporter: %{y}<br>Importer: %{x}<br>Trade: $%{z:.0f}M<extra></extra>",
    ))

    fig.update_layout(
        title="Bilateral Trade Heatmap (Top 30 Exporters × Top 30 Importers)",
        xaxis_title="Importer (ISO3)",
        yaxis_title="Exporter (ISO3)",
        height=900,
        width=950,
    )

    fig.write_html(output_path)
    print(f"Created trade heatmap: {output_path}")


def create_coefficient_plot(coefs_df: pd.DataFrame, output_path: Path):
    """
    Creates an interactive coefficient plot with confidence intervals.
    """
    # Filter to main gravity variables (exclude fixed effects)
    gravity_vars = ["ln_dist", "contig", "comlang_off", "comcol", "ln_gdp_o", "ln_gdp_d",
                    "ln_pop_o", "ln_pop_d", "rta_coverage"]
    coefs_subset = coefs_df[coefs_df["term"].isin(gravity_vars)].copy()

    # Calculate confidence intervals
    coefs_subset["ci_lower"] = coefs_subset["coef"] - 1.96 * coefs_subset["std_err"]
    coefs_subset["ci_upper"] = coefs_subset["coef"] + 1.96 * coefs_subset["std_err"]

    # Sort by coefficient size
    coefs_subset = coefs_subset.sort_values("coef")

    # Create coefficient plot
    fig = go.Figure()

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=coefs_subset["ci_lower"],
        y=coefs_subset["term"],
        mode="markers",
        marker=dict(size=8, color="lightgray"),
        name="95% CI Lower",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=coefs_subset["ci_upper"],
        y=coefs_subset["term"],
        mode="markers",
        marker=dict(size=8, color="lightgray"),
        name="95% CI Upper",
        showlegend=False,
    ))

    # Add coefficient points
    fig.add_trace(go.Scatter(
        x=coefs_subset["coef"],
        y=coefs_subset["term"],
        mode="markers",
        marker=dict(size=12, color="steelblue"),
        name="Coefficient",
        error_x=dict(
            type="data",
            array=1.96 * coefs_subset["std_err"],
            color="gray",
            thickness=2,
        ),
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.3f}<extra></extra>",
    ))

    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)

    fig.update_layout(
        title="PPML Gravity Model Coefficients (with 95% Confidence Intervals)",
        xaxis_title="Coefficient Estimate",
        yaxis_title="Variable",
        height=600,
        showlegend=False,
    )

    fig.write_html(output_path)
    print(f"Created coefficient plot: {output_path}")


def create_trade_time_series(df: pd.DataFrame, output_path: Path):
    """
    Creates an interactive time series of trade flows with animation.
    """
    # Aggregate by year
    trade_by_year = df.groupby("year", as_index=False)["trade_value_usd_millions"].sum()

    # Create animated bar chart by country over time
    top_countries = (
        df.groupby("iso_o")["trade_value_usd_millions"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    df_top = df[df["iso_o"].isin(top_countries)]
    trade_by_country_year = (
        df_top.groupby(["year", "iso_o"], as_index=False)["trade_value_usd_millions"]
        .sum()
        .sort_values(["year", "trade_value_usd_millions"], ascending=[True, False])
    )

    fig = px.bar(
        trade_by_country_year,
        x="iso_o",
        y="trade_value_usd_millions",
        animation_frame="year",
        color="iso_o",
        title="Top 15 Exporters: Trade Value Over Time",
        labels={
            "trade_value_usd_millions": "Trade Value (USD Millions)",
            "iso_o": "Exporter (ISO3)",
        },
        height=600,
    )

    fig.update_layout(showlegend=False)
    fig.write_html(output_path)
    print(f"Created time series animation: {output_path}")


def create_distribution_plots(df: pd.DataFrame, output_path: Path):
    """
    Creates interactive distribution plots for trade values.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Trade Value Distribution (Linear Scale)",
            "Trade Value Distribution (Log Scale)",
            "Zero Trade Flows by Year",
            "Top 20 Country Pairs by Total Trade",
        ),
    )

    # 1. Linear scale histogram
    fig.add_trace(
        go.Histogram(
            x=df["trade_value_usd_millions"],
            name="Trade Value",
            nbinsx=50,
            marker_color="steelblue",
        ),
        row=1, col=1,
    )

    # 2. Log scale histogram
    df_nonzero = df[df["trade_value_usd_millions"] > 0]
    fig.add_trace(
        go.Histogram(
            x=df_nonzero["trade_value_usd_millions"],
            name="Trade Value (log)",
            nbinsx=50,
            marker_color="coral",
        ),
        row=1, col=2,
    )

    # 3. Zero flows by year
    zeros_by_year = df.groupby("year").apply(
        lambda x: (x["trade_value_usd_millions"] == 0).mean() * 100
    ).reset_index()
    zeros_by_year.columns = ["year", "zero_pct"]

    fig.add_trace(
        go.Bar(x=zeros_by_year["year"], y=zeros_by_year["zero_pct"], marker_color="darkred"),
        row=2, col=1,
    )

    # 4. Top country pairs
    top_pairs = (
        df.groupby(["iso_o", "iso_d"], as_index=False)["trade_value_usd_millions"]
        .sum()
        .sort_values("trade_value_usd_millions", ascending=False)
        .head(20)
    )
    top_pairs["pair"] = top_pairs["iso_o"] + " → " + top_pairs["iso_d"]

    fig.add_trace(
        go.Bar(
            y=top_pairs["pair"][::-1],
            x=top_pairs["trade_value_usd_millions"][::-1],
            orientation="h",
            marker_color="teal",
        ),
        row=2, col=2,
    )

    # Update axes
    fig.update_xaxes(title_text="Trade Value (USD Millions)", row=1, col=1)
    fig.update_xaxes(title_text="Trade Value (USD Millions)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Trade Value (USD Millions)", row=2, col=2)

    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Zero Trade (% of Flows)", row=2, col=1)
    fig.update_yaxes(title_text="Country Pair", row=2, col=2)

    fig.update_layout(
        title_text="Trade Data Distributions & Patterns",
        height=900,
        showlegend=False,
    )

    fig.write_html(output_path)
    print(f"Created distribution plots: {output_path}")


def create_index_html(output_dir: Path):
    """
    Creates an index.html file linking all visualizations for GitHub Pages.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gravity Model Analysis - Interactive Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
        }
        .viz-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .viz-card h2 {
            color: #34495e;
            margin-top: 0;
        }
        .viz-card p {
            color: #7f8c8d;
            line-height: 1.6;
        }
        .viz-link {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .viz-link:hover {
            background-color: #2980b9;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>🌍 Gravity Model Analysis Dashboard</h1>
    <p class="subtitle">Interactive Visualizations of International Trade Flows (BACI Data, 2019-2021)</p>

    <div class="viz-card">
        <h2>📊 3D Trade Flow Globe</h2>
        <p>
            Explore the top 100 bilateral trade flows visualized as arcs on an interactive 3D globe.
            Rotate, zoom, and hover to see trade relationships between countries.
        </p>
        <a href="trade_globe_3d.html" class="viz-link">Open 3D Globe →</a>
    </div>

    <div class="viz-card">
        <h2>🔥 Bilateral Trade Heatmap</h2>
        <p>
            Interactive heatmap showing trade intensity between the top 30 exporters and importers.
            Hover over cells to see exact trade values.
        </p>
        <a href="trade_heatmap.html" class="viz-link">Open Heatmap →</a>
    </div>

    <div class="viz-card">
        <h2>📈 PPML Coefficient Plot</h2>
        <p>
            Gravity model coefficient estimates with 95% confidence intervals.
            See the impact of distance, GDP, language, borders, and trade agreements on bilateral trade.
        </p>
        <a href="coefficient_plot.html" class="viz-link">Open Coefficients →</a>
    </div>

    <div class="viz-card">
        <h2>🎬 Trade Time Series Animation</h2>
        <p>
            Watch how trade patterns evolved from 2019-2021 for the top 15 exporting countries.
            Use the animation controls to play, pause, or step through years.
        </p>
        <a href="trade_time_series.html" class="viz-link">Open Animation →</a>
    </div>

    <div class="viz-card">
        <h2>📊 Distribution Analysis</h2>
        <p>
            Comprehensive view of trade data distributions including value histograms,
            zero-trade analysis, and top country pairs.
        </p>
        <a href="distribution_plots.html" class="viz-link">Open Distributions →</a>
    </div>

    <div class="footer">
        <p>Generated using Python (Plotly) | Data: BACI (CEPII) | Model: PPML Gravity</p>
        <p>All visualizations are fully interactive - click, zoom, hover, and explore!</p>
    </div>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    index_path.write_text(html_content)
    print(f"Created index page: {index_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_parquet(args.baci_sample)

    # Load coefficients if available
    coefs_path = Path(args.ppml_coefs)
    if coefs_path.exists():
        coefs_df = pd.read_csv(coefs_path)
    else:
        print(f"Warning: Coefficients file not found at {coefs_path}. Skipping coefficient plot.")
        coefs_df = None

    # Load country coordinates if available
    coords_path = Path(args.gravity_coords)
    if coords_path.exists():
        country_coords = pd.read_parquet(coords_path)
    else:
        print(f"Warning: Country coordinates not found at {coords_path}. Skipping 3D globe.")
        country_coords = None

    print("Generating interactive visualizations...")

    # Generate all visualizations
    if country_coords is not None:
        create_trade_globe_3d(df, country_coords, out_dir / "trade_globe_3d.html")

    create_trade_heatmap(df, out_dir / "trade_heatmap.html")

    if coefs_df is not None:
        create_coefficient_plot(coefs_df, out_dir / "coefficient_plot.html")

    create_trade_time_series(df, out_dir / "trade_time_series.html")
    create_distribution_plots(df, out_dir / "distribution_plots.html")

    # Create index page
    create_index_html(out_dir)

    print(f"\n✅ Dashboard generation complete!")
    print(f"📁 All files saved to: {out_dir}")
    print(f"🌐 Open {out_dir / 'index.html'} in a browser to view the dashboard")
    print(f"📤 Ready for GitHub Pages deployment!")


if __name__ == "__main__":
    main()
