#!/usr/bin/env python3
"""
Build Full Gravity Dataset - All Years, All Countries
=====================================================

Extracts complete BACI bilateral trade data merged with CEPII gravity variables.

Features:
- All available years (typically 2000-2023)
- All country pairs (not just top N)
- Optional filtering by year range, countries, minimum trade value
- Efficient parquet storage with compression
- Memory-efficient processing for large datasets

Usage:
    python scripts/05_build_full_dataset.py --help
    python scripts/05_build_full_dataset.py --min-year 2010 --max-year 2023
    python scripts/05_build_full_dataset.py --countries USA,CHN,DEU,JPN
    python scripts/05_build_full_dataset.py --min-trade 1.0  # Filter flows < $1M
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load project configuration."""
    config_path = Path(__file__).parent.parent / "config" / "project_config.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_baci_data(data_dir, min_year=None, max_year=None, countries=None):
    """
    Load BACI bilateral trade data with optional filtering.

    Parameters:
    -----------
    data_dir : Path
        Path to trade data warehouse
    min_year : int, optional
        Minimum year to include
    max_year : int, optional
        Maximum year to include
    countries : list of str, optional
        List of ISO3 country codes to filter (both origin and destination)

    Returns:
    --------
    pd.DataFrame
        BACI trade flows with columns: year, iso_o, iso_d, trade_value_usd_millions
    """
    baci_path = data_dir / "baci" / "baci_bilateral_totals.parquet"
    logger.info(f"Loading BACI data from {baci_path}")

    # Read parquet file
    df = pd.read_parquet(baci_path)
    logger.info(f"Loaded {len(df):,} rows, {df['year'].nunique()} years, "
                f"{df['iso_o'].nunique()} origin countries")

    # Apply year filter
    if min_year is not None or max_year is not None:
        min_yr = min_year or df['year'].min()
        max_yr = max_year or df['year'].max()
        df = df[(df['year'] >= min_yr) & (df['year'] <= max_yr)]
        logger.info(f"Filtered to years {min_yr}-{max_yr}: {len(df):,} rows")

    # Apply country filter
    if countries is not None:
        country_set = set(countries)
        df = df[
            df['iso_o'].isin(country_set) & df['iso_d'].isin(country_set)
        ]
        logger.info(f"Filtered to {len(countries)} countries: {len(df):,} rows")

    return df


def load_gravity_data(data_dir):
    """
    Load CEPII gravity dataset.

    Returns:
    --------
    pd.DataFrame
        Gravity variables: dist, contig, comlang_off, comcol, etc.
    """
    gravity_path = data_dir / "gravity" / "gravity_v202211.parquet"
    logger.info(f"Loading gravity data from {gravity_path}")

    df = pd.read_parquet(gravity_path)

    # Rename columns to match our convention
    df = df.rename(columns={
        'country_id_o': 'iso_o',
        'country_id_d': 'iso_d'
    })

    logger.info(f"Loaded gravity data: {len(df):,} country pairs, "
                f"{df['year'].nunique()} years")

    return df


def merge_baci_gravity(baci_df, gravity_df):
    """
    Merge BACI trade flows with gravity variables.

    Parameters:
    -----------
    baci_df : pd.DataFrame
        BACI bilateral trade flows
    gravity_df : pd.DataFrame
        CEPII gravity variables

    Returns:
    --------
    pd.DataFrame
        Merged dataset with trade flows and gravity variables
    """
    logger.info("Merging BACI with gravity data...")

    # Merge on year, iso_o, iso_d
    df = pd.merge(
        baci_df,
        gravity_df,
        on=['year', 'iso_o', 'iso_d'],
        how='inner'  # Only keep pairs with both trade and gravity data
    )

    logger.info(f"Merged dataset: {len(df):,} observations")
    logger.info(f"Coverage: {df['year'].min()}-{df['year'].max()}, "
                f"{df['iso_o'].nunique()} origins, {df['iso_d'].nunique()} destinations")

    # Check merge quality
    merge_rate = len(df) / len(baci_df) * 100
    logger.info(f"Merge rate: {merge_rate:.1f}% of BACI observations matched")

    return df


def engineer_features(df):
    """
    Create derived features for gravity estimation.

    Parameters:
    -----------
    df : pd.DataFrame
        Merged BACI + gravity data

    Returns:
    --------
    pd.DataFrame
        Dataset with additional engineered features
    """
    logger.info("Engineering features...")

    # Log transformations
    df['ln_dist'] = np.log(df['dist'])
    df['ln_trade'] = np.log1p(df['trade_value_usd_millions'])

    # GDP and population logs (if available)
    for var in ['gdp_o', 'gdp_d', 'pop_o', 'pop_d']:
        if var in df.columns:
            df[f'ln_{var}'] = np.log(df[var].replace(0, np.nan))

    # GDP product
    if 'gdp_o' in df.columns and 'gdp_d' in df.columns:
        df['gdp_prod'] = df['gdp_o'] * df['gdp_d']
        df['ln_gdp_prod'] = np.log(df['gdp_prod'].replace(0, np.nan))

    # Binary indicators
    for var in ['contig', 'comlang_off', 'comcol']:
        if var in df.columns:
            df[var] = df[var].fillna(0).astype(int)

    # RTA coverage (if available)
    if 'rta_coverage' in df.columns:
        df['rta_coverage'] = df['rta_coverage'].fillna(0)

    logger.info(f"Engineered features complete. Columns: {len(df.columns)}")

    return df


def filter_minimum_trade(df, min_trade_millions):
    """
    Remove very small trade flows.

    Parameters:
    -----------
    df : pd.DataFrame
        Trade data
    min_trade_millions : float
        Minimum trade value in USD millions

    Returns:
    --------
    pd.DataFrame
        Filtered dataset
    """
    if min_trade_millions > 0:
        initial_len = len(df)
        df = df[df['trade_value_usd_millions'] >= min_trade_millions]
        removed = initial_len - len(df)
        logger.info(f"Removed {removed:,} flows < ${min_trade_millions}M "
                    f"({removed/initial_len*100:.1f}%)")
    return df


def save_dataset(df, output_path, compression='snappy'):
    """
    Save dataset to parquet with compression.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to save
    output_path : Path
        Output file path
    compression : str
        Compression algorithm ('snappy', 'gzip', 'zstd')
    """
    logger.info(f"Saving to {output_path} with {compression} compression...")

    df.to_parquet(
        output_path,
        index=False,
        compression=compression,
        engine='pyarrow'
    )

    # Report file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(df):,} rows to {output_path.name} ({size_mb:.1f} MB)")


def generate_summary_stats(df):
    """Generate and print summary statistics."""
    logger.info("\n" + "="*70)
    logger.info("DATASET SUMMARY")
    logger.info("="*70)

    print(f"\nDimensions:")
    print(f"  Observations: {len(df):,}")
    print(f"  Variables: {len(df.columns)}")

    print(f"\nTemporal Coverage:")
    print(f"  Years: {df['year'].min()}-{df['year'].max()} ({df['year'].nunique()} years)")
    print(f"  Observations per year: {len(df) / df['year'].nunique():.0f} average")

    print(f"\nSpatial Coverage:")
    print(f"  Origin countries: {df['iso_o'].nunique()}")
    print(f"  Destination countries: {df['iso_d'].nunique()}")
    print(f"  Unique dyads: {df.groupby(['iso_o', 'iso_d']).ngroups:,}")

    print(f"\nTrade Values (USD Millions):")
    print(f"  Total: ${df['trade_value_usd_millions'].sum():,.0f}M")
    print(f"  Mean: ${df['trade_value_usd_millions'].mean():,.0f}M")
    print(f"  Median: ${df['trade_value_usd_millions'].median():,.0f}M")
    print(f"  Std Dev: ${df['trade_value_usd_millions'].std():,.0f}M")

    print(f"\nZero Trade Flows:")
    zeros = (df['trade_value_usd_millions'] == 0).sum()
    print(f"  Count: {zeros:,} ({zeros/len(df)*100:.1f}%)")

    print(f"\nGravity Variables:")
    print(f"  Mean log distance: {df['ln_dist'].mean():.2f}")
    print(f"  Contiguous pairs: {df['contig'].sum():,} ({df['contig'].mean()*100:.1f}%)")
    if 'comlang_off' in df.columns:
        print(f"  Common language: {df['comlang_off'].sum():,} ({df['comlang_off'].mean()*100:.1f}%)")
    if 'comcol' in df.columns:
        print(f"  Colonial ties: {df['comcol'].sum():,} ({df['comcol'].mean()*100:.1f}%)")
    if 'rta_coverage' in df.columns:
        print(f"  Mean RTA coverage: {df['rta_coverage'].mean():.3f}")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build full gravity dataset from BACI and CEPII data"
    )
    parser.add_argument(
        '--min-year', type=int, default=None,
        help='Minimum year to include (default: all available)'
    )
    parser.add_argument(
        '--max-year', type=int, default=None,
        help='Maximum year to include (default: all available)'
    )
    parser.add_argument(
        '--countries', type=str, default=None,
        help='Comma-separated list of ISO3 country codes to filter (default: all)'
    )
    parser.add_argument(
        '--min-trade', type=float, default=0.0,
        help='Minimum trade value in USD millions (default: 0)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: data/processed/baci_gravity_full.parquet)'
    )
    parser.add_argument(
        '--compression', type=str, default='snappy',
        choices=['snappy', 'gzip', 'zstd'],
        help='Compression algorithm (default: snappy)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    data_dir = Path(config['data_directory'])

    # Parse countries if provided
    countries = None
    if args.countries:
        countries = [c.strip().upper() for c in args.countries.split(',')]
        logger.info(f"Filtering to countries: {countries}")

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "processed" / "baci_gravity_full.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        baci_df = load_baci_data(data_dir, args.min_year, args.max_year, countries)
        gravity_df = load_gravity_data(data_dir)

        # Merge datasets
        df = merge_baci_gravity(baci_df, gravity_df)

        # Engineer features
        df = engineer_features(df)

        # Filter minimum trade
        df = filter_minimum_trade(df, args.min_trade)

        # Save dataset
        save_dataset(df, output_path, args.compression)

        # Generate summary statistics
        generate_summary_stats(df)

        logger.info("✓ Dataset build complete!")

        # Suggest next steps
        print("\nNext steps:")
        print(f"  1. Run PPML estimation: python scripts/03_ppml.py --input {output_path}")
        print(f"  2. Generate visualizations: python scripts/04_prepare_viz_data.py --input {output_path}")
        print(f"  3. Create interactive dashboard: python scripts/04_interactive_dashboard.py --input {output_path}")

    except Exception as e:
        logger.error(f"Error building dataset: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
