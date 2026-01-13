Data handling notes
- Primary data lives in /Users/ian/trade_data_warehouse (external to this repo).
- Keep total local data under 500MB.
- Use subfolders:
  raw/ - immutable extracts or downloaded files.
  interim/ - cleaned but not final, reproducible intermediates.
  processed/ - analysis-ready datasets.
  external/ - small reference files or metadata.
