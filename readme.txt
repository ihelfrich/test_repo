## Project Overview

### Project Name
### Objectives:

##### Run a gravity model analysis of international trade using data from {DATA DIRECTORY}

#### DATA DIRECTORY: /Users/ian/trade_data_warehouse

### Start Date: Jan 13, 2026

### Author: Dr. Ian Helfrich.

#### Limitations: Use less than 500MB of data

### Project Structure
- config/: project metadata and path configuration
- data/: raw/interim/processed data (kept small; primary data stays external)
- docs/: literature, methods, and meeting notes
- logs/: running log for progress and decisions
- notebooks/: exploratory analysis
- outputs/: figures, tables, and saved models
- scripts/: pipeline scripts and CLI entry points
- src/: reusable analysis code

### Running Log
- logs/running_log.md is the canonical log for progress, decisions, and key notes for collaboration.



##### Setting up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


##### To run existing environments: 
# find the environment.yml file
# .\.venv\Scripts\Activate.ps1
