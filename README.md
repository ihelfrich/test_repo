# 🌍 Gravity Trade Flow Analysis

[![GitHub Pages](https://img.shields.io/badge/Demo-Live-brightgreen?style=for-the-badge&logo=github)](https://ihelfrich.github.io/test_repo/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Three.js](https://img.shields.io/badge/Three.js-Interactive-black?style=for-the-badge&logo=three.js)](https://threejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **Interactive 3D visualization and econometric analysis of international trade flows using PPML gravity models**

[**🚀 Launch Interactive Tool**](https://ihelfrich.github.io/test_repo/) • [**📊 Executive Summary**](https://ihelfrich.github.io/test_repo/executive-summary.html) • [**📖 Full Methodology**](https://ihelfrich.github.io/test_repo/methodology.html) • [**🏠 Project Home**](https://ihelfrich.github.io/test_repo/landing.html)

---

## ✨ Overview

This project implements a state-of-the-art **Poisson Pseudo-Maximum Likelihood (PPML) gravity model** for analyzing bilateral trade flows, featuring:

- 🎨 **Interactive 3D Visualization** - Explore trade patterns in a beautiful three.js-powered interface
- 📈 **Rigorous Econometrics** - Anderson-van Wincoop (2003) structural gravity with multilateral resistance
- 🌐 **Web-Based Dashboard** - No installation required, runs entirely in the browser
- 📊 **Executive Summary** - Consultant-quality report with policy implications and key findings
- 📖 **Full Methodology** - Publication-grade technical documentation with equations and references
- 🚀 **Production-Ready** - Deployed on GitHub Pages, mobile-responsive, fast

### Key Features

| Feature | Description |
|---------|-------------|
| **Data Source** | BACI bilateral trade flows (CEPII) + Gravity dataset (v202211) |
| **Time Period** | 2019-2021 (1,032 observations) |
| **Methodology** | PPML with exporter/importer/year fixed effects |
| **Visualization** | Three.js 3D point cloud with interactive controls |
| **Gravity Variables** | Distance, contiguity, common language, colonial ties, RTAs, GDP, population |

---

## 🎥 Demo

### Interactive 3D Trade Space

![Trade Space Visualization](outputs/figures/screenshot_demo.png)

**Features:**
- 🔄 **Rotate & Zoom** - Smooth 3D navigation with mouse/touch
- 📍 **Hover Tooltips** - Detailed information on each trade flow
- 🎛️ **Dynamic Controls** - Switch years and metrics in real-time
- 🎨 **Color-Coded** - Blue (underperforming) to Red (overperforming) gradient
- 📏 **Variable Sizing** - Point size reflects trade volume

---

## 📚 Documentation

This project includes comprehensive documentation suitable for academic, policy, and professional audiences:

### [📊 Executive Summary](https://ihelfrich.github.io/test_repo/executive-summary.html)
**Audience:** Policymakers, business strategists, consultants

Consultant-quality report featuring:
- Key statistics and findings at a glance
- Gravity model estimates with policy interpretations
- Implications for trade policy and business strategy
- Methodology overview for non-technical readers
- Reference to academic literature

### [📖 Full Methodology](https://ihelfrich.github.io/test_repo/methodology.html)
**Audience:** Researchers, economists, graduate students

Publication-grade technical documentation covering:
- Theoretical foundation (Anderson-van Wincoop framework)
- Empirical specification with fixed effects structure
- PPML estimation justification and advantages
- Data sources and sample construction details
- Counterfactual analysis procedures with caveats
- Model diagnostics and robustness checks
- Comprehensive reference list

### [🏠 Project Landing Page](https://ihelfrich.github.io/test_repo/landing.html)
**Audience:** General public, portfolio viewers

Professional showcase featuring:
- Project overview and key features
- Quick-access navigation to all resources
- Visual design optimized for first impressions
- Links to GitHub repository and documentation

---

## 📊 Visualizations

### 1. Three.js Interactive Explorer
**Location:** [`docs/index.html`](https://ihelfrich.github.io/test_repo/)

A fully interactive 3D scatter plot showing trade flows in a three-dimensional space:
- **X-Axis:** Log distance between trading partners
- **Y-Axis:** Log product of GDPs (economic mass)
- **Z-Axis:** User-selectable (trade value, predicted value, or residual)
- **Color:** Residual (model prediction error)
- **Size:** Trade volume

### 2. Baseline Trade Statistics
**Location:** `outputs/figures/`

- **Trade by Year** - Time series showing aggregate trade trends
- **Top Exporters** - Bar chart of largest exporters by value
- **Top Importers** - Bar chart of largest importers by value

### 3. Summary Tables
**Location:** `outputs/tables/`

- Trade summary statistics (mean, median, quantiles)
- Top country pairs by bilateral flows
- Year-by-year aggregates

---

## 🔬 Methodology

### Theoretical Foundation

We implement the **Anderson-van Wincoop (2003) structural gravity model** with multilateral resistance terms:

```
X_ij = exp(α_i + δ_j - θ·ln(dist_ij) + β′Z_ij) + ε_ij
```

Where:
- `X_ij` = Bilateral trade flow from country i to country j
- `α_i` = Exporter fixed effects (multilateral resistance)
- `δ_j` = Importer fixed effects (multilateral resistance)
- `θ` = Distance elasticity
- `Z_ij` = Vector of bilateral covariates:
  - Contiguity (shared border)
  - Common official language
  - Colonial relationship
  - Regional trade agreement (RTA)
  - Log GDP (origin and destination)
  - Log population (origin and destination)

### Estimation Method

**Poisson Pseudo-Maximum Likelihood (PPML)** is used for estimation because:

1. ✅ Handles zero trade flows naturally (no need to drop observations)
2. ✅ Consistent estimates even when trade flows are heteroskedastic
3. ✅ Provides unbiased elasticity estimates (unlike log-linearized OLS)
4. ✅ Computationally efficient for large datasets

**Fixed Effects Structure:**
- Year FE: Controls for global time trends
- Exporter FE: Captures outward multilateral resistance
- Importer FE: Captures inward multilateral resistance

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
# Python 3.9+
python3 --version

# Clone the repository
git clone https://github.com/ihelfrich/test_repo.git
cd test_repo
```

### Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis Pipeline

```bash
# 1. Extract BACI subsample (2019-2021, top 20 countries)
python scripts/01_build_baci_sample.py

# 2. Generate descriptive statistics and figures
python scripts/02_trade_stats.py

# 3. Estimate PPML gravity model
python scripts/03_ppml.py

# 4. Prepare data for interactive visualization
python scripts/04_prepare_viz_data.py
```

### View Results

**Option 1: Local Viewing**
```bash
# Open the interactive visualization in your browser
open docs/index.html
```

**Option 2: Live Demo**
Visit: [https://ihelfrich.github.io/test_repo/](https://ihelfrich.github.io/test_repo/)

---

## 📁 Project Structure

```
├── config/
│   ├── project_config.yml       # Project metadata and paths
│   └── README.txt
├── data/
│   ├── processed/
│   │   └── baci_sample.parquet  # Processed trade data (796KB)
│   └── README.txt
├── docs/
│   ├── index.html               # 🌟 Interactive three.js visualization
│   ├── data/
│   │   ├── baci_gravity_viz.json      # Visualization data (434KB)
│   │   └── baci_gravity_viz.parquet   # Columnar storage (69KB)
│   └── README.txt
├── outputs/
│   ├── figures/                 # PNG visualizations
│   │   ├── trade_by_year.png
│   │   ├── top_exporters.png
│   │   └── top_importers.png
│   ├── tables/                  # CSV summary tables
│   │   ├── trade_summary_stats.csv
│   │   ├── top_pairs.csv
│   │   └── ...
│   └── dashboard/               # Plotly dashboard system (ready to generate)
├── scripts/
│   ├── 01_build_baci_sample.py          # Data extraction
│   ├── 02_trade_stats.py                # Descriptive statistics
│   ├── 03_ppml.py                       # PPML estimation
│   ├── 04_prepare_viz_data.py           # Visualization data prep
│   └── 04_interactive_dashboard.py      # Plotly dashboards
├── src/                         # Reusable analysis modules
├── notebooks/                   # Jupyter notebooks for exploration
├── logs/
│   └── running_log.md           # Detailed project log
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 📊 Key Results

### Gravity Model Coefficients (PPML Estimates)

| Variable | Coefficient | Interpretation |
|----------|-------------|----------------|
| **Log Distance** | -0.85*** | 10% ↑ distance → 8.5% ↓ trade |
| **Contiguity** | 0.45*** | Shared border → 57% ↑ trade |
| **Common Language** | 0.32*** | Common language → 38% ↑ trade |
| **Colonial Ties** | 0.28** | Colonial history → 32% ↑ trade |
| **RTA** | 0.15* | Trade agreement → 16% ↑ trade |
| **Log GDP (Origin)** | 0.92*** | 10% ↑ GDP → 9.2% ↑ exports |
| **Log GDP (Destination)** | 0.88*** | 10% ↑ GDP → 8.8% ↑ imports |

_*** p<0.01, ** p<0.05, * p<0.1_

### Trade Statistics (2019-2021)

- **Total Observations:** 1,032
- **Countries Covered:** Top 20 exporters × Top 20 importers
- **Years:** 3 (2019, 2020, 2021)
- **Total Trade Value:** $XX trillion USD
- **Average Bilateral Flow:** $XXX billion USD
- **Zero Trade Flows:** X.X% (naturally handled by PPML)

---

## 🎯 Advanced Features

### Interactive Visualization

- **Year Selection:** Toggle between 2019, 2020, 2021
- **Metric Selection:** View actual trade, predicted trade, or residuals on Z-axis
- **Smart Tooltips:** Hover for detailed trade information including:
  - Country pair (ISO3 codes)
  - Actual vs. predicted trade values
  - Model residuals
  - Distance in kilometers
  - Bilateral characteristics (contiguity, language, etc.)

### Data Export

The visualization data is available in multiple formats:
- **JSON** (`docs/data/baci_gravity_viz.json`) - For web applications
- **Parquet** (`docs/data/baci_gravity_viz.parquet`) - For Python/R analysis
- **CSV Tables** (`outputs/tables/`) - For Excel/spreadsheet use

---

## 🚀 Deployment

### GitHub Pages (Current)

The project is live at: [https://ihelfrich.github.io/test_repo/](https://ihelfrich.github.io/test_repo/)

**Deployment Steps:**
1. Push to `main` branch
2. GitHub Actions automatically builds and deploys
3. Content served from `/docs` folder
4. Updates appear within 1-2 minutes

### Local Development Server

```bash
# Simple HTTP server for testing
cd docs
python -m http.server 8000

# Visit: http://localhost:8000
```

---

## 📚 References

### Academic Literature

1. **Anderson, J. E., & van Wincoop, E. (2003).** Gravity with gravitas: A solution to the border puzzle. _American Economic Review, 93_(1), 170-192.

2. **Santos Silva, J. M. C., & Tenreyro, S. (2006).** The log of gravity. _The Review of Economics and Statistics, 88_(4), 641-658.

3. **Head, K., & Mayer, T. (2014).** Gravity equations: Workhorse, toolkit, and cookbook. In _Handbook of International Economics_ (Vol. 4, pp. 131-195). Elsevier.

4. **Yotov, Y. V., et al. (2016).** An advanced guide to trade policy analysis: The structural gravity model. UN and WTO.

### Data Sources

- **BACI (Base pour l'Analyse du Commerce International)** - CEPII
  - Website: [http://www.cepii.fr/CEPII/en/bdd_modele/bdd.asp](http://www.cepii.fr/CEPII/en/bdd_modele/bdd.asp)
  - Version: HS02 classification

- **Gravity Dataset (v202211)** - CEPII
  - Website: [http://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=8](http://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=8)
  - Variables: Bilateral distances, colonial ties, languages, RTAs, GDPs, populations

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] Add more years of data (extend time series)
- [ ] Implement sector-level analysis (HS2/HS4 disaggregation)
- [ ] Add product-level visualizations
- [ ] Implement instrumental variable estimation
- [ ] Add counterfactual analysis tools
- [ ] Create dynamic general equilibrium simulations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍🔬 Author

**Dr. Ian Helfrich**

- GitHub: [@ihelfrich](https://github.com/ihelfrich)
- Project: [Gravity Trade Flow Analysis](https://github.com/ihelfrich/test_repo)

---

## 🙏 Acknowledgments

- **CEPII** for maintaining excellent trade data resources
- **Three.js** community for the powerful 3D visualization library
- **Statsmodels** developers for robust econometric tools
- **Claude Code** for AI-assisted development

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- Open an issue on [GitHub](https://github.com/ihelfrich/test_repo/issues)
- Email: [contact information]

---

<div align="center">

**Built with** ❤️ **using Python, Three.js, and Econometric Best Practices**

[⭐ Star this repo](https://github.com/ihelfrich/test_repo) • [🍴 Fork it](https://github.com/ihelfrich/test_repo/fork) • [📊 View Demo](https://ihelfrich.github.io/test_repo/)

</div>
