# 🚀 Interactive Gravity Model Platform - Expansion Plan

**Current State:** Proof-of-concept with 3 years (2019-2021), top 20 countries, single Anderson-van Wincoop specification

**Target State:** World-class interactive research platform for gravity model analysis

---

## 📊 Phase 1: Data Expansion (Foundation)

### 1.1 Temporal Coverage Expansion
**Current:** 3 years (2019-2021)
**Target:** Full time series (2000-2023, 24 years)

**Implementation:**
- Modify `scripts/01_build_baci_sample.py` to extract all available years from BACI
- Update data processing pipeline to handle ~8x more observations
- Implement time-series visualization components (animated timeline, trend analysis)

**Benefits:**
- Long-run gravity elasticity estimation
- Structural break analysis (e.g., 2008 financial crisis, COVID-19)
- Dynamic panel methods

### 1.2 Country Coverage Expansion
**Current:** Top 20 × Top 20 (400 dyads per year, 1,200 total)
**Target:** All available countries (~200 × 200 = 40,000 dyads per year, ~1M observations)

**Implementation:**
- Remove `top_n` filter in data extraction script
- Implement server-side filtering/pagination for web interface
- Add geographic filtering (by continent, region, income level)

**Benefits:**
- Comprehensive trade pattern analysis
- Small economy dynamics
- Regional trade agreement impact assessment

### 1.3 Sector-Level Disaggregation
**Current:** Aggregate bilateral trade flows
**Target:** HS2 (97 sectors) and HS4 (~1,200 products) level analysis

**Data Sources:**
- `/Users/ian/trade_data_warehouse/baci/bilateral_sector_flows.parquet` (HS2 level)
- `/Users/ian/trade_data_warehouse/baci/hs_by_dyad/` directory (HS4 detailed)
- `/Users/ian/trade_data_warehouse/baci/product_codes_hs02.parquet` (sector descriptions)

**Implementation:**
- Create new script: `scripts/05_build_sector_data.py`
- Add sector selection dropdown to web interface
- Implement product-specific gravity variables (e.g., distance elasticity varies by sector)

**Benefits:**
- Heterogeneous trade cost effects across products
- Comparative advantage analysis
- Product-level counterfactuals (e.g., "What if tariffs on automobiles increased 10%?")

---

## 🔬 Phase 2: Model Specification Expansion

### 2.1 Multiple Gravity Model Variants

**Current:** Single Anderson-van Wincoop (2003) PPML with exporter/importer/year FE

**Target:** User-selectable model specifications

#### Model A: Anderson-van Wincoop (2003) - Structural Gravity
```
X_ij = exp(α_i + δ_j − θ·ln(dist) + β′Z_ij) + ε_ij
```
- Multilateral resistance via FE
- PPML estimation
- **Status:** ✅ Currently implemented

#### Model B: Head-Mayer (2014) - Toolkit Specification
```
X_ij = exp(α_i^t + δ_j^t + γ_ij) + ε_ij
where γ_ij = bilateral gravity variables
```
- Exporter-year and importer-year FE (captures time-varying multilateral resistance)
- Pair FE option for panel data
- PPML estimation

#### Model C: Yotov et al. (2016) - Advanced Guide Specification
```
X_ij = exp(α_i^t + δ_j^t + η_ij + φ·BORDER_ij^t + ρ·RTA_ij^t) + ε_ij
```
- Exporter-year FE, importer-year FE, pair FE
- Time-varying policy variables (RTAs, borders)
- Enables GE counterfactuals

#### Model D: Instrumental Variables (IV) Estimation
- Address endogeneity of RTAs (selection bias)
- Instruments: Geographic proximity to major trading blocs, colonial ties
- 2SLS or Control Function approach

### 2.2 Robustness Checks & Diagnostics

**Implemented Automatically:**
- Standard errors: Clustered (by exporter, importer, dyad, multi-way)
- Heteroskedasticity tests
- Reset test for functional form
- Sensitivity to zero flows
- Comparison: PPML vs. OLS on log(X+1) vs. Heckman selection

---

## 🌐 Phase 3: General Equilibrium (GE) Counterfactuals

### 3.1 Partial Equilibrium (Current)
**What it does:**
- Changes trade costs (distance, borders, RTAs)
- Holds fixed effects constant (no re-balancing)
- Shows immediate "first-order" trade flow response

**Limitations:**
- Ignores price adjustments
- No equilibrium constraints (trade doesn't balance)
- Unrealistic for large shocks

### 3.2 Structural General Equilibrium (Target)

**Ge-PPML Implementation (Larch & Yotov 2016):**

```python
# Iterative algorithm:
1. Estimate baseline gravity model → get trade costs τ_ij
2. Apply counterfactual shock → τ_ij^CF
3. Solve for new prices/wages satisfying:
   - Market clearing: ∑_j X_ij = Y_i (output = exports)
   - Trade balance: ∑_i X_ij = E_j (expenditure = imports)
4. Compute welfare changes: ΔW_i = (P_i^CF / P_i)^(-1/(σ-1))
```

**Features:**
- Endogenous trade costs (multilateral resistance updates)
- Welfare decomposition (terms of trade, variety, volume effects)
- Factory-gate vs. consumer price changes
- Real wage effects by country

**Use Cases:**
- Brexit impact on UK and EU welfare
- US-China tariff war simulation
- RCEP formation effects on non-members

---

## 💻 Phase 4: User Interface Enhancements

### 4.1 Advanced Filtering & Selection
- **Country picker:** Multi-select dropdown with search
- **Year range slider:** Select continuous range (e.g., 2010-2020)
- **Sector filter:** Browse HS2/HS4 product hierarchy
- **Trade flow thresholds:** Hide flows below $X million

### 4.2 Model Comparison Dashboard
Split-screen view:
- Left: Model A results (e.g., AvW 2003)
- Right: Model B results (e.g., Yotov 2016)
- Side-by-side coefficient comparison
- Elasticity differences highlighted

### 4.3 Time Series Animation
- Play button to animate trade flows over years
- See how trade patterns evolve (e.g., China's rise 2000-2020)
- Pause at key years (2008 crisis, 2020 pandemic)

### 4.4 3D Visualization Enhancements
**Current:** Static point cloud with X=distance, Y=GDP product, Z=metric

**Additions:**
- **Network view:** Countries as nodes, trade flows as edges (thickness = volume)
- **Globe view:** Geographic projection with arc-based flow visualization
- **Heatmap:** Matrix of origin × destination with color intensity = trade value
- **Sector comparison:** Multiple 3D plots for different products

### 4.5 Export & Reproducibility
- **Download data:** CSV, Excel, Stata .dta, R .rds formats
- **Download regression tables:** LaTeX, Word, HTML
- **Export figures:** PNG, SVG, interactive HTML (Plotly standalone)
- **API endpoint:** RESTful API for programmatic access
  ```
  GET /api/v1/gravity?year=2020&origin=USA&dest=CHN&model=yotov
  ```

---

## 🚀 Phase 5: Performance & Scalability

### 5.1 Data Size Challenges

**Current:**
- 1,200 observations (3 years × 400 dyads)
- ~434KB JSON payload
- Loads instantly in browser

**Target (All countries × all years):**
- ~1,000,000 observations (24 years × 40,000 dyads)
- Estimated 300MB+ JSON payload
- **Problem:** Cannot load entire dataset in browser

### 5.2 Solutions

#### Backend API with Filtering
**Architecture:**
```
Frontend (Three.js viz) → API Server (Flask/FastAPI) → DuckDB/Parquet
```

**Workflow:**
1. User selects filters (year, countries, sector)
2. Frontend sends API request
3. Backend queries parquet file (fast columnar reads)
4. Returns filtered JSON (~50KB typical response)
5. Frontend renders visualization

**Benefits:**
- Instant loading (only load what's needed)
- Scalable to billions of observations
- Can add authentication, rate limiting

#### Web Workers for Client-Side Computation
- Offload heavy computation (GE counterfactuals) to Web Worker threads
- Keep UI responsive during calculations
- Progress bar for iterative GE algorithm

#### Lazy Loading / Pagination
- Load data in chunks (e.g., one year at a time)
- Infinite scroll for leaderboard tables
- Pre-fetch next year in background

---

## 📚 Phase 6: Documentation & Pedagogy

### 6.1 Interactive Tutorials
- **"What is a gravity model?"** - Step-by-step explanation with live examples
- **"Understanding multilateral resistance"** - Visual demonstration
- **"GE vs. PE counterfactuals"** - Side-by-side comparison with sliders

### 6.2 Case Studies
- **Case 1:** Brexit impact (2016-2023)
- **Case 2:** NAFTA/USMCA effects (1994-2023)
- **Case 3:** China's WTO accession (2001)
- **Case 4:** COVID-19 trade collapse and recovery (2020-2023)

### 6.3 Replication Packages
- Jupyter notebooks for each case study
- Downloadable data + code
- One-click "Run in Google Colab" buttons

---

## 🎯 Implementation Priorities

### **Sprint 1 (Week 1): Data Foundation** ⭐ HIGH PRIORITY
- [ ] Expand year coverage to 2000-2023
- [ ] Remove top-20 filter (all countries)
- [ ] Create `scripts/05_build_full_dataset.py`
- [ ] Optimize parquet storage (compression, partitioning)

### **Sprint 2 (Week 2): Backend API**
- [ ] Set up Flask/FastAPI server
- [ ] Implement filtering endpoints
- [ ] Add DuckDB query engine
- [ ] Deploy API to cloud (Heroku, Railway, or Vercel)

### **Sprint 3 (Week 3): Model Variants**
- [ ] Implement Head-Mayer specification
- [ ] Implement Yotov specification
- [ ] Add model selection dropdown to UI
- [ ] Create coefficient comparison table

### **Sprint 4 (Week 4): Sector-Level Analysis**
- [ ] Extract HS2 sector data
- [ ] Add sector filter to UI
- [ ] Product-specific visualizations
- [ ] Sector elasticity estimates

### **Sprint 5 (Week 5): GE Counterfactuals**
- [ ] Implement Ge-PPML algorithm (Python backend)
- [ ] Welfare calculation module
- [ ] Terms-of-trade decomposition
- [ ] UI for GE results display

### **Sprint 6 (Week 6): Polish & Deploy**
- [ ] Performance optimization (caching, CDN)
- [ ] Mobile responsive design
- [ ] Accessibility (WCAG 2.1 AA)
- [ ] SEO optimization
- [ ] Google Analytics integration

---

## 📦 Tech Stack Recommendations

### Current Stack
✅ **Frontend:** Three.js, vanilla JavaScript, GitHub Pages
✅ **Data:** Parquet (Python) → JSON (web)
✅ **Estimation:** Statsmodels GLM (PPML)

### Proposed Additions

**Backend API:**
- **Framework:** FastAPI (Python) - Fast, modern, auto-docs
- **Database:** DuckDB (in-process OLAP, perfect for parquet)
- **Deployment:** Railway.app or Render.com (free tier available)

**Advanced Estimation:**
- **Econometrics:** `linearmodels` (panel IV, clustering)
- **GE Solver:** Custom implementation following Larch & Yotov (2016)
- **Parallel processing:** `joblib` or `dask` for large datasets

**Frontend Enhancements:**
- **Charting:** Plotly.js (for 2D charts, time series)
- **UI Components:** Shoelace (web components, modern, accessible)
- **State Management:** Zustand or Jotai (lightweight React-like state)

---

## 🎓 Educational Value

This platform would be suitable for:

1. **Graduate courses** in international trade
   - Students explore gravity models interactively
   - Run their own counterfactuals
   - Reproduce published papers

2. **Policy analysis** workshops
   - Trade negotiators simulate RTA effects
   - Central banks assess trade shocks
   - Development agencies evaluate integration policies

3. **Research dissemination**
   - Published papers include interactive visualizations
   - Replication packages with web interfaces
   - Public engagement with research findings

---

## 📈 Success Metrics

- **Usage:** 1,000+ monthly active users
- **Citations:** Cited in 10+ academic papers
- **Performance:** <2s load time for any query
- **Coverage:** 200+ countries, 24+ years, 97+ sectors
- **Models:** 4+ gravity specifications, GE counterfactuals
- **Engagement:** Average session >5 minutes, >3 counterfactuals per session

---

## 🔗 References

- Anderson, J. E., & van Wincoop, E. (2003). Gravity with gravitas. *AER*, 93(1).
- Head, K., & Mayer, T. (2014). Gravity equations: Workhorse, toolkit, and cookbook. *Handbook of IE*, Vol. 4.
- Yotov, Y. V., et al. (2016). *An Advanced Guide to Trade Policy Analysis.* WTO/UNCTAD.
- Larch, M., & Yotov, Y. V. (2016). General equilibrium trade policy analysis with structural gravity. *CESifo Working Paper*.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *RESTAT*, 88(4).

---

**Document Status:** Draft v1.0
**Last Updated:** 2026-01-14
**Author:** Dr. Ian Helfrich
**Next Review:** After Sprint 1 completion
