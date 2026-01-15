# International Trade Economics: Complete Learning Portal
**A Comprehensive Guide to Gravity Models and Topological Methods**

**Last Updated**: 2026-01-15

Welcome to the complete learning portal for modern international trade economics. This portal provides rigorous, publication-grade educational materials covering both traditional econometric methods and cutting-edge topological approaches.

---

## 📚 Table of Contents

1. [Learning Paths](#learning-paths)
2. [Core Theory Documents](#core-theory-documents)
3. [Practical Implementation](#practical-implementation)
4. [Interactive Demonstrations](#interactive-demonstrations)
5. [Research Applications](#research-applications)
6. [Additional Resources](#additional-resources)

---

## Learning Paths

### Path 1: Graduate Student / Researcher (New to Gravity Models)

**Week 1-2: Foundations**
1. Read [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1) - Sections 1-4
   - Historical development
   - Theoretical foundations (Armington, CES)
   - Structural gravity models (Anderson-van Wincoop)
   - Basic estimation (OLS vs PPML)

**Week 3-4: Advanced Estimation**
2. Read [ESTIMATION_METHODS.md](ESTIMATION_METHODS.md:1) - Complete
   - PPML implementation details
   - High-dimensional fixed effects
   - Standard errors and clustering
   - Model diagnostics

**Week 5-6: Extensions**
3. Read [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1) - Sections 5-8
   - Multilateral resistance (mathematical derivation)
   - Extensions (Melitz, services, dynamic)
   - Empirical applications
   - Best practices

**Week 7-8: Topological Methods**
4. Read [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1) - Sections 1-6
   - Field construction from residuals
   - Persistent homology basics
   - Wasserstein distance
   - MDS embeddings

**Week 9-10: Network Analysis**
5. Read [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1) - Sections 7-9
   - Critical slowing down
   - Network science foundations
   - Empirical applications (COVID, Brexit)

**Week 11-12: Implementation**
6. Work through [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md:1)
   - Run scripts 01-05 (data pipeline)
   - Generate topology fields (script 09)
   - Compute network metrics (script 14)
   - Validate pipeline (script 13)

### Path 2: Practicing Economist (Refresher on Modern Methods)

**Fast Track (1-2 weeks)**:
1. [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1) - Section 4 (Estimation Methods)
2. [ESTIMATION_METHODS.md](ESTIMATION_METHODS.md:1) - Sections 3-4 (PPML, Fixed Effects)
3. [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1) - Section 5 (Wasserstein Distance)
4. [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md:1) - Section on Scripts 13-15

**Focus**: Modern best practices, topological methods for structural break detection

### Path 3: Undergraduate / General Interest

**Conceptual Understanding (no math)**:
1. [CURRENT_STATUS.md](../CURRENT_STATUS.md:1) - Overview of what's implemented
2. Interactive visualizations:
   - [Gravity Explorer](https://ihelfrich.github.io/test_repo/)
   - [Topology Signals](https://ihelfrich.github.io/test_repo/topology.html)
   - [Research Lab](https://ihelfrich.github.io/test_repo/advanced_topology.html)
3. [INNOVATION_SUMMARY.md](../INNOVATION_SUMMARY.md:1) - High-level research contributions

**Gradual Math Introduction**:
1. [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1) - Sections 1-2 (conceptual)
2. [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1) - Section 1 (motivation)

---

## Core Theory Documents

### 1. [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1) (15,000+ words)

**Complete coverage of structural gravity models**

**Contents**:
1. **Historical Development** (Tinbergen 1962 → Anderson-van Wincoop 2003)
   - Evolution of gravity equation
   - Empirical puzzles (border effects, missing trade)
   - Theoretical breakthroughs

2. **Theoretical Foundations**
   - Armington assumption (product differentiation)
   - CES utility function
   - Budget constraints and market clearing
   - Derivation of bilateral trade equation

3. **Structural Gravity Models**
   - Anderson-van Wincoop (2003) - Full derivation
   - Head-Mayer (2014) approximations
   - Comparison table of specifications

4. **Estimation Methods**
   - OLS problems (Jensen's inequality, zeros, heteroskedasticity)
   - PPML solution (Santos Silva & Tenreyro 2006)
   - Implementation with code examples

5. **Multilateral Resistance**
   - Economic intuition (general equilibrium effects)
   - Mathematical derivation (system of equations)
   - Solving algorithms (fixed-point iteration)
   - Connection to network centrality

6. **Extensions**
   - Melitz (2003) firm heterogeneity
   - Intermediate goods and value chains
   - Services trade
   - Dynamic models

7. **Empirical Applications**
   - Regional trade agreements (RTAs)
   - Border effects
   - Currency unions
   - Trade wars and tariffs

8. **Best Practices**
   - PPML with high-dimensional FE
   - Cluster-robust standard errors
   - Common mistakes to avoid
   - Reporting standards

**Key Features**:
- Full mathematical derivations
- Python code examples throughout
- Comparison tables
- Historical context
- Publication-quality rigor

**Prerequisites**:
- Intermediate microeconomics
- Basic calculus (derivatives, optimization)
- Linear algebra (matrix notation)

**Time to Master**: 2-3 weeks intensive study

---

### 2. [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1) (12,000+ words)

**Rigorous mathematical foundations for topological methods**

**Contents**:
1. **Introduction and Motivation**
   - Why topology for trade?
   - Core innovation: residuals as fields
   - Theoretical precedents

2. **Mathematical Preliminaries**
   - Metric spaces and distance functions
   - Topological spaces (formal definition)
   - Homology groups and Betti numbers
   - Gradient, divergence, Laplacian

3. **From Discrete Residuals to Continuous Fields**
   - Multidimensional Scaling (MDS) algorithm
   - Gridding and interpolation (weighted Gaussian)
   - Smoothing techniques
   - Field statistics (variance, energy, autocorrelation)

4. **Persistent Homology Theory**
   - Filtrations and persistence
   - Persistence diagrams and barcodes
   - Betti numbers (formal definition via chain complexes)
   - Computational implementation (Ripser)
   - Statistical inference (bootstrap tests)

5. **Optimal Transport and Wasserstein Distance**
   - Monge-Kantorovich problem (mathematical setup)
   - Wasserstein-p distances
   - Discrete implementation (Earth Mover's Distance)
   - Kantorovich-Rubinstein theorem
   - Trade applications (year-to-year shifts, event detection)

6. **Multidimensional Scaling and Embeddings**
   - Classical MDS (eigendecomposition)
   - Metric MDS (stress minimization)
   - t-SNE and UMAP (alternatives)
   - Trade-specific distance metrics

7. **Critical Slowing Down Detection**
   - Theory of early warning signals
   - Indicators (autocorrelation, variance, skewness)
   - Application to trade networks
   - Robustness checks

8. **Connection to Gravity Models**
   - Residuals as topological data
   - Augmented gravity with network features
   - Multilateral resistance vs. PageRank
   - Structural break detection

9. **Network Science Foundations**
   - Graph representations
   - Centrality measures (degree, PageRank, betweenness)
   - Clustering and community detection
   - Network evolution models
   - Assortativity

10. **Empirical Applications**
    - COVID-19 shock analysis
    - Brexit network position shift
    - US-China trade war
    - RTA impact with topology

11. **Computational Implementation**
    - Complete data pipeline (Python)
    - Validation tests
    - Performance optimization

12. **Best Practices and Limitations**
    - When to use each method
    - Known limitations
    - Robustness checks

**Key Features**:
- Formal mathematical definitions
- Full algorithm derivations
- Implementation code (Python)
- Real-world applications validated on data
- Connection to gravity theory

**Prerequisites**:
- Linear algebra (eigenvectors, matrix decomposition)
- Multivariable calculus (gradients, optimization)
- Basic topology (optional but helpful)
- Python programming

**Time to Master**: 3-4 weeks intensive study

---

### 3. [ESTIMATION_METHODS.md](ESTIMATION_METHODS.md:1) (10,000+ words)

**Complete practitioner's guide to econometric estimation**

**Contents**:
1. **Why Estimation Method Matters**
   - Jensen's inequality problem
   - Zero trade problem
   - Heteroskedasticity
   - Summary comparison table

2. **Ordinary Least Squares (OLS)**
   - Basic specification
   - OLS estimator (closed form)
   - Implementation (Python)
   - Interpretation
   - Why OLS fails for gravity

3. **Poisson Pseudo-Maximum Likelihood (PPML)**
   - Poisson model setup
   - Log-likelihood derivation
   - First-order conditions
   - Consistency under misspecification
   - Handling zeros naturally
   - IRLS algorithm
   - Implementation (Python with statsmodels)
   - Interpretation

4. **Fixed Effects Implementation**
   - Why fixed effects?
   - Exporter-importer FE
   - Theory connection (multilateral resistance)
   - Exporter-year and importer-year FE
   - High-dimensional FE (ppmlhdfe, fixest)
   - Degrees of freedom adjustment
   - Cluster-robust SEs

5. **Two-Stage Least Squares (2SLS)**
   - Endogeneity in gravity (RTAs, migration)
   - Instrumental variables (requirements)
   - 2SLS procedure
   - Implementation
   - Hausman test for endogeneity
   - Weak instruments problem

6. **Tetrads Method**
   - Motivation (eliminate FE via differencing)
   - Tetrad transformation (mathematical derivation)
   - Estimation procedure
   - Advantages and disadvantages

7. **Bonus Vetus OLS (BVO)**
   - The BVO estimator
   - Performance (Monte Carlo results)
   - When to use

8. **Structural Estimation**
   - Counterfactual analysis
   - General equilibrium system (equations)
   - Solving algorithm (Anderson & Yotov 2016)
   - Implementation (Python solver)
   - Welfare analysis

9. **Standard Errors and Inference**
   - Heteroskedasticity-robust SEs (White 1980)
   - Clustered SEs (by dyad)
   - Multi-way clustering
   - Bootstrap SEs

10. **Model Selection and Diagnostics**
    - Pseudo-R² for PPML
    - Information criteria (AIC, BIC)
    - Specification tests (RESET, link test)
    - Residual diagnostics (QQ-plot, Breusch-Pagan, Durbin-Watson)

11. **Computational Implementation**
    - Complete PPML example (with code)
    - Performance optimization (Numba, sparse matrices)

12. **Best Practices**
    - Estimation checklist
    - Common mistakes
    - Robustness checks
    - Reporting standards (table format)

**Key Features**:
- Step-by-step algorithms
- Complete Python implementations
- Diagnostic procedures
- Real-world best practices
- Publication-ready table formats

**Prerequisites**:
- Econometrics (OLS, MLE)
- Basic probability theory
- Python programming

**Time to Master**: 2-3 weeks

---

## Practical Implementation

### [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md:1) (9,000+ words)

**Complete usage guide for the analysis pipeline**

**Contents**:
1. **Currently Implemented & Working**
   - Data pipeline (Scripts 01-05)
   - Topology & advanced analysis (Scripts 08-12)
   - New analysis scripts (13-15)

2. **Interactive Visualizations**
   - Gravity Explorer
   - Topology Signals
   - Research Lab
   - Other pages

3. **Data Files**
   - Visualization data (`docs/data/`)
   - Processed data (`data/processed/`)
   - File sizes and purposes

4. **What's Actually Novel & Validated**
   - Data-backed topology fields
   - Wasserstein distance for trade shifts
   - Multi-model gravity framework
   - Residual dispersion as stability metric

5. **Key Empirical Results**
   - Gravity model estimates
   - Temporal trends
   - Network structure metrics

6. **Known Limitations & Gaps**
   - Dataset scope
   - Temporal frequency
   - Product disaggregation
   - Advanced methods status
   - Validation gaps

7. **Next Steps (Prioritized)**
   - High, medium, low priority tasks

8. **Documentation Map**
   - Links to all project documents

9. **Development Workflow**
   - Adding new analysis
   - Fixing bugs
   - Deploying changes

**Use Cases**:
- Running the complete pipeline
- Understanding data outputs
- Extending with new analysis
- Validating results
- Deploying to production

**Prerequisites**: Python programming, basic command line

**Time to Complete**: 1-2 days to run full pipeline

---

### [CURRENT_STATUS.md](../CURRENT_STATUS.md:1) (3,000+ words)

**Quick status reference**

**Contents**:
- What's working right now (data pipeline, visualizations, scripts)
- Latest improvements (validation, network metrics, master analysis)
- Key empirical results
- What's actually novel (validated methods)
- Known issues and limitations
- Next actions (immediate, short-term, medium-term)
- Key insights from latest analysis
- Documentation status
- Research contributions (validated vs. planned)

**Use Case**: Quick overview before diving into details

**Time to Read**: 15-20 minutes

---

## Interactive Demonstrations

### 1. [Gravity Explorer](https://ihelfrich.github.io/test_repo/)

**Interactive 3D visualization of gravity models**

**Features**:
- Multi-model selection (4 gravity specifications)
- Counterfactual sliders with model-consistent coefficients
- Dynamic model insights
- Three.js 3D visualization
- Dyad-level explanations

**Learning Outcomes**:
- Understand how distance affects trade
- See impact of RTAs, borders, language
- Compare model specifications
- Intuition for multilateral resistance

**Time**: 30-45 minutes exploration

---

### 2. [Topology Signals](https://ihelfrich.github.io/test_repo/topology.html)

**Real-time topology field visualization**

**Features**:
- Data-driven residual fields by year (2005-2021)
- Real-time diagnostics (energy, variance, autocorrelation)
- Betti number proxies (connected components)
- Shock controls (China, US-tariff, Brexit, random)
- Multiple display modes (field, gradient, laplacian, energy)
- 5 colormaps

**Learning Outcomes**:
- Visualize spatial patterns in trade deviations
- Understand field statistics
- See impact of shocks on topology
- Intuition for Betti numbers

**Time**: 1 hour exploration

---

### 3. [Research Lab](https://ihelfrich.github.io/test_repo/advanced_topology.html)

**Four research methods with year-by-year data**

**Features**:
1. Residual Dispersion (std of gravity residuals)
2. Transport Shift (Wasserstein-1 distance)
3. Network Concentration (HHI metrics)
4. Topology Field (variance and Betti proxies)

- Year-by-year charts with trends
- Dynamic explanations
- Real data from 2005-2021

**Learning Outcomes**:
- Compare different topological methods
- Identify crisis years (COVID, Brexit, Greece)
- Understand trend analysis
- See methods in action on real data

**Time**: 45-60 minutes

---

## Research Applications

### Case Studies (from TOPOLOGY_THEORY.md Section 10)

**1. COVID-19 Impact Analysis**
- Wasserstein distance spike in 2020
- Field variance increase (+40%)
- Betti-1 decrease (-30% in trade cycles)
- Interpretation: supply chain fragmentation

**2. Brexit Network Position Shift**
- UK's MDS position tracking (2015-2020)
- PageRank decrease (0.045 → 0.038)
- Betweenness centrality decrease (-25%)
- Procrustes distance test (p = 0.008)

**3. US-China Trade War**
- Trade diversion to Vietnam/Mexico
- PageRank of Vietnam +15% (2017-2019)
- Wasserstein distance comparison (affected vs. unaffected products)
- Network structure stability

**4. Regional Trade Agreement Impact**
- Augmented gravity with PageRank interaction
- RTA effect amplification for central countries
- Network spillover mechanisms

---

## Additional Resources

### Implemented Scripts

**Data Pipeline**:
- [scripts/01_build_baci_sample.py](../scripts/01_build_baci_sample.py:1): BACI extraction
- [scripts/02_trade_stats.py](../scripts/02_trade_stats.py:1): Descriptive statistics
- [scripts/03_ppml.py](../scripts/03_ppml.py:1): Gravity estimation
- [scripts/04_prepare_viz_data.py](../scripts/04_prepare_viz_data.py:1): Visualization prep
- [scripts/05_build_full_dataset.py](../scripts/05_build_full_dataset.py:1): Full dataset

**Topology & Analysis**:
- [scripts/09_build_topology_fields.py](../scripts/09_build_topology_fields.py:1): Field construction
- [scripts/11_unified_analysis_pipeline.py](../scripts/11_unified_analysis_pipeline.py:1): Research metrics
- [scripts/12_build_country_embeddings.py](../scripts/12_build_country_embeddings.py:1): MDS embeddings

**Validation & Metrics**:
- [scripts/13_validate_pipeline.py](../scripts/13_validate_pipeline.py:1): Automated testing
- [scripts/14_network_metrics.py](../scripts/14_network_metrics.py:1): Network analysis
- [scripts/15_master_analysis.py](../scripts/15_master_analysis.py:1): Comprehensive report

---

### Data Sources

**BACI** (Base pour l'Analyse du Commerce International)
- URL: http://www.cepii.fr/CEPII/en/bdd_modele/bdd.asp
- Coverage: Bilateral trade flows, 1995-2021
- Classification: HS02, HS92, HS96, HS02, HS07, HS12, HS17

**CEPII Gravity Dataset**
- URL: http://www.cepii.fr/CEPII/en/bdd_modele/presentation.asp?id=8
- Version: v202211
- Variables: Distance, contiguity, language, colonial ties, RTAs, GDP, population

---

### Key References

**Gravity Theory**:
- Anderson & van Wincoop (2003): "Gravity with Gravitas", *AER*
- Santos Silva & Tenreyro (2006): "The Log of Gravity", *REStat*
- Head & Mayer (2014): "Gravity Equations", *Handbook*

**Topology & Optimal Transport**:
- Edelsbrunner & Harer (2010): *Computational Topology*
- Villani (2009): *Optimal Transport: Old and New*
- Carlsson (2009): "Topology and Data", *Bull. AMS*

**Network Science**:
- Newman (2010): *Networks: An Introduction*
- Barabási (2016): *Network Science*

**Trade & Networks**:
- Chaney (2014): "The Network Structure of International Trade", *AER*
- Bernard et al. (2018): "Networks and Trade", *Annual Review*

---

## Summary

This learning portal provides:

✅ **Comprehensive Theory** (40,000+ words of rigorous mathematics)
✅ **Practical Implementation** (Complete Python pipeline with validation)
✅ **Interactive Demonstrations** (3 web-based visualizations)
✅ **Real-World Applications** (Validated on 2005-2021 trade data)
✅ **Multiple Learning Paths** (Undergraduate → PhD → Professional)

**Total Study Time**:
- **Fast Track (Professional Refresher)**: 1-2 weeks
- **Graduate Student (Complete Mastery)**: 10-12 weeks
- **Undergraduate (Conceptual Understanding)**: 4-6 weeks

**Pedagogical Philosophy**:
- Start with economic intuition
- Build mathematical rigor gradually
- Provide complete derivations (no "it can be shown")
- Implement everything in code
- Validate on real data
- Connect theory to practice

---

**Questions or Feedback?**

- Open an issue at [GitHub Issues](https://github.com/ihelfrich/test_repo/issues)
- See [PROJECT_STATUS.md](../PROJECT_STATUS.md:1) for current implementation status
- Check [INNOVATION_SUMMARY.md](../INNOVATION_SUMMARY.md:1) for research contributions

---

**Happy Learning!** 🎓📊🌍
