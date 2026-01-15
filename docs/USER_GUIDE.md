# Complete User Guide: Understanding the Visualizations
**Intuition, Math, and Interpretation**

**Last Updated**: 2026-01-15

This guide explains every visualization and concept in plain language, with mathematical foundations and practical interpretation tips.

---

## Table of Contents

1. [The Big Picture: What We're Doing](#1-the-big-picture-what-were-doing)
2. [Gravity Explorer (Main Tool)](#2-gravity-explorer-main-tool)
3. [Topology Signals](#3-topology-signals)
4. [Research Lab](#4-research-lab)
5. [Trade Map](#5-trade-map)
6. [Residual Surface](#6-residual-surface)
7. [How Everything Connects](#7-how-everything-connects)

---

## 1. The Big Picture: What We're Doing

### 1.1 The Core Question

**Simple version**: If we know the GDP of two countries and the distance between them, can we predict how much they trade?

**Answer**: Yes! This is called a "gravity model" (like Newton's law of gravitation, but for trade).

### 1.2 The Gravity Equation (Intuitive)

Countries trade more when:
- They're **bigger** (higher GDP) → more to buy and sell
- They're **closer** (lower distance) → cheaper to ship
- They **share characteristics** (same language, border, trade agreement) → easier to do business

**The basic formula**:
```
Trade between i and j ≈ (GDP_i × GDP_j) / Distance_ij
```

### 1.3 Why Topology?

The gravity model is pretty good, but it's not perfect. Countries don't trade exactly what the model predicts. The **differences** between actual and predicted trade are interesting:

- **Positive residual**: Countries trade MORE than expected (why?)
- **Negative residual**: Countries trade LESS than expected (why?)

**Topology methods** help us visualize and analyze these patterns of "weird" trade.

---

## 2. Gravity Explorer (Main Tool)

**URL**: `index.html` (the main interactive tool)

### 2.1 What You're Looking At

**The 3D Space**:
- **X-axis**: Log distance between countries
  - Left = close together, Right = far apart
- **Y-axis**: Log(GDP_i × GDP_j) - "economic mass"
  - Bottom = small economies, Top = large economies
- **Z-axis**: You choose! (trade value, predicted trade, or residual)
- **Color**: Residual (blue = under-trading, red = over-trading)
- **Size**: Trade volume (bigger = more trade)

### 2.2 Intuition: The 3D Cloud

**What the cloud shape tells you**:

1. **Downward slope (left to right)**:
   - Countries closer together trade more
   - The slope is the "distance elasticity" (typically -0.85)
   - Interpretation: 10% further away → 8.5% less trade

2. **Upward slope (bottom to top)**:
   - Bigger economies trade more
   - This is just GDP effect

3. **Scatter around the surface**:
   - Points above the surface = over-trading (positive residual)
   - Points below = under-trading (negative residual)
   - The scatter shows what the gravity model **can't** explain

### 2.3 The Math (Formal)

**Structural gravity model** (Anderson-van Wincoop 2003):

```
X_ij = (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ)
```

Where:
- `X_ij`: Exports from i to j
- `Y_i, Y_j`: GDP of countries i and j
- `Y^W`: World GDP
- `t_ij`: Trade costs (distance, tariffs, etc.)
- `Π_i`: "Outward multilateral resistance" (how hard it is for i to export to anyone)
- `P_j`: "Inward multilateral resistance" (how hard it is for j to import from anyone)
- `σ`: Elasticity of substitution (typically 5-7)

**In logs** (what we estimate):
```
ln(X_ij) = α_i + δ_j + β₁ ln(dist_ij) + β₂ contig_ij + β₃ comlang_ij + β₄ RTA_ij + ε_ij
```

Where:
- `α_i`: Exporter fixed effect (captures `Y_i` and `Π_i`)
- `δ_j`: Importer fixed effect (captures `Y_j` and `P_j`)
- `β₁`: Distance elasticity (typically -0.85)
- `β₂`: Border effect (sharing a border → ~60% more trade)
- `β₃`: Language effect (same language → ~40% more trade)
- `β₄`: RTA effect (trade agreement → ~15% more trade)
- `ε_ij`: **RESIDUAL** - what we can't explain

### 2.4 How to Use It

**Step 1: Choose a year** (dropdown)
- See how trade patterns evolved over time
- Compare 2019 (normal) vs 2020 (COVID) vs 2021 (recovery)

**Step 2: Choose Z-axis metric**
- **Actual trade**: See raw bilateral flows
- **Predicted trade**: See what the model says "should" happen
- **Residual**: See the difference (this is the interesting part!)

**Step 3: Explore patterns**
- **Hover over points**: See exact dyad info (Germany → France, etc.)
- **Rotate**: Get different angles on the 3D cloud
- **Look for outliers**: Points far from the surface are "weird" trade relationships

### 2.5 Interpretation Guide

**Examples**:

1. **USA → Canada**: Huge positive residual (red, high)
   - They trade WAY more than distance/GDP alone would predict
   - Why? Shared border, language, NAFTA, integrated supply chains
   - This is what `β₂ + β₃ + β₄` captures

2. **China → Brazil**: Near prediction (white/neutral)
   - Trade is about what you'd expect for their GDP and distance
   - No special relationship beyond gravity fundamentals

3. **Japan → Argentina**: Negative residual (blue, low)
   - Trade LESS than expected
   - Why? Different time zones, no trade agreement, different languages

**What to look for**:
- **Clusters of red points**: Trade blocs (EU, NAFTA, ASEAN)
- **Vertical shifts over time**: Changes in trade relationships
- **Outliers**: Countries with unusual trade patterns (embargoes, conflicts, unique partnerships)

---

## 3. Topology Signals

**URL**: `topology.html`

### 3.1 What Is a "Topology Field"?

**Simple explanation**:
Imagine the world as a flat map. At each point on the map, we color it based on whether nearby countries are over-trading (red) or under-trading (blue).

**Why it's called "topology"**:
- We're treating trade residuals as a **continuous field** (like temperature on a weather map)
- We can study the **shape** of this field (hills, valleys, peaks, troughs)
- Topology = the study of shapes that don't change when you stretch or bend them

### 3.2 How We Build the Field

**Step-by-step process**:

1. **Start with gravity residuals**: `r_ij = ln(X_ij) - ln(X̂_ij)` for each country pair

2. **Embed countries in 2D space**:
   - Use **Multidimensional Scaling (MDS)** to map each country to a 2D coordinate
   - Distance in 2D space ≈ actual geographic distance
   - Result: `{USA: (x₁, y₁), Germany: (x₂, y₂), ...}`

3. **Create a grid**: 64×64 points covering the 2D space

4. **Interpolate residuals onto grid**:
   - For each grid point `(x, y)`, look at nearby dyads
   - Weight residuals by distance using Gaussian kernel
   - Formula: `field(x,y) = Σ w_ij × r_ij / Σ w_ij`
   - Where `w_ij = exp(-d²/(2σ²))` and `d` = distance from `(x,y)` to dyad midpoint

5. **Smooth**: Apply Gaussian blur to reduce noise

**Result**: A 64×64 grid where each cell has a value (positive = over-trading region, negative = under-trading)

### 3.3 The Math (Formal)

**MDS embedding**:
```
min_{x₁,...,x_N ∈ ℝ²} Σ_{i,j} (d_ij^geo - ‖x_i - x_j‖)²
```

Finds 2D positions that best preserve geographic distances.

**Gridding (Gaussian interpolation)**:
```
r̂(x, y) = Σ_{all dyads (i,j)} exp(-‖(x,y) - (x_i+x_j)/2‖²/(2σ²)) × r_ij
         ────────────────────────────────────────────────────────────
                  Σ_{all dyads} exp(-‖(x,y) - (x_i+x_j)/2‖²/(2σ²))
```

**Smoothing (convolution)**:
```
r̃(x,y) = ∫∫ K_σ(‖(x,y)-(x',y')‖) r̂(x',y') dx' dy'
```
where `K_σ` is Gaussian kernel.

### 3.4 What You See in the Visualization

**Color map**:
- **Dark blue**: Strong under-trading (residuals around -0.5)
- **White/Gray**: Near prediction (residuals ≈ 0)
- **Bright red/yellow**: Strong over-trading (residuals around +0.5)

**Controls**:

1. **Year slider**: Watch the field evolve over time
   - 2005: Early patterns
   - 2020: COVID shock
   - 2021: Recovery

2. **Display mode**:
   - **Field**: Raw residual values
   - **Gradient**: Direction of steepest change (arrows pointing to high trade)
   - **Laplacian**: Curvature (detects peaks and valleys)
   - **Energy**: Gradient magnitude squared (where field is changing fastest)

3. **Shock scenarios**: What if...
   - **China shock**: Reduce China's exports by 30%
   - **US tariff shock**: Increase US import costs by 25%
   - **Brexit**: Remove UK-EU trade benefits
   - **Random shock**: Add noise to see resilience

### 3.5 Interpretation Guide

**What the field tells you**:

1. **Hot spots (red regions)**:
   - Trade blocs (EU cluster, NAFTA cluster)
   - Countries that over-trade with each other
   - Example: EU region shows persistent red (integrated market)

2. **Cold spots (blue regions)**:
   - Isolated regions
   - Countries that under-trade
   - Example: Africa-Asia often shows blue (limited trade despite potential)

3. **Gradients (changing colors)**:
   - Boundaries between trade blocs
   - Trade "frontiers" where integration is incomplete
   - Example: EU-Russia border shows gradient (trade drops off sharply)

4. **Peaks and valleys (Laplacian mode)**:
   - **Negative Laplacian (peak)**: Trade hub
   - **Positive Laplacian (valley)**: Trade sink
   - Identifies critical nodes in network

### 3.6 Real-World Examples

**2008 Financial Crisis**:
- Field variance increased (more red and blue patches)
- Gradients steepened (blocs became more isolated)
- Energy concentration in financial centers (US, EU)

**COVID-19 (2020)**:
- Sudden shift in field patterns
- Increase in blue regions (under-trading everywhere)
- Red patches concentrate in essential goods trade (medical supplies, food)

**Brexit (2016-2020)**:
- UK region shifts from red (EU cluster) to blue (outside cluster)
- Gradient develops around UK border
- Field variance increases (more heterogeneity)

---

## 4. Research Lab

**URL**: `advanced_topology.html`

### 4.1 The Four Methods

This page shows **four different ways** to measure how trade patterns change over time.

#### Method 1: Residual Dispersion

**What it measures**: How spread out the gravity residuals are

**Math**: Standard deviation of log trade gaps
```
σ_residual = sqrt( (1/N) Σ_ij (r_ij - r̄)² )
```

**Intuition**:
- **High dispersion**: Trade is unpredictable, gravity model doesn't fit well
- **Low dispersion**: Trade follows gravity model closely
- **Increasing trend**: Trade becoming less predictable (crisis, fragmentation)
- **Decreasing trend**: Trade becoming more rule-based (integration, agreements)

**What to look for**:
- Spikes in crisis years (2008, 2020)
- Long-term trend (is trade becoming more or less predictable?)

**Interpretation**:
```
σ = 0.3: Model explains ~90% of variation (good fit)
σ = 0.5: Model explains ~75% of variation (moderate fit)
σ = 0.7: Model explains ~50% of variation (poor fit)
```

#### Method 2: Transport Shift (Wasserstein Distance)

**What it measures**: How much the distribution of residuals changed from one year to the next

**Math**: Wasserstein-1 distance (Earth Mover's Distance)
```
W₁(r^t, r^{t+1}) = (1/N) Σ_i |r_{(i)}^t - r_{(i)}^{t+1}|
```
where `r_{(i)}` are sorted residuals.

**Intuition**:
Imagine residuals as piles of dirt. Wasserstein distance = minimum cost to reshape pile from year t to year t+1.

- **Small W₁ (~0.05)**: Distribution barely changed (stable year)
- **Large W₁ (~0.20)**: Distribution shifted a lot (structural break, crisis)

**What to look for**:
- Sudden spikes = major events (trade wars, pandemics, currency crises)
- Gradual increase = slow divergence from gravity (de-globalization?)
- Gradual decrease = convergence to gravity (integration, standardization)

**Real data example** (from our platform):
```
2016: W₁ = 0.23 → Brexit referendum
2020: W₁ = 0.21 → COVID-19
2010: W₁ = 0.18 → Greek debt crisis
2018: W₁ = 0.12 → Normal year
```

**Interpretation**:
Think of it as "distance the trade pattern moved" between years. Large movements = big shocks.

#### Method 3: Network Concentration (HHI)

**What it measures**: How concentrated trade is (do a few countries dominate, or is trade spread out?)

**Math**: Herfindahl-Hirschman Index
```
HHI_export = Σ_i (share_i)²
```
where `share_i = exports_i / total_exports`

**Intuition**:
- **HHI = 1.0**: One country does all the trade (monopoly)
- **HHI = 1/N**: Trade perfectly distributed across N countries
- **HHI = 0.10**: Moderate concentration (~10 major players)

**What to look for**:
- Increasing HHI: Trade concentrating (China rise, hub-and-spoke networks)
- Decreasing HHI: Trade diversifying (new exporters emerging)
- Difference between export and import HHI (asymmetric network)

**Interpretation**:
```
HHI < 0.05: Very diversified (robust to shocks)
HHI = 0.08: Moderate concentration (typical for global trade)
HHI > 0.15: High concentration (vulnerable to hub failures)
```

**Real example**:
- China's export share: ~15% of world trade
- Contribution to HHI: 0.15² = 0.0225
- If 10 countries like China existed: HHI ≈ 0.225 (very concentrated)
- Actual HHI ≈ 0.08 (China + 50 smaller exporters)

#### Method 4: Topology Field Variance

**What it measures**: How "bumpy" the topology field is

**Math**: Variance of the gridded field
```
Var(field) = (1/M²) Σ_{k,ℓ} (r̂[k,ℓ] - r̄)²
```

**Intuition**:
- **Low variance**: Field is smooth (homogeneous trade patterns)
- **High variance**: Field is bumpy (lots of hot/cold spots, fragmentation)

**What to look for**:
- Increasing variance: Network fragmenting into blocs
- Decreasing variance: Network homogenizing (globalization)
- Spikes: Sudden re-organization (policy changes, shocks)

**Physical analogy**:
Think of the field as a landscape:
- Low variance = flat plain (uniform trade everywhere)
- High variance = mountainous terrain (peaks of over-trading, valleys of under-trading)

**Connection to other methods**:
- Field variance ≈ spatial version of residual dispersion
- Captures geographic clustering that residual dispersion misses

### 4.2 How to Use the Research Lab

**Step 1**: Look at all four charts
- Do they move together? (Synchronized shocks)
- Do they tell different stories? (Nuanced dynamics)

**Step 2**: Identify crisis years
- Look for spikes in Wasserstein distance (biggest signal)
- Check if other metrics also spike (confirms crisis)
- Example: 2020 should spike in ALL four metrics

**Step 3**: Identify trends
- Is residual dispersion increasing or decreasing?
- Is HHI increasing (concentration) or decreasing (diversification)?
- Does field variance show fragmentation or integration?

**Step 4**: Compare metrics
- If Wasserstein spikes but HHI stable → distribution shift without concentration change
- If field variance increases but residual dispersion stable → spatial re-organization
- If HHI increases but field variance decreases → centralization without fragmentation

### 4.3 Interpretation Examples

**Scenario 1: Globalization (1995-2007)**
- Residual dispersion: Decreasing (trade more predictable)
- Wasserstein: Low, stable (gradual evolution)
- HHI: Slightly increasing (China rising)
- Field variance: Decreasing (homogenization)

**Scenario 2: Financial Crisis (2008)**
- Residual dispersion: Spikes up (unpredictable collapse)
- Wasserstein: Large spike (sudden shift)
- HHI: Temporarily decreases (proportional collapse)
- Field variance: Spikes up (fragmentation)

**Scenario 3: COVID-19 (2020)**
- Residual dispersion: Largest spike ever
- Wasserstein: Second-largest shift (after Brexit)
- HHI: Increases (essential goods concentration)
- Field variance: Largest spike (extreme fragmentation)

**Scenario 4: Trade War (2018-2019)**
- Residual dispersion: Moderate increase
- Wasserstein: Elevated but not spike
- HHI: Increases (China-US decoupling)
- Field variance: Increases (bloc formation)

---

## 5. Trade Map

**URL**: `trade-map.html`

### 5.1 What You're Looking At

A **geographic visualization** of actual trade flows on a world map.

**Elements**:
- **Blue/gray world map**: Base layer (countries)
- **Curved lines**: Trade flows (great circle arcs between countries)
- **Colored dots**: Country nodes (sized by total trade)

### 5.2 How to Interpret

**Line properties**:
- **Thickness**: Trade volume (thicker = more trade)
- **Color**:
  - For actual/predicted trade: Teal gradient (darker = more trade)
  - For residuals: Red = over-trading, Orange = under-trading
- **Curve**: Great circle path (shortest route on sphere)

**Node properties**:
- **Size**: Total trade (sum of imports + exports shown on map)
- **Color**: Gold/yellow (hub countries)
- **Position**: Geographic centroid of country

### 5.3 What to Look For

1. **Dense regions**: Trade blocs
   - EU: Dense network of thick lines
   - Asia: Star pattern (China hub)
   - Americas: NAFTA triangle

2. **Long-distance flows**: Globalization
   - China → US: Thick line across Pacific
   - EU → Asia: Long arcs
   - Volume has grown over time (compare 2005 vs 2021)

3. **Missing flows**: Isolation
   - Africa internally: Few lines (low intra-African trade)
   - South America: Some countries disconnected

4. **Hubs vs periphery**:
   - Hubs: Large dots, many lines (US, China, Germany)
   - Periphery: Small dots, few lines (smaller economies)

### 5.4 Controls

- **Year**: See evolution over time
- **Metric**: Switch between actual, predicted, residual
- **Top N flows**: Filter to show only largest flows (clarity vs completeness)
- **Min trade threshold**: Remove small flows (reduce clutter)

### 5.5 Use Cases

**Compare trade blocs**:
- Set metric to "residual"
- Look for red clusters (over-trading blocs): EU, NAFTA, ASEAN
- Look for blue gaps (under-trading): Africa-Asia, South America-Africa

**Track COVID impact**:
- Year: 2019 (baseline)
- Year: 2020 (shock)
- Look for: Thinner lines (less trade), fewer lines (disruptions)

**Identify critical routes**:
- Top N flows: 25
- See the 25 most important trade relationships
- These are the "arteries" of global trade

---

## 6. Residual Surface

**URL**: `residual-surface.html`

### 6.1 What You're Looking At

A **3D surface plot** showing the gravity model's prediction surface with actual data points floating above/below it.

**Axes**:
- **X**: Log distance
- **Y**: Log(GDP_i × GDP_j)
- **Z**: Log trade value
- **Surface**: Model prediction
- **Points**: Actual data

### 6.2 How to Interpret

**The surface**:
- Smooth 3D mesh representing `X̂_ij = f(distance, GDP)`
- Tilts down (left to right): Distance effect
- Slopes up (front to back): GDP effect
- Curvature: Interaction between distance and GDP

**Points relative to surface**:
- **Above surface (red)**: Over-trading (positive residual)
- **On surface (white)**: Exact prediction (zero residual)
- **Below surface (blue)**: Under-trading (negative residual)

### 6.3 What Patterns Mean

**Systematic deviations**:

1. **Vertical band of red points at low distance**:
   - Close neighbors over-trade
   - Captures border effect, regional integration
   - Example: EU countries, NAFTA members

2. **Scattered red points at specific GDP levels**:
   - Certain economic pairings work well
   - Example: Middle-income to high-income (supply chains)

3. **Blue cluster at high distance, low GDP**:
   - Small, distant countries under-trade
   - High costs, low benefits
   - Example: African-Latin American trade

**Random scatter**:
- Points randomly above/below surface
- Unpredictable factors (politics, culture, history)
- This is what gravity model **can't** explain

### 6.4 Use Cases

**Model diagnostics**:
- Is scatter symmetric? (Should be if model is unbiased)
- Are there systematic patterns? (Missed variables)
- Is variance constant? (Check for heteroskedasticity)

**Find outliers**:
- Points far from surface
- Investigate: Why does this dyad deviate so much?
- Examples: Cuba-US (embargo), Qatar-neighbors (blockade)

**Compare models**:
- Switch between different gravity specifications
- Does a more complex model reduce scatter?
- Trade-off: Simplicity vs. fit

---

## 7. How Everything Connects

### 7.1 The Analysis Pipeline

```
BACI trade data (bilateral flows)
         ↓
    Gravity model estimation (PPML)
         ↓
    Predictions (X̂_ij) vs Actual (X_ij)
         ↓
    Residuals (r_ij = ln(X_ij) - ln(X̂_ij))
         ↓
    ┌──────────┬──────────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓          ↓
3D Cloud   Topology   Research  Trade Map   Surface
(Gravity)  (Fields)   (Metrics)  (Geo)      (Model)
```

### 7.2 Which Visualization for Which Question?

**Question**: "How does distance affect trade?"
→ **Gravity Explorer**: See the slope (distance elasticity)

**Question**: "Are there trade blocs?"
→ **Topology Signals**: Look for persistent red clusters
→ **Trade Map**: Look for dense regional networks

**Question**: "Did COVID disrupt trade?"
→ **Research Lab**: Check Wasserstein spike in 2020
→ **Gravity Explorer**: Compare 2019 vs 2020 scatter

**Question**: "Which countries are trade hubs?"
→ **Trade Map**: Look for large nodes with many lines
→ **Research Lab**: Check network concentration (HHI)

**Question**: "Is my gravity model good?"
→ **Residual Surface**: Check scatter around surface
→ **Research Lab**: Check residual dispersion

**Question**: "How has trade evolved over time?"
→ **All visualizations**: Use year slider to track changes

### 7.3 Connecting the Math

All methods use the same foundation: **gravity residuals**

```
r_ij = ln(X_ij) - ln(X̂_ij) = ε_ij
```

Then we analyze these residuals in different ways:

1. **Gravity Explorer**: Plot raw values in 3D
2. **Topology Fields**: Interpolate to continuous field, study shape
3. **Research Lab Method 1**: Compute dispersion (standard deviation)
4. **Research Lab Method 2**: Measure distribution shifts (Wasserstein)
5. **Research Lab Method 3**: Aggregate to network level (HHI)
6. **Research Lab Method 4**: Spatial variance of field
7. **Trade Map**: Visualize geographically with flows
8. **Residual Surface**: Show deviations from prediction

### 7.4 The Topology Philosophy

**Traditional approach**:
- Residuals are "noise" to minimize
- Focus on reducing σ² (R-squared maximization)
- Ignore structure in residuals

**Topological approach**:
- Residuals contain **information** about what gravity misses
- Structure in residuals reveals: blocs, crises, fragmentation
- Analyze residuals **as data**, not noise

**Key insight**:
The "error" from your model is not random—it has patterns. Those patterns tell you about aspects of trade that gravity doesn't capture (culture, politics, networks, institutions).

---

## Quick Reference: Glossary of Terms

**Gravity Model**: Regression predicting trade from distance, GDP, and characteristics

**Residual**: Difference between actual and predicted trade (r_ij = ln(X_ij) - ln(X̂_ij))

**Topology Field**: Continuous 2D surface created by interpolating residuals

**MDS (Multidimensional Scaling)**: Method to map countries to 2D space preserving distances

**Wasserstein Distance**: "Earth mover's distance" between two distributions

**HHI (Herfindahl-Hirschman Index)**: Measure of concentration (Σ share²)

**Betti Number**: Counts topological features (connected components, holes, voids)

**Laplacian**: Second derivative (curvature) of field

**Gradient**: Direction of steepest increase in field

**Multilateral Resistance**: General equilibrium effect (how easy to trade with everyone)

**PPML**: Poisson Pseudo-Maximum Likelihood (estimation method)

**Fixed Effects**: Dummy variables for each country/year (absorbs unobserved heterogeneity)

**Elasticity**: Percent change in trade per percent change in explanatory variable

---

## Need More Help?

**For mathematical details**: See [GRAVITY_THEORY.md](GRAVITY_THEORY.md), [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md), [ESTIMATION_METHODS.md](ESTIMATION_METHODS.md)

**For implementation**: See [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md)

**For research applications**: See [TOPOLOGY_THEORY.md Section 10](TOPOLOGY_THEORY.md#10-empirical-applications)

**For learning path**: See [TEACHING_INDEX.md](TEACHING_INDEX.md)

---

**Happy exploring!** 🌍📊🔬
