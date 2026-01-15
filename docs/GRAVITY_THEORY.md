# Gravity Models in International Trade: Complete Theoretical Foundation
**A Comprehensive Pedagogical Guide**

---

## Table of Contents

1. [Historical Development](#1-historical-development)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Structural Gravity Models](#3-structural-gravity-models)
4. [Estimation Methods](#4-estimation-methods)
5. [Multilateral Resistance](#5-multilateral-resistance)
6. [Extensions and Variations](#6-extensions-and-variations)
7. [Empirical Applications](#7-empirical-applications)

---

## 1. Historical Development

### 1.1 Origins: The Physics Analogy

The gravity equation in economics derives from Newton's law of universal gravitation:

**Newton's Law** (1687):
```
F_ij = G × (m_i × m_j) / d_ij²
```

**Tinbergen's Gravity Equation** (1962):
```
X_ij = A × (Y_i × Y_j) / D_ij
```

Where:
- `X_ij`: Bilateral trade from country i to j
- `Y_i, Y_j`: Economic sizes (GDP)
- `D_ij`: Distance between countries
- `A`: Constant of proportionality

**Key Insight**: Trade flows are proportional to economic size and inversely related to distance, analogous to gravitational attraction.

### 1.2 Early Empirical Success

**Why gravity models worked empirically** (1960s-1990s):
- R² typically 0.6-0.9 in cross-sectional regressions
- Distance elasticity remarkably stable: β ≈ -0.9 to -1.1
- Robust across countries, time periods, and products

**The "mystery"** (Leamer & Levinsohn, 1995):
> "The gravity equation has been one of the most successful empirical tools in economics, yet for many years it lacked strong theoretical foundations."

### 1.3 Theoretical Microfoundations

**Three independent derivations** emerged in the 1970s-1980s:

1. **Anderson (1979)**: CES preferences + product differentiation
2. **Bergstrand (1985, 1989)**: General equilibrium model
3. **Deardorff (1998)**: Showed gravity emerges from multiple trade theories

**Key breakthrough**: Anderson & van Wincoop (2003)
- Introduced **multilateral resistance** terms
- Showed naive gravity regressions suffer from omitted variable bias
- Provided structural estimation framework

---

## 2. Theoretical Foundations

### 2.1 The Armington Assumption

**Assumption**: Goods are differentiated by country of origin.

**Utility Function** (CES):
```
U_j = [Σ_i β_i^(1/σ) c_ij^((σ-1)/σ)]^(σ/(σ-1))
```

Where:
- `c_ij`: Consumption in j of goods from i
- `β_i`: Preference parameter for goods from i
- `σ`: Elasticity of substitution (σ > 1)

**Properties**:
- Consumers love variety (σ > 1)
- Constant elasticity of substitution between any two origins
- Separable across origins (weak separability)

### 2.2 Budget Constraint and Demand

**Consumer's Problem**:
```
max U_j  s.t.  Σ_i p_ij c_ij ≤ Y_j
```

**Solution** (Roy's identity):
```
c_ij = β_i (p_ij / P_j)^(-σ) Y_j / P_j
```

**Price Index**:
```
P_j = [Σ_i β_i p_ij^(1-σ)]^(1/(1-σ))
```

**Interpretation**:
- `P_j`: Cost of living in country j
- Depends on all source prices (multilateral)
- Higher when trade costs are high

### 2.3 Trade Costs and Iceberg Form

**Iceberg Trade Costs** (Samuelson, 1954):
- To deliver 1 unit to j from i, must ship `t_ij ≥ 1` units
- `t_ij - 1`: Fraction "melted" in transit
- Examples: transportation, tariffs, regulatory barriers

**Consumer Price**:
```
p_ij = p_i × t_ij
```

Where `p_i` is factory-gate price in i.

**Standard Specification**:
```
t_ij = dist_ij^ρ × exp(β₁contig_ij + β₂lang_ij + β₃rta_ij + ε_ij)
```

**Trade Cost Components**:
1. **Distance** (`dist_ij^ρ`): Physical shipping costs
2. **Contiguity** (`contig_ij`): Shared border reduces costs
3. **Language** (`lang_ij`): Common language facilitates trade
4. **Trade agreements** (`rta_ij`): Preferential access
5. **Unobserved** (`ε_ij`): Cultural affinity, networks

---

## 3. Structural Gravity Models

### 3.1 Anderson-van Wincoop (2003) Model

**Market Clearing**:
```
Y_i = Σ_j X_ij   (supply equals exports)
Y_j = Σ_i X_ij   (expenditure equals imports)
```

**Bilateral Trade Flow**:
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ)
```

**Multilateral Resistance Terms**:

**Outward Multilateral Resistance** (OMR):
```
Π_i^(1-σ) = Σ_j (t_ij / P_j)^(1-σ) (Y_j / Y^W)
```

**Inward Multilateral Resistance** (IMR):
```
P_j^(1-σ) = Σ_i (t_ij / Π_i)^(1-σ) (Y_i / Y^W)
```

**Interpretation**:
- `Π_i`: "Outward remoteness" - how far i is from all markets
- `P_j`: "Inward remoteness" - how far j is from all sources
- **Key insight**: Bilateral trade depends on multilateral trade costs

**System of Equations**:
- N equations for `Π_i` (i = 1, ..., N)
- N equations for `P_j` (j = 1, ..., N)
- Non-linear system, solved iteratively

### 3.2 Head-Mayer (2014) Approach

**Simplification**: Use observable variables as proxies for MR terms.

**Proxy for Outward MR**:
```
Π_i ≈ [Σ_j dist_ij^(-θ) Y_j]^(-1/θ)
```

**Proxy for Inward MR**:
```
P_j ≈ [Σ_i dist_ij^(-θ) Y_i]^(-1/θ)
```

**Trade Flow**:
```
X_ij = (Y_i Y_j / Y^W) × dist_ij^(-θ) × exp(β'Z_ij) / (Π_i P_j)
```

**Advantages**:
- No iterative solving required
- Computationally simple
- Good approximation when distance dominates trade costs

**Limitations**:
- Ignores unobserved bilateral heterogeneity
- Distance proxy may not capture all MR variation

### 3.3 Structural Interpretation

**Comparative Statics**: How does trade respond to changes?

**Bilateral Trade Elasticity w.r.t. Trade Costs**:
```
∂ln(X_ij) / ∂ln(t_ij) = -(σ - 1) [1 + ∂ln(Π_i)/∂ln(t_ij) + ∂ln(P_j)/∂ln(t_ij)]
```

**Three Effects**:
1. **Direct effect**: ↑ t_ij → ↓ X_ij (negative, size: -(σ-1))
2. **OMR effect**: ↑ t_ij → ↑ Π_i → ↑ X_ij (positive, dampening)
3. **IMR effect**: ↑ t_ij → ↑ P_j → ↑ X_ij (positive, dampening)

**General Equilibrium Impact**:
- Naive estimate: -6 (σ = 7)
- True GE estimate: -4 (accounting for MR feedback)
- **Bias**: Omitting MR overstates trade cost effects by ~50%

---

## 4. Estimation Methods

### 4.1 Naive OLS (Wrong Approach)

**Log-linearized Equation**:
```
ln(X_ij) = α₀ + α₁ln(Y_i) + α₂ln(Y_j) + β₁ln(dist_ij) + β₂contig_ij + ε_ij
```

**Problems**:

1. **Zero Trade Flows**: ln(0) undefined
   - Dropping zeros biases sample (selection bias)
   - Missing ≠ random (systematic differences)

2. **Heteroskedasticity**: Var(ε_ij|X) ∝ X_ij
   - OLS inefficient
   - Standard errors incorrect

3. **Jensen's Inequality**: E[ln(X)] ≠ ln(E[X])
   - Log transformation creates bias
   - Estimates inconsistent even with homoskedasticity

**Empirical Evidence** (Santos Silva & Tenreyro, 2006):
- OLS elasticity: -1.2 (distance)
- PPML elasticity: -0.8 (distance)
- **Bias**: ~50% overstatement

### 4.2 Poisson Pseudo-Maximum Likelihood (PPML)

**Estimating Equation** (Santos Silva & Tenreyro, 2006):
```
X_ij = exp(α_i + δ_j + β'z_ij + ε_ij)
```

**Fixed Effects**:
- `α_i`: Exporter fixed effect = captures Y_i and Π_i
- `δ_j`: Importer fixed effect = captures Y_j and P_j
- `β`: Coefficients on bilateral variables (distance, etc.)

**Estimation**:
- Maximize Poisson likelihood (even though X_ij not count data)
- **Pseudo-ML**: Consistent if E[X_ij|z_ij, α_i, δ_j] correctly specified
- Robust to distributional misspecification

**First-Order Conditions**:
```
Σ_ij (X_ij - X̂_ij) z_ij = 0
Σ_j (X_ij - X̂_ij) = 0  for each i
Σ_i (X_ij - X̂_ij) = 0  for each j
```

**Properties**:

✅ **Handles Zeros**: No need to drop zero trade flows
✅ **Consistent under Heteroskedasticity**: Robust standard errors
✅ **No Log Transformation Bias**: Works in levels
✅ **Preserves Adding-Up**: Σ_j X̂_ij = Σ_j X_ij (market clearing)

**Implementation** (Python):
```python
import statsmodels.api as sm

# PPML specification
model = sm.GLM(
    trade,
    design_matrix,
    family=sm.families.Poisson(),
    drop='none'  # Don't drop zeros
)

result = model.fit(
    cov_type='HC1'  # Robust standard errors
)
```

### 4.3 Fixed Effects Structure

**Three-Way Fixed Effects**:
```
X_ij = exp(α_i + δ_j + γ_t + β'z_ijt)
```

**Interpretation**:

1. **Exporter FE** (`α_i`):
   - Captures: Y_i, production costs, outward MR (Π_i)
   - Absorbs all time-invariant exporter characteristics
   - Example: China's α_i captures its manufacturing capacity

2. **Importer FE** (`δ_j`):
   - Captures: Y_j, preferences, inward MR (P_j)
   - Absorbs all time-invariant importer characteristics
   - Example: US's δ_j captures its market size and openness

3. **Time FE** (`γ_t`):
   - Captures global trends (e.g., containerization, WTO)
   - Controls for aggregate shocks
   - Example: 2008 financial crisis

**Identification**:
- Need normalization: Set α₁ = 0 or constrain Σα_i = 0
- Bilateral variables (distance, etc.) identified from variation across dyads
- Fixed effects absorb N+N+T-2 degrees of freedom

### 4.4 Tetrads (Head, Mayer & Ries, 2010)

**Motivation**: Avoid estimating (and normalizing) fixed effects.

**Trade Ratio**:
```
(X_ij X_lk) / (X_ik X_lj) = (t_ij t_lk / (t_ik t_lj))^(1-σ)
```

**Properties**:
- **Ratio eliminates** all country-specific terms (Y, Π, P)
- Only bilateral trade costs remain
- Can estimate σ without FE

**Log-Linear Form**:
```
ln(X_ij X_lk / X_ik X_lj) = (1-σ)[ln(t_ij) + ln(t_lk) - ln(t_ik) - ln(t_lj)]
```

**Application**:
- Useful for estimating elasticity of substitution σ
- Avoids incidental parameters problem (large N, small T)
- Trade-off: Loses information (uses ratios instead of levels)

---

## 5. Multilateral Resistance

### 5.1 Economic Intuition

**The "Remoteness" Concept**:

Consider two scenarios:
1. **Brazil-Argentina trade**: Close neighbors, but far from rest of world
2. **Belgium-Netherlands trade**: Close neighbors, near large EU market

**Question**: Who trades more bilaterally?

**Naive answer**: Both pairs equally (same distance)

**Correct answer**: Brazil-Argentina trade MORE
- They're remote from alternatives → high Π, P
- Belgium-Netherlands have low-cost EU alternatives → low Π, P
- Remoteness ↑ bilateral trade (diverts trade inward)

### 5.2 Mathematical Derivation

**Starting Point**: Bilateral trade equation
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ)
```

**Market Clearing for Country i**:
```
Y_i = Σ_j X_ij = Σ_j (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ)
```

**Solve for Π_i**:
```
1 = Σ_j (Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ)
```

**Rearrange**:
```
Π_i^(1-σ) = Σ_j (Y_j / Y^W) × (t_ij / P_j)^(1-σ)
```

**Similarly for P_j**:
```
P_j^(1-σ) = Σ_i (Y_i / Y^W) × (t_ij / Π_i)^(1-σ)
```

**System Properties**:
- **Non-linear**: Π_i depends on P_j and vice versa
- **Implicit**: No closed-form solution
- **Unique**: Under mild regularity conditions (connectivity)

### 5.3 Solving the MR System

**Iterative Algorithm** (Anderson & van Wincoop, 2003):

1. **Initialize**: Π_i⁽⁰⁾ = P_j⁽⁰⁾ = 1 for all i,j
2. **Update OMR**:
   ```
   Π_i^(1-σ),⁽ᵏ⁺¹⁾ = Σ_j (Y_j / Y^W) × (t_ij / P_j⁽ᵏ⁾)^(1-σ)
   ```
3. **Update IMR**:
   ```
   P_j^(1-σ),⁽ᵏ⁺¹⁾ = Σ_i (Y_i / Y^W) × (t_ij / Π_i⁽ᵏ⁺¹⁾)^(1-σ)
   ```
4. **Check Convergence**: |Π⁽ᵏ⁺¹⁾ - Π⁽ᵏ⁾| < ε?
5. **Repeat** until convergence

**Convergence Properties**:
- Guaranteed under trade cost symmetry and connectivity
- Typically converges in 10-50 iterations
- Damping factor (0.5-0.8) speeds convergence

**Python Implementation**:
```python
def solve_multilateral_resistance(t_ij, Y, sigma=5, max_iter=100, tol=1e-6):
    """
    Solve Anderson-van Wincoop multilateral resistance system.

    Parameters:
    -----------
    t_ij : ndarray (N x N)
        Bilateral trade cost matrix
    Y : ndarray (N,)
        Country GDPs
    sigma : float
        Elasticity of substitution
    """
    N = len(Y)
    Y_world = Y.sum()

    # Initialize
    Pi = np.ones(N)
    P = np.ones(N)

    for iteration in range(max_iter):
        Pi_old = Pi.copy()

        # Update OMR
        for i in range(N):
            Pi[i] = (
                sum(
                    (Y[j] / Y_world) * (t_ij[i,j] / P[j])**(1-sigma)
                    for j in range(N)
                )
            )**(1/(1-sigma))

        # Update IMR
        for j in range(N):
            P[j] = (
                sum(
                    (Y[i] / Y_world) * (t_ij[i,j] / Pi[i])**(1-sigma)
                    for i in range(N)
                )
            )**(1/(1-sigma))

        # Check convergence
        if np.max(np.abs(Pi - Pi_old)) < tol:
            print(f"Converged in {iteration+1} iterations")
            break

    return Pi, P
```

### 5.4 Empirical Magnitudes

**Typical MR Values** (Anderson & van Wincoop, 2003):

| Country | Π (OMR) | P (IMR) | Interpretation |
|---------|---------|---------|----------------|
| Canada | 0.95 | 0.98 | Close to US (large market) |
| Mexico | 0.92 | 0.96 | Close to US |
| Brazil | 1.15 | 1.12 | Remote from major markets |
| Australia | 1.18 | 1.15 | Very remote ("down under") |
| Belgium | 0.88 | 0.90 | Central in EU |
| New Zealand | 1.22 | 1.18 | Most remote |

**Interpretation**:
- MR > 1: Remote (e.g., New Zealand: +22% trade costs)
- MR < 1: Central (e.g., Belgium: -12% trade costs)
- **Border effect**: Canada-US trade would be 44% higher without border (after accounting for MR)

---

## 6. Extensions and Variations

### 6.1 Firm Heterogeneity (Melitz, 2003)

**Setup**: Firms differ in productivity φ, CES preferences, Pareto productivity distribution.

**Gravity Equation** (Helpman, Melitz & Rubinstein, 2008):
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-σ) × [1 - (φ*_ij / φ̄_i)^k]
```

**New Terms**:
- `φ*_ij`: Productivity cutoff for exporting to j
- `k`: Pareto shape parameter
- `[...]`: Extensive margin (fraction of firms exporting)

**Interpretation**:
- Lower trade costs → More firms export → Larger extensive margin
- Gravity still holds, but with selection correction
- **Estimation**: Heckman selection model or PPML with firm-level data

### 6.2 Intermediate Goods (Eaton & Kortum, 2002)

**Production Function**:
```
Y_i = A_i L_i^α Π_j X_ij^(β_j)
```

Where `X_ij` includes intermediate imports from j.

**Gravity with Intermediates**:
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / (Π_i P_j))^(1-θ) × [1 + γ × Share_intermediates]
```

**Properties**:
- Trade costs bite twice (inputs + final goods)
- Amplification effect: ∂ln(X_ij)/∂ln(t_ij) = -(1+γ)(θ-1)
- Empirically: γ ≈ 0.3-0.5 (30-50% amplification)

### 6.3 Trade in Services

**Challenge**: Services trade often not recorded in customs data.

**Gravity for Services** (Francois & Hoekman, 2010):
```
S_ij = (Y_i^S Y_j^S / Y^W_S) × (τ_ij / (Π_i^S P_j^S))^(1-σ_S)
```

**Key Differences**:
- `τ_ij`: Regulatory barriers (not tariffs)
- `σ_S`: Higher for services (less differentiated)
- Distance effect weaker (digital delivery)

**Empirical Finding**: Services trade costs ~2x goods trade costs.

### 6.4 Dynamic Gravity (Olivero & Yotov, 2012)

**Intertemporal Trade**:
```
X_ijt = (Y_it Y_jt / Y^W_t) × (t_ijt / (Π_it P_jt))^(1-σ) × exp(ρ ln(X_ij,t-1))
```

**New Element**:
- `ρ`: Persistence parameter (sunk costs of entering market)
- Lagged trade → State dependence

**Interpretation**:
- Firms pay fixed cost to enter market j
- Once in, they stay (hysteresis)
- **Policy implication**: Trade liberalization has long-run dynamic gains

---

## 7. Empirical Applications

### 7.1 Trade Agreement Effects

**Question**: Do RTAs increase trade?

**Specification**:
```
X_ijt = exp(α_it + δ_jt + β₁RTA_ijt + ε_ijt)
```

**Meta-Analysis Results** (Head & Mayer, 2014):
- Average RTA effect: exp(0.50) ≈ **65% increase** in trade
- Range: 30% (WTO) to 150% (EU)
- Heterogeneity by: depth of integration, sector, geography

**Challenges**:
- **Endogeneity**: Countries form RTAs when trade is growing
- **Solution**: IV (geography, political alignment) or panel FE

### 7.2 Border Effects

**"Border Puzzle"** (McCallum, 1995):
- Canada-Canada trade 22x larger than Canada-US trade (controlling for size, distance)
- Puzzle: US-Canada border shouldn't matter (NAFTA)

**Anderson-van Wincoop Solution** (2003):
- Naive estimate (OLS without MR): 22x
- Corrected estimate (PPML with MR): 4x
- **Resolution**: Most of "border effect" was multilateral resistance bias

**Current Estimates**:
- Canada-US border: 4-5x (tariff equivalent: 170%)
- EU internal borders: 2-3x (declining over time)

### 7.3 Currency Union Effects (Rose, 2000)

**Claim**: Common currency triples trade.

**Specification**:
```
X_ij = exp(α_i + δ_j + β_currency CU_ij + controls)
```

**Results**:
- Rose (2000): β = 1.2 → exp(1.2) ≈ **3.3x increase**
- Meta-studies: β ≈ 0.3-0.5 → **35-65% increase**
- **Endogeneity concerns**: Countries join CU when already highly integrated

**Euro Effect** (Baldwin et al., 2008):
- Ex-ante predictions: 50-100% increase
- Ex-post estimates: 5-15% increase
- **Puzzle**: Much smaller than cross-sectional estimates

### 7.4 Distance "Puzzle"

**Empirical Fact**: Distance coefficient has NOT declined over time.

**Stylized Facts**:
- 1960s: β_dist ≈ -1.0
- 2020s: β_dist ≈ -0.9
- **Puzzle**: Shipping costs ↓ 80%, but distance still matters!

**Explanations** (Disdier & Head, 2008):
1. **Composition**: High-value goods (more distance-sensitive) ↑
2. **Time costs**: Speed matters more (just-in-time production)
3. **Information costs**: Distance proxies for cultural/information barriers
4. **Quality**: Distant trade requires higher quality (selection effect)

---

## 8. Comparison of Major Specifications

| Model | Year | Key Innovation | Pros | Cons |
|-------|------|---------------|------|------|
| **Tinbergen** | 1962 | First gravity equation | Simple, empirical success | No theory |
| **Anderson (CES)** | 1979 | Product differentiation | Microfoundations | No multilateral resistance |
| **Anderson-van Wincoop** | 2003 | Multilateral resistance | Structural, GE consistent | Computationally intensive |
| **Head-Mayer** | 2014 | Observable MR proxies | Easy to implement | Approximation error |
| **PPML (Santos Silva)** | 2006 | Handles zeros, robust | Consistent estimator | Requires nonlinear estimation |
| **Melitz-Chaney** | 2008 | Firm heterogeneity | Extensive margin | Requires firm-level data |
| **Eaton-Kortum** | 2002 | Ricardian + intermediates | Multiple sectors | Calibration-heavy |

---

## 9. Best Practices for Estimation

### 9.1 Specification Checklist

✅ **Use PPML**, not OLS on logged trade
✅ **Include zeros** in the sample
✅ **Exporter and importer fixed effects** to capture MR
✅ **Year fixed effects** for global trends
✅ **Cluster standard errors** by country-pair
✅ **Report extensive margin** (zeros vs positives) separately
✅ **Test for heterogeneous effects** (by sector, country size)

### 9.2 Common Mistakes to Avoid

❌ **Dropping zero trade flows** → Selection bias
❌ **OLS on log(trade)** → Inconsistent due to heteroskedasticity
❌ **Omitting MR terms** → Overestimates bilateral effects
❌ **Using only cross-section** → Cannot control for unobservables
❌ **Ignoring endogeneity** of trade policy
❌ **Not testing for heterogeneity** across country pairs/sectors

---

## 10. Further Reading

### Foundational Papers
- Anderson, J. E. (1979). "A Theoretical Foundation for the Gravity Equation." *American Economic Review*, 69(1), 106-116.
- Anderson, J. E., & van Wincoop, E. (2003). "Gravity with Gravitas." *American Economic Review*, 93(1), 170-192.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). "The Log of Gravity." *Review of Economics and Statistics*, 88(4), 641-658.

### Textbooks
- Head, K., & Mayer, T. (2014). "Gravity Equations." In *Handbook of International Economics*, Vol. 4, 131-195.
- Feenstra, R. C. (2016). *Advanced International Trade: Theory and Evidence* (2nd ed.). Princeton University Press.

### Surveys
- Yotov, Y. V., et al. (2016). *An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model*. WTO-UNCTAD.
- Head, K., & Mayer, T. (2015). "Gravity in International Trade." *Oxford Research Encyclopedia of Economics and Finance*.

---

**This document is a living resource. Suggestions for improvements welcome.**

**Last Updated**: 2026-01-15
**Maintainer**: Trade Network Topology Platform
