# Econometric Estimation Methods for Gravity Models
**A Complete Practitioner's Guide**

**Last Updated**: 2026-01-15

This document provides comprehensive coverage of estimation techniques for structural gravity models, from basic OLS to advanced PPML with high-dimensional fixed effects.

---

## Table of Contents

1. [Why Estimation Method Matters](#1-why-estimation-method-matters)
2. [Ordinary Least Squares (OLS)](#2-ordinary-least-squares-ols)
3. [Poisson Pseudo-Maximum Likelihood (PPML)](#3-poisson-pseudo-maximum-likelihood-ppml)
4. [Fixed Effects Implementation](#4-fixed-effects-implementation)
5. [Two-Stage Least Squares (2SLS)](#5-two-stage-least-squares-2sls)
6. [Tetrads Method](#6-tetrads-method)
7. [Bonus Vetus OLS (BVO)](#7-bonus-vetus-ols-bvo)
8. [Structural Estimation](#8-structural-estimation)
9. [Standard Errors and Inference](#9-standard-errors-and-inference)
10. [Model Selection and Diagnostics](#10-model-selection-and-diagnostics)
11. [Computational Implementation](#11-computational-implementation)
12. [Best Practices](#12-best-practices)

---

## 1. Why Estimation Method Matters

### 1.1 The Jensen's Inequality Problem

**Naive approach**: Take logs and run OLS
```
ln(X_ij) = α_i + δ_j + β' z_ij + ε_ij
```

**Problem**: If errors are multiplicative (log-normal),
```
E[ln(X_ij)] ≠ ln(E[X_ij])
```

This is **Jensen's inequality**: the log of expectation ≠ expectation of log for non-linear functions.

**Consequence**: OLS estimates are **biased** when heteroskedasticity is present.

### 1.2 The Zero Trade Problem

**Reality**: Many country pairs have zero trade (40-60% of dyads)

**Log linearization**: `ln(0)` is undefined!

**Ad-hoc fixes** (all problematic):
1. Drop zeros → **selection bias**
2. Add 1: `ln(X_ij + 1)` → **arbitrary, changes estimates**
3. Tobit model → assumes censoring (economically dubious)

**Solution**: Estimate in levels using PPML (no logs needed)

### 1.3 Heteroskedasticity

**Empirical regularity**: Variance of trade flows increases with trade value

**Formal test** (Breusch-Pagan):
```python
from statsmodels.stats.diagnostic import het_breuschpagan

# After OLS estimation
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)

if lm_pvalue < 0.05:
    print("Heteroskedasticity detected - OLS inefficient!")
```

**Consequence**: OLS standard errors are **wrong** (too small → spurious significance)

### 1.4 Summary of Issues

| Issue | OLS | PPML |
|-------|-----|------|
| Jensen's inequality | ✗ Biased | ✓ Unbiased |
| Zero trade | ✗ Must drop | ✓ Handles naturally |
| Heteroskedasticity | ✗ Inefficient | ✓ Efficient |
| Consistency | ✗ Under misspecification | ✓ Robust |

**Bottom line**: Use PPML for gravity estimation (Santos Silva & Tenreyro 2006)

---

## 2. Ordinary Least Squares (OLS)

### 2.1 Basic Specification

```
ln(X_ij) = β₀ + β₁ ln(dist_ij) + β₂ contig_ij + β₃ comlang_ij + β₄ RTA_ij + ε_ij
```

**Assumptions**:
1. `E[ε_ij | z_ij] = 0` (exogeneity)
2. `Var(ε_ij) = σ²` (homoskedasticity)
3. `Cov(ε_ij, ε_kℓ) = 0` for `(i,j) ≠ (k,ℓ)` (no correlation)

### 2.2 OLS Estimator

**Closed form**:
```
β̂_OLS = (X'X)⁻¹ X'y
```

where:
- `y = [ln(X_ij)]` (stacked trade flows)
- `X = [z_ij]` (covariates)

**Properties** (under assumptions):
- Unbiased: `E[β̂] = β`
- Efficient: Lowest variance among linear unbiased estimators (BLUE)
- Consistent: `β̂ → β` as `n → ∞`

### 2.3 Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_parquet("data/trade_data.parquet")

# Drop zeros (problematic!)
df_nonzero = df[df['trade_value'] > 0].copy()

# Log transformation
df_nonzero['ln_trade'] = np.log(df_nonzero['trade_value'])
df_nonzero['ln_dist'] = np.log(df_nonzero['dist'])

# OLS estimation
formula = 'ln_trade ~ ln_dist + contig + comlang_off + rta'
model = sm.OLS.from_formula(formula, data=df_nonzero)
result = model.fit()

print(result.summary())
```

### 2.4 Interpretation

**Distance elasticity**:
```
β̂₁ = -0.9
→ 10% increase in distance ⇒ 9% decrease in trade
```

**RTA effect**:
```
β̂₄ = 0.15
→ exp(0.15) - 1 = 16.2% increase in trade from RTA
```

### 2.5 Why OLS Fails for Gravity

**Problem 1**: Selection bias from dropping zeros

**Example**: Suppose small countries trade less and have more zeros
- Dropping zeros → overweight large countries
- Estimates reflect large-country trade patterns, not general effects

**Problem 2**: Heteroskedasticity biases elasticities

**Proof** (simplified):

If `Var(ε_ij) = σ² × E[X_ij]` (variance ∝ trade level), then:
```
plim β̂_OLS ≠ β  (inconsistent!)
```

Santos Silva & Tenreyro (2006) show OLS can **overestimate** distance elasticity by 20-40%.

---

## 3. Poisson Pseudo-Maximum Likelihood (PPML)

### 3.1 The Poisson Model

**Setup**: Assume `X_ij ~ Poisson(λ_ij)` where
```
λ_ij = exp(α_i + δ_j + β' z_ij)
```

**Likelihood**:
```
L = Π_{ij} [λ_ij^{X_ij} exp(-λ_ij)] / X_ij!
```

**Log-likelihood**:
```
ℓ = Σ_{ij} [X_ij ln(λ_ij) - λ_ij - ln(X_ij!)]
```

Dropping constant terms:
```
ℓ = Σ_{ij} [X_ij (α_i + δ_j + β' z_ij) - exp(α_i + δ_j + β' z_ij)]
```

### 3.2 First-Order Conditions

**Derivatives**:
```
∂ℓ/∂β = Σ_{ij} [X_ij - exp(α_i + δ_j + β' z_ij)] z_ij = 0
```

**Interpretation**: Equate actual trade to predicted trade, weighted by covariates

**Key insight**: PPML estimates satisfy
```
Σ_{ij} X_ij = Σ_{ij} X̂_ij  (trade levels match)
```

This holds **even if X_ij is not Poisson**! (hence "pseudo")

### 3.3 Consistency Under Misspecification

**Theorem** (Gourieroux, Monfort, Trognon 1984):

PPML is consistent for `β` if:
```
E[X_ij | z_ij] = exp(α_i + δ_j + β' z_ij)
```

**Does NOT require**:
- Poisson distribution (can be any distribution)
- Homoskedasticity
- Normality

**Only requires**: Correct conditional mean specification

### 3.4 Handling Zeros

**Natural property**: Poisson allows `X_ij = 0` (probability > 0 for all values)

**No ad-hoc fixes needed**: Just include zeros in estimation

**Advantage**: Uses full sample → no selection bias

### 3.5 Estimation Algorithm

**Iteratively Reweighted Least Squares (IRLS)**:

1. Initialize: `β^{(0)}`, `α_i^{(0)}`, `δ_j^{(0)}`

2. Compute fitted values:
   ```
   X̂_ij^{(t)} = exp(α_i^{(t)} + δ_j^{(t)} + β^{(t)}' z_ij)
   ```

3. Compute weights:
   ```
   w_ij^{(t)} = X̂_ij^{(t)}
   ```

4. Weighted least squares:
   ```
   β^{(t+1)} = argmin_β Σ_{ij} w_ij^{(t)} [X_ij/w_ij^{(t)} - exp(β' z_ij)]²
   ```

5. Repeat until convergence: `|β^{(t+1)} - β^{(t)}| < tol`

**Convergence**: Typically 5-20 iterations

### 3.6 Implementation

```python
import statsmodels.api as sm

# PPML estimation (no logs!)
formula = 'trade_value ~ ln_dist + contig + comlang_off + rta'
model = sm.GLM.from_formula(
    formula,
    data=df,  # Include zeros!
    family=sm.families.Poisson()
)

result = model.fit()
print(result.summary())
```

### 3.7 Interpretation

**Distance elasticity**:
```
β̂₁ = -0.85
→ 10% increase in distance ⇒ 8.5% decrease in trade
```

(Note: Smaller magnitude than OLS due to bias correction)

**RTA effect**:
```
β̂₄ = 0.15
→ exp(0.15) - 1 = 16.2% increase in trade
```

(Same calculation as OLS - exponential model)

---

## 4. Fixed Effects Implementation

### 4.1 Why Fixed Effects?

**Omitted variable bias**: Many determinants of trade are unobserved
- Culture, institutions, quality
- Historical ties
- Informal networks

**Solution**: Control for all time-invariant country-specific factors

### 4.2 Exporter-Importer Fixed Effects

**Model**:
```
X_ij = exp(α_i + δ_j + β' z_ij + ε_ij)
```

where:
- `α_i`: Exporter fixed effect (absorbs all exporter characteristics)
- `δ_j`: Importer fixed effect (absorbs all importer characteristics)

**Identification**: `β` identified from variation in `z_ij` **within** dyads

**Advantages**:
1. Controls for multilateral resistance (Anderson-van Wincoop)
2. Absorbs GDP, remoteness, etc.
3. Eliminates omitted variable bias from country characteristics

### 4.3 Theory Connection: Multilateral Resistance

**Anderson-van Wincoop structural gravity**:
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / Π_i P_j)^{1-σ}
```

Taking logs:
```
ln(X_ij) = ln(Y_i) + ln(Y_j) - ln(Y^W) - (σ-1)[ln(t_ij) + ln(Π_i) + ln(P_j)]
```

**With fixed effects**:
```
α_i ≡ ln(Y_i) - (σ-1) ln(Π_i)  (outward multilateral resistance)
δ_j ≡ ln(Y_j) - (σ-1) ln(P_j)  (inward multilateral resistance)
```

**Result**: Fixed effects automatically control for MR terms!

### 4.4 Exporter-Year and Importer-Year Fixed Effects

**Panel model**:
```
X_ij,t = exp(α_{i,t} + δ_{j,t} + β' z_ij,t + ε_ij,t)
```

**Why time-varying FE?**:
- GDP changes over time
- MR terms change with network structure
- Policy environment evolves

**Cost**: Very high-dimensional (N_exporters × T + N_importers × T parameters)

**Solution**: Specialized software (ppmlhdfe, fixest)

### 4.5 Implementation with High-Dimensional FE

**Python (ppml_panel_sg)**:
```python
# Note: This is conceptual - actual package may differ
from ppml_panel_sg import ppml_hdfe

# Create exporter-year and importer-year identifiers
df['exp_year'] = df['iso_o'] + '_' + df['year'].astype(str)
df['imp_year'] = df['iso_d'] + '_' + df['year'].astype(str)

result = ppml_hdfe(
    data=df,
    y='trade_value',
    x=['ln_dist', 'contig', 'comlang_off', 'rta'],
    fixed_effects=['exp_year', 'imp_year']
)
```

**R (fixest package)**:
```r
library(fixest)

model <- fepois(
  trade_value ~ ln_dist + contig + comlang_off + rta | exp_year + imp_year,
  data = df,
  cluster = ~pair_id
)

summary(model)
```

### 4.6 Degrees of Freedom Adjustment

**Challenge**: With K = N_exporters × T + N_importers × T fixed effects, standard errors need adjustment

**Correction**:
```
SE_corrected = SE_naive × sqrt(n / (n - K))
```

**Cluster-robust SEs**: Allow correlation within country-pairs over time
```python
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['pair_id']})
```

---

## 5. Two-Stage Least Squares (2SLS)

### 5.1 Endogeneity in Gravity

**Problem**: Some trade determinants are endogenous

**Examples**:
1. **RTAs**: Countries sign agreements because they already trade heavily
2. **Migration**: Driven by trade opportunities (reverse causality)
3. **Distance**: May proxy for unobserved cultural distance

### 5.2 Instrumental Variables

**Requirement**: Instrument `z_ij` must satisfy:
1. **Relevance**: `Corr(z_ij, X_ij^endogenous) ≠ 0`
2. **Exogeneity**: `Cov(z_ij, ε_ij) = 0`

**Common instruments**:
- Historical trade (Frankel & Rose 1998)
- Colonial ties (Head, Mayer & Ries 2010)
- Topography/terrain (Feyrer 2009)

### 5.3 2SLS Procedure

**First stage**: Regress endogenous variable on instruments
```
RTA_ij = γ₀ + γ₁ colonial_ij + γ₂ comlang_ij + γ' X + u_ij
```

Get fitted values: `RTA_ij^hat`

**Second stage**: Use fitted values in structural equation
```
ln(X_ij) = α_i + δ_j + β₁ ln(dist_ij) + β₂ RTA_ij^hat + ε_ij
```

**Implementation**:
```python
from linearmodels.iv import IV2SLS

# Specify formula
formula = 'ln_trade ~ 1 + ln_dist + [rta ~ colonial + comlang_off]'

# Estimate
model = IV2SLS.from_formula(formula, data=df)
result = model.fit()
print(result.summary)
```

### 5.4 Testing for Endogeneity

**Hausman test**:
```
H₀: OLS and IV give same estimates (no endogeneity)
```

**Procedure**:
1. Estimate OLS: `β̂_OLS`
2. Estimate IV: `β̂_IV`
3. Test: `(β̂_OLS - β̂_IV)' V⁻¹ (β̂_OLS - β̂_IV) ~ χ²(k)`

**If reject H₀**: Use IV estimator

### 5.5 Weak Instruments

**Problem**: If instruments weakly correlated with endogenous variable, IV is biased

**Test**: First-stage F-statistic
```
F > 10: Strong instruments
F < 10: Weak instruments (IV unreliable)
```

**Robust inference**: Anderson-Rubin confidence sets (valid even with weak instruments)

---

## 6. Tetrads Method

### 6.1 Motivation

**Challenge**: Fixed effects absorb too much variation, may not identify trade costs

**Tetrads solution**: Eliminate fixed effects via clever differencing

### 6.2 The Tetrad Transformation

**Structural gravity**:
```
X_ij = (Y_i Y_j / Y^W) × (t_ij / Π_i P_j)^{1-σ}
```

**For four countries i, j, k, ℓ**, take ratio:
```
(X_ij × X_kℓ) / (X_iℓ × X_kj) = (t_ij × t_kℓ) / (t_iℓ × t_kj)^{1-σ}
```

**Key**: All country-specific terms (Y, Π, P) cancel!

### 6.3 Estimation

**Log-linearize**:
```
ln(X_ij × X_kℓ) - ln(X_iℓ × X_kj) = (1-σ)[ln(t_ij) + ln(t_kℓ) - ln(t_iℓ) - ln(t_kj)]
```

**Expand trade costs**:
```
ln(t_ij) = β₁ ln(dist_ij) + β₂ RTA_ij + ...
```

**Substitute and estimate**: OLS on tetrad-transformed data

### 6.4 Advantages and Disadvantages

**Advantages**:
- No need for GDP data or MR terms
- Eliminates multilateral resistance without FE
- Identifies pure trade cost effects

**Disadvantages**:
- Requires many zero-trade dyads to be dropped (lose efficiency)
- Standard errors complex (tetrad correlations)
- Less used in practice than PPML

---

## 7. Bonus Vetus OLS (BVO)

### 7.1 The BVO Estimator

**Idea**: Combine virtues of OLS (simple) with PPML (consistent)

**Head & Mayer (2014) proposal**:

1. Estimate PPML → get `β̂_PPML`
2. Construct: `X_ij^* = X_ij + exp(β̂_PPML' z_ij)`
3. Estimate OLS on: `ln(X_ij^*) = α_i + δ_j + β' z_ij + ε_ij`

**Why it works**: Reduces heteroskedasticity by shrinking variance at low trade levels

### 7.2 Performance

**Monte Carlo results** (Head & Mayer 2014):
- BVO nearly as efficient as PPML
- More robust to misspecification than PPML
- Easier to implement (standard OLS software)

**When to use**: PPML convergence issues, very large datasets

---

## 8. Structural Estimation

### 8.1 Counterfactual Analysis

**Goal**: Predict impact of policy changes

**Example**: What if US-China tariff increased 25%?

**Naive approach** (partial equilibrium):
```
% change in X_ij = β̂₁ × % change in tariff
```

**Problem**: Ignores general equilibrium effects (MR terms change!)

### 8.2 General Equilibrium System

**Equations to solve**:

1. **Bilateral trade**:
   ```
   X_ij = (Y_i Y_j / Y^W) × (t_ij / Π_i P_j)^{1-σ}
   ```

2. **Outward MR**:
   ```
   Π_i^{1-σ} = Σ_j (t_ij / P_j)^{1-σ} (Y_j / Y^W)
   ```

3. **Inward MR**:
   ```
   P_j^{1-σ} = Σ_i (t_ij / Π_i)^{1-σ} (Y_i / Y^W)
   ```

4. **Market clearing**:
   ```
   Y_i = Σ_j X_ij  (exports = output)
   E_j = Σ_i X_ij  (imports = expenditure)
   ```

### 8.3 Solving the System

**Algorithm** (Anderson & Yotov 2016):

1. Estimate gravity → get `β̂`
2. Construct baseline trade costs: `t_ij^0 = exp(β̂' z_ij)`
3. Initialize: `Π_i^{(0)} = P_j^{(0)} = 1`
4. Iterate:
   - Update Π using inward MR formula
   - Update P using outward MR formula
   - Compute X_ij using bilateral trade formula
5. Converge when `|Π^{(t+1)} - Π^{(t)}| < tol`
6. Apply shock: `t_ij^1 = t_ij^0 × (1 + Δtariff_ij)`
7. Re-solve for new equilibrium: `{Π_i^1, P_j^1, X_ij^1}`
8. Compare: `% change = (X_ij^1 - X_ij^0) / X_ij^0`

### 8.4 Implementation

```python
def solve_gravity_equilibrium(Y, t_ij, sigma, max_iter=100, tol=1e-6):
    """
    Solve Anderson-van Wincoop structural gravity system.

    Parameters:
    -----------
    Y : array (N,) - country outputs/expenditures
    t_ij : array (N, N) - bilateral trade costs
    sigma : float - elasticity of substitution
    max_iter : int - maximum iterations
    tol : float - convergence tolerance

    Returns:
    --------
    dict with keys: Pi (outward MR), P (inward MR), X (trade matrix)
    """
    N = len(Y)
    Y_world = Y.sum()

    # Initialize
    Pi = np.ones(N)
    P = np.ones(N)

    for iteration in range(max_iter):
        Pi_old = Pi.copy()

        # Update outward MR
        for i in range(N):
            Pi[i] = (
                sum((Y[j] / Y_world) * (t_ij[i, j] / P[j])**(1 - sigma)
                    for j in range(N))
            )**(1 / (1 - sigma))

        # Update inward MR
        for j in range(N):
            P[j] = (
                sum((Y[i] / Y_world) * (t_ij[i, j] / Pi[i])**(1 - sigma)
                    for i in range(N))
            )**(1 / (1 - sigma))

        # Check convergence
        if np.max(np.abs(Pi - Pi_old)) < tol:
            break

    # Compute equilibrium trade
    X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            X[i, j] = (
                (Y[i] * Y[j] / Y_world) *
                (t_ij[i, j] / (Pi[i] * P[j]))**(1 - sigma)
            )

    return {'Pi': Pi, 'P': P, 'X': X, 'iterations': iteration + 1}
```

### 8.5 Welfare Analysis

**Consumer welfare** (real income):
```
W_j = E_j / P_j
```

**Change in welfare** from shock:
```
% ΔW_j = % ΔE_j - % ΔP_j
```

**Interpretation**:
- If imports become cheaper (P_j ↓) → welfare ↑
- If tariffs reduce expenditure (E_j ↓) → welfare ↓

---

## 9. Standard Errors and Inference

### 9.1 Heteroskedasticity-Robust Standard Errors

**OLS assumes**: `Var(ε_ij) = σ²` (constant)

**Reality**: `Var(ε_ij)` varies with trade size

**White (1980) robust SEs**:
```
V̂_robust = (X'X)⁻¹ X' Ω̂ X (X'X)⁻¹
```

where `Ω̂ = diag(ê_ij²)`

**Implementation**:
```python
result = model.fit(cov_type='HC0')  # White's SEs
```

### 9.2 Clustered Standard Errors

**Problem**: Observations within country-pairs correlated over time

**Solution**: Cluster by dyad
```
V̂_cluster = (X'X)⁻¹ [Σ_c X_c' ê_c ê_c' X_c] (X'X)⁻¹
```

where `c` indexes clusters (dyads)

**Implementation**:
```python
result = model.fit(cov_type='cluster', cov_kwds={'groups': df['pair_id']})
```

**Rule of thumb**: Always cluster by dyad in panel gravity

### 9.3 Multi-Way Clustering

**Cameron, Gelbach, Miller (2011)**: Cluster by exporter AND importer

**Formula**:
```
V̂_multi = V̂_exp + V̂_imp - V̂_exp∩imp
```

**When to use**: Worried about exporter-specific shocks (e.g., currency crisis) or importer-specific shocks (e.g., demand boom)

### 9.4 Bootstrap Standard Errors

**Procedure**:
1. Resample dyads with replacement
2. Re-estimate model
3. Repeat 1000 times
4. Standard deviation of estimates = bootstrap SE

**Advantages**: Nonparametric, robust to arbitrary correlation

**Disadvantages**: Computationally intensive

---

## 10. Model Selection and Diagnostics

### 10.1 Pseudo-R²

**For PPML**:
```
pseudo-R² = 1 - (deviance / null deviance)
```

where deviance = -2 × log-likelihood

**Interpretation**: Similar to OLS R², but not identical

**Typical values**:
- Without FE: 0.6-0.7
- With exporter-importer FE: 0.85-0.95
- With exporter-year + importer-year FE: 0.90-0.98

### 10.2 Information Criteria

**AIC** (Akaike):
```
AIC = -2 × log-likelihood + 2K
```

**BIC** (Bayesian):
```
BIC = -2 × log-likelihood + K × ln(n)
```

where K = number of parameters

**Use**: Compare nested models (lower is better)

### 10.3 Specification Tests

**Ramsey RESET test**:
```
H₀: No omitted variables
```

Add powers of fitted values, test joint significance

**Link test**:
```
X_ij = γ₀ + γ₁ X̂_ij + γ₂ X̂_ij² + η_ij
```

If `γ₂` significant → model misspecified

### 10.4 Residual Diagnostics

**Normality** (QQ-plot):
```python
import scipy.stats as stats
import matplotlib.pyplot as plt

stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-Plot")
plt.show()
```

**Heteroskedasticity** (Breusch-Pagan):
```python
from statsmodels.stats.diagnostic import het_breuschpagan

lm, p_val, f_stat, f_p_val = het_breuschpagan(residuals, X)
print(f"Heteroskedasticity test p-value: {p_val:.4f}")
```

**Autocorrelation** (Durbin-Watson):
```python
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw:.4f}")  # ~2 = no autocorrelation
```

---

## 11. Computational Implementation

### 11.1 Complete PPML Example

```python
import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
df = pd.read_parquet("data/baci_gravity_viz.parquet")

# Merge with gravity variables (distance, RTA, etc.)
gravity = pd.read_csv("data/gravity_variables.csv")
df = df.merge(gravity, on=['iso_o', 'iso_d'], how='left')

# Log distance
df['ln_dist'] = np.log(df['dist'])

# Create exporter-year and importer-year FE
df['exp_year'] = df['iso_o'] + '_' + df['year'].astype(str)
df['imp_year'] = df['iso_d'] + '_' + df['year'].astype(str)

# PPML with high-dimensional FE
# Note: Using categorical encoding for FE
df['exp_year_cat'] = pd.Categorical(df['exp_year'])
df['imp_year_cat'] = pd.Categorical(df['imp_year'])

# Build formula
formula = '''
trade_value_usd_millions ~ ln_dist + contig + comlang_off + rta +
C(exp_year_cat) + C(imp_year_cat) - 1
'''

# Estimate
model = sm.GLM.from_formula(
    formula,
    data=df,
    family=sm.families.Poisson()
)

result = model.fit(maxiter=100, method='newton')

# Display results
print(result.summary())

# Extract coefficients of interest
coefs = result.params
print("\nKey coefficients:")
print(f"Distance elasticity: {coefs['ln_dist']:.4f}")
print(f"RTA effect: {coefs['rta']:.4f} ({100*(np.exp(coefs['rta'])-1):.2f}%)")
print(f"Border effect: {coefs['contig']:.4f} ({100*(np.exp(coefs['contig'])-1):.2f}%)")

# Cluster-robust SEs (by dyad)
df['dyad_id'] = df['iso_o'] + '_' + df['iso_d']
result_cluster = result.get_robustcov_results(
    cov_type='cluster',
    groups=df['dyad_id']
)

print("\nCluster-robust standard errors:")
print(result_cluster.summary())
```

### 11.2 Performance Optimization

**For large datasets** (millions of observations):

1. **Use sparse matrices** (scipy.sparse) for FE
2. **Iterate efficiently** (avoid matrix inversions)
3. **Parallelize** (multiprocessing for bootstrap)
4. **Use compiled code** (Cython, Numba for hot loops)

**Example with Numba**:
```python
from numba import jit

@jit(nopython=True)
def compute_ppml_gradient(X, lambda_ij, z_ij):
    """Fast gradient computation."""
    n = len(X)
    grad = np.zeros(z_ij.shape[1])

    for i in range(n):
        grad += (X[i] - lambda_ij[i]) * z_ij[i, :]

    return grad
```

---

## 12. Best Practices

### 12.1 Estimation Checklist

1. ✅ **Use PPML** (not OLS) for gravity
2. ✅ **Include zeros** (don't drop)
3. ✅ **Add exporter-year and importer-year FE** (absorb MR)
4. ✅ **Cluster SEs by dyad** (account for panel correlation)
5. ✅ **Report pseudo-R²** (check model fit)
6. ✅ **Test for heteroskedasticity** (validate PPML choice)
7. ✅ **Check residuals** (QQ-plot, outliers)

### 12.2 Common Mistakes

❌ **Don't**: Add ln(GDP) when using exporter-importer FE (collinear!)
❌ **Don't**: Use OLS on ln(X+1) (biased and inefficient)
❌ **Don't**: Ignore zeros (selection bias)
❌ **Don't**: Forget to cluster SEs (overstated significance)
❌ **Don't**: Over-interpret pseudo-R² (not the same as OLS R²)

### 12.3 Robustness Checks

Always report:
1. **OLS vs. PPML** (show OLS is biased)
2. **With and without zeros** (show importance)
3. **Different FE specifications** (year FE, dyad FE, etc.)
4. **Alternative SE clustering** (dyad, exporter, importer)

### 12.4 Reporting Standards

**Table format**:
```
                     (1)      (2)      (3)      (4)
                    OLS     PPML     PPML     PPML
                  No FE   No FE   Exp-Imp  Exp-Year
                                    FE     Imp-Year FE
Distance         -0.92*** -0.85*** -0.88*** -0.83***
                 (0.03)   (0.04)   (0.05)   (0.06)

RTA               0.24**   0.18*    0.16*    0.12
                 (0.08)   (0.09)   (0.08)   (0.10)

Fixed Effects      No       No      Yes      Yes
Observations     5000     6500     6500     6500
(Pseudo) R²      0.68     0.72     0.89     0.95
---
Standard errors in parentheses
Clustered by dyad
* p<0.05, ** p<0.01, *** p<0.001
```

---

## Summary

This document covered estimation methods for structural gravity models:

1. **OLS**: Simple but biased (Jensen's inequality, zeros, heteroskedasticity)
2. **PPML**: Preferred estimator (consistent, handles zeros, efficient)
3. **Fixed Effects**: Control for multilateral resistance (essential)
4. **2SLS**: Address endogeneity (RTAs, migration, etc.)
5. **Structural Estimation**: General equilibrium counterfactuals
6. **Standard Errors**: Cluster by dyad (always!)
7. **Diagnostics**: Pseudo-R², residuals, specification tests

**Golden Rule**: Use PPML with exporter-year and importer-year fixed effects, cluster standard errors by dyad.

---

**Companion Documents**:
- [GRAVITY_THEORY.md](GRAVITY_THEORY.md:1): Theoretical foundations
- [TOPOLOGY_THEORY.md](TOPOLOGY_THEORY.md:1): Topological methods
- [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md:1): Practical usage

**References**:
- Santos Silva & Tenreyro (2006): "The Log of Gravity", *Review of Economics and Statistics*
- Anderson & van Wincoop (2003): "Gravity with Gravitas", *American Economic Review*
- Head & Mayer (2014): "Gravity Equations: Workhorse, Toolkit, and Cookbook", *Handbook of International Economics*
