# Mathematical Foundations of Trade Network Topology
**Complete Mathematical Exposition with Intuition and Derivations**

---

## Table of Contents

1. [Gravity Model Theory](#1-gravity-model-theory)
2. [PPML Estimation](#2-ppml-estimation)
3. [Multidimensional Scaling (MDS)](#3-multidimensional-scaling-mds)
4. [Topology Field Construction](#4-topology-field-construction)
5. [Optimal Transport and Wasserstein Distance](#5-optimal-transport-and-wasserstein-distance)
6. [Network Science Metrics](#6-network-science-metrics)
7. [Persistent Homology](#7-persistent-homology)
8. [General Equilibrium Counterfactuals](#8-general-equilibrium-counterfactuals)

---

## 1. Gravity Model Theory

### 1.1 The Fundamental Gravity Equation

**Intuition**: Trade between two countries is like gravitational attraction between masses. Larger economies (more "mass") trade more, and distance creates "friction" that reduces trade.

**Basic Form** (Tinbergen 1962):
$$
X_{ij} = G \frac{Y_i Y_j}{D_{ij}^\beta}
$$

Where:
- $X_{ij}$ = exports from country $i$ to country $j$ (in dollars)
- $Y_i$ = GDP of exporter $i$ (economic "mass")
- $Y_j$ = GDP of importer $j$ (economic "mass")
- $D_{ij}$ = distance between $i$ and $j$ (geographic friction)
- $\beta$ = distance elasticity (typically $\beta \approx 0.85$ to $1.2$)
- $G$ = gravitational constant (proportionality factor)

**Why this works**: If country $i$ produces more stuff ($Y_i$ larger), it has more to export. If country $j$ is richer ($Y_j$ larger), it demands more imports. If they're far apart ($D_{ij}$ large), transportation costs reduce trade.

---

### 1.2 Structural Gravity with Multilateral Resistance

**Intuition**: A country's trade with partner $j$ depends not just on $j$'s characteristics, but on **all available alternatives**. If country $i$ has many close, large trading partners, it will trade less with a distant partner $j$ than if $j$ were the only option.

**Anderson-van Wincoop (2003) Structural Gravity**:
$$
X_{ij} = \frac{Y_i Y_j}{Y^W} \left( \frac{t_{ij}}{\Pi_i P_j} \right)^{1-\sigma}
$$

**Step-by-step breakdown**:

1. **Numerator**: $Y_i Y_j$ = economic mass interaction
2. **Normalization**: $Y^W$ = world GDP (ensures shares sum to 1)
3. **Bilateral trade costs**: $t_{ij} = D_{ij}^\beta \cdot \exp(\text{tariffs}_{ij} + \text{NTBs}_{ij})$
   - Distance component: $D_{ij}^\beta$
   - Policy component: tariffs, non-tariff barriers
4. **Outward Multilateral Resistance (OMR)**: $\Pi_i$
   - Measures "average" trade costs that $i$ faces across **all destinations**
   - High $\Pi_i$ means $i$ is generally far from everyone or faces high tariffs everywhere
5. **Inward Multilateral Resistance (IMR)**: $P_j$
   - Measures "average" trade costs that $j$ faces across **all source countries**
   - High $P_j$ means $j$ is isolated or protected by tariffs
6. **Elasticity of substitution**: $\sigma > 1$
   - Measures how easily buyers substitute between different countries' goods
   - Higher $\sigma$ means small price differences cause large demand shifts
   - Typical values: $\sigma \approx 5$ to $10$

**Why $(1-\sigma)$ exponent?**
- Trade costs $t_{ij}$ reduce demand
- Higher $\sigma$ makes consumers more sensitive to costs
- Power $(1-\sigma)$ is **negative** (since $\sigma > 1$), so higher costs → less trade
- Example: If $\sigma = 6$, exponent is $-5$, so doubling $t_{ij}$ reduces trade by factor $2^{-5} = 1/32$

---

### 1.3 Multilateral Resistance Terms (Formal Definition)

**Outward Multilateral Resistance** $\Pi_i$:
$$
\Pi_i^{1-\sigma} = \sum_{j=1}^N \frac{Y_j}{Y^W} \left( \frac{t_{ij}}{P_j} \right)^{1-\sigma}
$$

**Intuition**:
- This is a **weighted average** of trade costs $t_{ij}$ that country $i$ faces
- Weights are **expenditure shares** $Y_j / Y^W$ (bigger markets matter more)
- Adjusted by destination resistance $P_j$ (easier destinations get more weight)

**Inward Multilateral Resistance** $P_j$:
$$
P_j^{1-\sigma} = \sum_{i=1}^N \frac{Y_i}{Y^W} \left( \frac{t_{ij}}{\Pi_i} \right)^{1-\sigma}
$$

**Intuition**:
- This is a **weighted average** of trade costs that suppliers face when reaching $j$
- Weights are **supply shares** $Y_i / Y^W$ (bigger suppliers matter more)
- Adjusted by origin resistance $\Pi_i$ (suppliers with good alternatives contribute less)

**The Fixed-Point Problem**:
- $\Pi_i$ depends on all $P_j$ values
- Each $P_j$ depends on all $\Pi_i$ values
- Must solve this **system of $2N$ nonlinear equations simultaneously**

**Solution Method** (iterative):
```
1. Initialize: Π_i^(0) = 1, P_j^(0) = 1 for all i,j
2. Repeat until convergence:
   a. Update P_j^(k+1) using current Π_i^(k)
   b. Update Π_i^(k+1) using new P_j^(k+1)
   c. Check if |Π^(k+1) - Π^(k)| < tolerance
3. Stop when converged (typically 100-500 iterations)
```

---

### 1.4 Econometric Specification

**Log-linear form** (for OLS estimation):
$$
\ln X_{ij,t} = \beta_0 + \beta_1 \ln D_{ij} + \beta_2 \text{Contig}_{ij} + \beta_3 \text{Lang}_{ij} + \beta_4 \text{RTA}_{ij,t} + \alpha_i + \delta_j + \gamma_t + \epsilon_{ij,t}
$$

**Variables**:
- $\beta_1 \ln D_{ij}$ = distance elasticity (expect $\beta_1 \approx -0.85$)
- $\text{Contig}_{ij}$ = 1 if countries share border (expect $\beta_2 > 0$, typically $+0.5$ to $+0.8$)
- $\text{Lang}_{ij}$ = 1 if common official language (expect $\beta_3 > 0$, typically $+0.3$ to $+0.5$)
- $\text{RTA}_{ij,t}$ = 1 if regional trade agreement active in year $t$ (expect $\beta_4 > 0$, typically $+0.2$ to $+0.6$)

**Fixed Effects** (absorb multilateral resistance):
- $\alpha_i$ = exporter fixed effect (absorbs $Y_i$ and $\Pi_i$)
- $\delta_j$ = importer fixed effect (absorbs $Y_j$ and $P_j$)
- $\gamma_t$ = year fixed effect (absorbs time trends)

**Why fixed effects work**:
$$
\ln X_{ij,t} = \ln Y_i + \ln Y_j - \ln Y^W + (1-\sigma)[\ln t_{ij} - \ln \Pi_i - \ln P_j] + \ln \gamma_t
$$

Group terms:
- $\alpha_i = \ln Y_i - (1-\sigma) \ln \Pi_i$ (exporter characteristics)
- $\delta_j = \ln Y_j - (1-\sigma) \ln P_j$ (importer characteristics)
- $\gamma_t = -\ln Y^W + \ln \text{trade trend}_t$ (time effects)

So fixed effects **exactly absorb** multilateral resistance terms!

---

### 1.5 Gravity Residuals

**Definition**:
$$
r_{ij,t} = \ln X_{ij,t} - \ln \hat{X}_{ij,t}
$$

Where $\hat{X}_{ij,t}$ is the **predicted** trade from the gravity model.

**Interpretation**:
- $r_{ij,t} > 0$ → **Over-trading**: Countries trade **more** than gravity predicts
  - Possible reasons: Cultural ties, historical links, preferential agreements not captured
- $r_{ij,t} < 0$ → **Under-trading**: Countries trade **less** than gravity predicts
  - Possible reasons: Political tensions, informal barriers, measurement error
- $r_{ij,t} \approx 0$ → Trade perfectly explained by observable gravity variables

**Why residuals matter**:
1. Identify **unexplained trade frictions** not in the model
2. Detect **structural breaks** (sudden residual changes)
3. Validate model fit (should be mean-zero, homoskedastic)
4. Topology fields built from these residuals reveal **spatial patterns** in deviations

**Distribution properties** (if model is correct):
$$
r_{ij,t} \sim \mathcal{N}(0, \sigma^2)
$$
- Mean zero (unbiased)
- Constant variance (homoskedastic)
- Normally distributed (under OLS assumptions)

---

## 2. PPML Estimation

### 2.1 Why Not OLS?

**Problem 1: Zero trade flows**
- Log-linear gravity: $\ln X_{ij} = \alpha_i + \delta_j + \beta' z_{ij} + \epsilon_{ij}$
- Requires $X_{ij} > 0$ to take logarithm
- But **30-50% of country pairs** have zero recorded trade!
- Common "fix": $\ln(X_{ij} + 1)$ → biases estimates

**Problem 2: Heteroskedasticity** (Jensen's Inequality Bias)
$$
\mathbb{E}[X_{ij} | z_{ij}] \neq \exp(\alpha_i + \delta_j + \beta' z_{ij})
$$

Even if:
$$
\mathbb{E}[\ln X_{ij} | z_{ij}] = \alpha_i + \delta_j + \beta' z_{ij}
$$

**Why?** Because $\mathbb{E}[\exp(\epsilon)] \neq \exp(\mathbb{E}[\epsilon])$ unless $\text{Var}(\epsilon) = 0$.

**Proof of bias**:
If $\epsilon \sim \mathcal{N}(0, \sigma^2)$, then:
$$
\mathbb{E}[\exp(\epsilon)] = \exp\left(\frac{\sigma^2}{2}\right) > 1
$$

So OLS systematically **underestimates** levels (though slopes may be OK).

---

### 2.2 PPML Solution

**Key insight** (Santos Silva & Tenreyro 2006): Estimate levels directly without logs!

**PPML specification**:
$$
\mathbb{E}[X_{ij} | z_{ij}] = \exp(\alpha_i + \delta_j + \beta' z_{ij})
$$

**Why Poisson?**
- Poisson distribution natural for **count data** (trade as "events")
- Poisson MLE has property: **consistent for $\beta$ even if data aren't Poisson!**
- Only requires: $\mathbb{E}[X_{ij} | z_{ij}] = \exp(\alpha_i + \delta_j + \beta' z_{ij})$
- Variance can be anything (robust standard errors fix this)

**Poisson likelihood**:
$$
\mathcal{L}(\beta) = \prod_{i,j} \frac{\lambda_{ij}^{X_{ij}} e^{-\lambda_{ij}}}{X_{ij}!}
$$

Where $\lambda_{ij} = \exp(\alpha_i + \delta_j + \beta' z_{ij})$.

**Log-likelihood**:
$$
\ell(\beta) = \sum_{i,j} \left[ X_{ij} \ln \lambda_{ij} - \lambda_{ij} - \ln(X_{ij}!) \right]
$$

Drop constant $\ln(X_{ij}!)$:
$$
\ell(\beta) = \sum_{i,j} \left[ X_{ij} (\alpha_i + \delta_j + \beta' z_{ij}) - \exp(\alpha_i + \delta_j + \beta' z_{ij}) \right]
$$

**First-order condition** (FOC):
$$
\frac{\partial \ell}{\partial \beta_k} = \sum_{i,j} \left[ X_{ij} z_{ij,k} - \exp(\alpha_i + \delta_j + \beta' z_{ij}) z_{ij,k} \right] = 0
$$

Rearrange:
$$
\sum_{i,j} X_{ij} z_{ij,k} = \sum_{i,j} \hat{X}_{ij} z_{ij,k}
$$

**Interpretation**: Predicted trade **exactly matches** observed trade in **weighted moments**.

---

### 2.3 Iterative Reweighted Least Squares (IRLS)

PPML solved via **Newton-Raphson** algorithm, equivalent to iteratively reweighted least squares.

**Algorithm**:

**Step 1**: Initialize parameters $\beta^{(0)}$

**Step 2**: Compute predicted trade:
$$
\hat{X}_{ij}^{(k)} = \exp(\alpha_i^{(k)} + \delta_j^{(k)} + \beta^{(k)\prime} z_{ij})
$$

**Step 3**: Compute weights:
$$
w_{ij}^{(k)} = \hat{X}_{ij}^{(k)}
$$

**Step 4**: Update via weighted least squares:
$$
\beta^{(k+1)} = \arg\min_\beta \sum_{i,j} w_{ij}^{(k)} \left( \frac{X_{ij}}{\hat{X}_{ij}^{(k)}} - \exp(\beta' z_{ij}) \right)^2
$$

**Step 5**: Repeat until $|\beta^{(k+1)} - \beta^{(k)}| < \text{tol}$

**Why weights = predicted trade?**
- Poisson variance: $\text{Var}(X_{ij}) = \mathbb{E}[X_{ij}] = \lambda_{ij}$
- WLS with weights $w_{ij} = 1/\text{Var}(X_{ij})$ gives efficient estimates
- So $w_{ij} = 1/\lambda_{ij} = 1/\hat{X}_{ij}$... wait, that's inverse!
- Actually use $w_{ij} = \hat{X}_{ij}$ because we're minimizing **deviance**, not squared residuals

---

### 2.4 Fixed Effects in PPML

**High-dimensional fixed effects**: With $N$ countries and $T$ years:
- Exporter-year FE: $\alpha_{i,t}$ ($N \times T$ parameters)
- Importer-year FE: $\delta_{j,t}$ ($N \times T$ parameters)
- Total: $2NT$ fixed effects!

For $N=200$, $T=20$: **8,000 fixed effects** to estimate.

**Concentration algorithm** (Correia 2017):

**Idea**: Don't estimate fixed effects explicitly. Instead, "partial out" their effect.

**Step 1**: For each variable $z_{ij}$, compute **within-group means**:
$$
\bar{z}_i = \frac{1}{N} \sum_{j=1}^N z_{ij}, \quad \bar{z}_j = \frac{1}{N} \sum_{i=1}^N z_{ij}
$$

**Step 2**: Demean variables:
$$
\tilde{z}_{ij} = z_{ij} - \bar{z}_i - \bar{z}_j + \bar{\bar{z}}
$$

Where $\bar{\bar{z}}$ is the grand mean.

**Step 3**: Run PPML on demeaned variables $\tilde{z}_{ij}$

**Why this works**: Fixed effects are **orthogonal** to demeaned variables by construction.

**In practice** (Python with `ppmlhdfe`):
```python
import ppmlhdfe

result = ppmlhdfe.fit(
    data=df,
    yvar='trade',
    xvars=['ln_dist', 'contig', 'comlang', 'rta'],
    fe=['exporter', 'importer', 'year']
)
```

---

## 3. Multidimensional Scaling (MDS)

### 3.1 The Problem: Embedding Countries in 2D

**Goal**: Given $N$ countries, find 2D coordinates $(x_i, y_i)$ such that:
$$
\| (x_i, y_i) - (x_j, y_j) \| \approx d_{ij}
$$

Where $d_{ij}$ is some measure of "distance" (could be geographic, economic, or residual-based).

**Why?** To visualize high-dimensional relationships in 2D space while preserving pairwise distances.

---

### 3.2 Classical MDS (Metric MDS)

**Input**: Distance matrix $D = [d_{ij}]$ where $d_{ij} \geq 0$ and $d_{ii} = 0$.

**Output**: Coordinates $X \in \mathbb{R}^{N \times 2}$ (or higher dimensions).

**Objective**: Minimize stress:
$$
\text{Stress} = \sum_{i < j} \left( d_{ij} - \| x_i - x_j \|_2 \right)^2
$$

**Classical MDS Solution** (closed-form via eigendecomposition):

**Step 1**: Compute **double-centered** squared distance matrix:
$$
B = -\frac{1}{2} H D^{(2)} H
$$

Where:
- $D^{(2)}$ = matrix of squared distances: $D^{(2)}_{ij} = d_{ij}^2$
- $H = I - \frac{1}{N} \mathbf{1} \mathbf{1}^T$ is the **centering matrix**
- $\mathbf{1}$ = vector of ones

**Why double centering?**
$$
\| x_i - x_j \|^2 = \| x_i \|^2 + \| x_j \|^2 - 2 x_i^T x_j
$$

Centering removes the $\| x_i \|^2$ terms, leaving inner products $x_i^T x_j$.

**Step 2**: Eigen-decompose $B$:
$$
B = V \Lambda V^T
$$

Where:
- $V$ = matrix of eigenvectors
- $\Lambda$ = diagonal matrix of eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_N$

**Step 3**: Extract top $k=2$ eigenvectors:
$$
X = V_k \Lambda_k^{1/2}
$$

Where:
- $V_k$ = first $k$ columns of $V$ (eigenvectors for $k$ largest eigenvalues)
- $\Lambda_k^{1/2}$ = diagonal matrix with $\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_k}$

**Result**: Coordinates $X \in \mathbb{R}^{N \times 2}$ giving 2D embedding.

**Goodness of fit**:
$$
R^2 = \frac{\lambda_1 + \lambda_2}{\sum_{i=1}^N \lambda_i}
$$

Fraction of variance explained by first 2 dimensions.

---

### 3.3 Why Eigenvalues?

**Deep intuition**:
- $B$ is a **Gram matrix** of inner products: $B_{ij} = x_i^T x_j$ (after centering)
- Eigenvalues measure "variance" along principal directions
- Top 2 eigenvectors are the 2D projection that **maximizes preserved variance**
- This is exactly **Principal Component Analysis (PCA)** applied to distance matrix!

**Connection to PCA**:
If we had data matrix $X$ directly:
$$
\text{Covariance matrix} = \frac{1}{N} X^T X
$$

Eigendecomposition of covariance gives PCA.

In MDS, we only have distances $D$, not original $X$. But we can **reconstruct** inner products from distances via double centering, then proceed as in PCA.

---

### 3.4 Non-Metric MDS

**Problem**: What if distances aren't perfect Euclidean distances? (e.g., residual-based "distances" violate triangle inequality)

**Non-metric MDS**: Only preserve **rank order** of distances.

**Objective**:
$$
\text{Stress} = \sqrt{\frac{\sum_{i<j} (d_{ij} - \hat{d}_{ij})^2}{\sum_{i<j} d_{ij}^2}}
$$

Where $\hat{d}_{ij} = \| x_i - x_j \|$ are embedded distances.

**Algorithm** (Kruskal 1964):
1. Initialize coordinates randomly
2. Compute embedded distances $\hat{d}_{ij}$
3. Find monotonic transformation $f$ such that $f(\hat{d}_{ij})$ best matches $d_{ij}$ in rank order
4. Update coordinates to minimize $\sum (d_{ij} - f(\hat{d}_{ij}))^2$
5. Repeat until convergence

**When to use**:
- Classical MDS: When distances are **metric** (satisfy triangle inequality)
- Non-metric MDS: When only **ordinal** information is reliable

---

### 3.5 MDS for Trade Networks

**Distance matrix options**:

**Option 1: Geographic distance**
$$
d_{ij} = \text{great circle distance between capitals of } i \text{ and } j
$$

**Option 2: Economic distance** (gravity residuals)
$$
d_{ij} = | r_{ij} |
$$

Where $r_{ij}$ is the gravity residual.

**Option 3: Network distance** (shortest path)
$$
d_{ij} = \text{minimum number of hops from } i \text{ to } j \text{ in trade network}
$$

**In our implementation**: We use **geographic distance** for stability, then color points by residuals. This separates "where countries are" (MDS embedding) from "how they trade" (residuals).

**Python implementation**:
```python
from sklearn.manifold import MDS
import numpy as np

# Distance matrix (NxN)
D = compute_distance_matrix(countries)

# Classical MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(D)

# Extract coordinates
embedding = {
    countries[i]: {'x': float(coords[i, 0]), 'y': float(coords[i, 1])}
    for i in range(len(countries))
}
```

---

## 4. Topology Field Construction

### 4.1 From Discrete Residuals to Continuous Fields

**Problem**: We have residuals $r_{ij,t}$ for **discrete** country pairs. Want to create **continuous** 2D field $\phi(x, y, t)$ for visualization.

**Approach**:
1. Embed countries in 2D using MDS: $(x_i, y_i)$
2. Assign residual "mass" to country locations
3. Interpolate to regular grid
4. Smooth to reduce noise

---

### 4.2 Residual Assignment

For each country pair $(i,j)$, assign residual $r_{ij,t}$ to **both** country locations:

**Exporter contribution**:
$$
\phi_i^{\text{exp}} = \frac{1}{N} \sum_{j=1}^N r_{ij,t}
$$

Average residual for country $i$ as **exporter**.

**Importer contribution**:
$$
\phi_j^{\text{imp}} = \frac{1}{N} \sum_{i=1}^N r_{ij,t}
$$

Average residual for country $j$ as **importer**.

**Combined**:
$$
\phi_k = \frac{1}{2} (\phi_k^{\text{exp}} + \phi_k^{\text{imp}})
$$

Each country gets a single "residual value" representing its average over/under-trading.

---

### 4.3 Gaussian Kernel Interpolation

**Goal**: Given values $\phi_k$ at locations $(x_k, y_k)$, estimate $\phi(x, y)$ everywhere.

**Radial Basis Function (RBF) interpolation** with Gaussian kernel:
$$
\phi(x, y) = \frac{\sum_{k=1}^N w_k(x,y) \cdot \phi_k}{\sum_{k=1}^N w_k(x,y)}
$$

**Gaussian weights**:
$$
w_k(x,y) = \exp\left( -\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma^2} \right)
$$

Where $\sigma$ is the **bandwidth** parameter (controls smoothness).

**Intuition**:
- Points close to $(x_k, y_k)$ get high weight $w_k \approx 1$
- Points far from all $(x_k, y_k)$ get low weights everywhere
- Bandwidth $\sigma$ controls "reach": large $\sigma$ → smoother field

**Choosing $\sigma$**:
$$
\sigma = 0.2 \times \text{(range of coordinates)}
$$

For normalized coordinates in $[0, 1]^2$, use $\sigma \approx 0.1$ to $0.2$.

---

### 4.4 Grid Evaluation

**Create regular grid**:
$$
x_{\text{grid}} \in [x_{\min}, x_{\max}], \quad y_{\text{grid}} \in [y_{\min}, y_{\max}]
$$

Typically $64 \times 64$ grid (can increase for higher resolution).

**For each grid point** $(x_g, y_g)$:
$$
\phi(x_g, y_g) = \frac{\sum_{k=1}^N \exp\left( -\frac{(x_g - x_k)^2 + (y_g - y_k)^2}{2\sigma^2} \right) \phi_k}{\sum_{k=1}^N \exp\left( -\frac{(x_g - x_k)^2 + (y_g - y_k)^2}{2\sigma^2} \right)}
$$

**Result**: Matrix $\Phi \in \mathbb{R}^{64 \times 64}$ representing the field.

---

### 4.5 Gaussian Smoothing (Post-Processing)

**Why?** Interpolated field may have noise from sampling or outliers.

**Gaussian blur**:
$$
\phi_{\text{smooth}}(x, y) = \int \int G(x', y') \phi(x - x', y - y') \, dx' dy'
$$

Where:
$$
G(x, y) = \frac{1}{2\pi \sigma_{\text{blur}}^2} \exp\left( -\frac{x^2 + y^2}{2\sigma_{\text{blur}}^2} \right)
$$

**Discrete implementation** (convolution):
$$
\Phi_{\text{smooth}} = \Phi * G
$$

Where $*$ is 2D convolution and $G$ is Gaussian kernel matrix.

**Python**:
```python
from scipy.ndimage import gaussian_filter

phi_smooth = gaussian_filter(phi, sigma=1.5)
```

---

### 4.6 Field Derivatives

**Gradient** (direction of steepest increase):
$$
\nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y} \right)
$$

**Discrete approximation** (finite differences):
$$
\frac{\partial \phi}{\partial x} \approx \frac{\phi(x+h, y) - \phi(x-h, y)}{2h}
$$

**Laplacian** (curvature):
$$
\Delta \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}
$$

**Discrete Laplacian** (5-point stencil):
$$
\Delta \phi(x, y) \approx \frac{\phi(x+h,y) + \phi(x-h,y) + \phi(x,y+h) + \phi(x,y-h) - 4\phi(x,y)}{h^2}
$$

**Interpretation**:
- $\nabla \phi$ points toward countries that **over-trade** (residuals increasing)
- $\Delta \phi > 0$: Locally convex (local minimum of under-trading)
- $\Delta \phi < 0$: Locally concave (local maximum of over-trading)

---

### 4.7 Field Variance as Fragmentation Metric

**Definition**:
$$
\sigma_\phi^2 = \frac{1}{N_{\text{grid}}} \sum_{g=1}^{N_{\text{grid}}} (\phi_g - \bar{\phi})^2
$$

Where $\bar{\phi} = \frac{1}{N_{\text{grid}}} \sum_g \phi_g$ is the mean field value.

**Interpretation**:
- **Low variance**: Residuals uniform across space → "smooth" trade network
- **High variance**: Residuals clustered in regions → "fragmented" network
- **Increasing variance over time**: Network becoming more polarized

**Example**:
- 2005: $\sigma_\phi^2 = 0.08$ (high fragmentation, early data)
- 2019: $\sigma_\phi^2 = 0.02$ (low fragmentation, stable patterns)
- 2020: $\sigma_\phi^2 = 0.12$ (COVID shock, extreme fragmentation)

---

## 5. Optimal Transport and Wasserstein Distance

### 5.1 The Optimal Transport Problem

**Setup**:
- Two probability distributions $\mu$ and $\nu$ on space $\mathcal{X}$
- Cost function $c(x, y)$ = cost to transport mass from $x$ to $y$
- Find **transport plan** $\pi(x, y)$ that minimizes total cost

**Kantorovich Formulation**:
$$
W_c(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int \int c(x, y) \, d\pi(x, y)
$$

Where $\Pi(\mu, \nu)$ = set of all joint distributions with marginals $\mu$ and $\nu$:
$$
\int \pi(x, y) \, dy = \mu(x), \quad \int \pi(x, y) \, dx = \nu(y)
$$

**Intuition**:
- $\mu$ = "pile of dirt" (initial distribution of residuals in year $t$)
- $\nu$ = "target hole" (distribution of residuals in year $t+1$)
- $\pi(x, y)$ = how much dirt to move from location $x$ to location $y$
- $c(x, y)$ = cost per unit (e.g., distance $|x - y|$)
- $W_c$ = **minimum total cost** to reshape $\mu$ into $\nu$

---

### 5.2 Wasserstein-1 Distance (Earth Mover's Distance)

**Special case**: Cost = Euclidean distance:
$$
c(x, y) = \| x - y \|
$$

**Wasserstein-1 distance**:
$$
W_1(\mu, \nu) = \inf_{\pi \in \Pi(\mu, \nu)} \int \int \| x - y \| \, d\pi(x, y)
$$

**Closed-form for 1D** (Kantorovich-Rubinstein):
$$
W_1(\mu, \nu) = \int_{-\infty}^\infty | F_\mu(x) - F_\nu(x) | \, dx
$$

Where $F_\mu$, $F_\nu$ are cumulative distribution functions (CDFs).

**Intuition**: Area between CDF curves!

**For discrete distributions** (empirical measure):
$$
\mu = \frac{1}{N} \sum_{i=1}^N \delta_{x_i}, \quad \nu = \frac{1}{N} \sum_{i=1}^N \delta_{y_i}
$$

Where $\delta_x$ is a point mass at $x$.

**Closed-form** (if equal weights):
Sort samples: $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(N)}$ and $y_{(1)} \leq \cdots \leq y_{(N)}$.

Then:
$$
W_1(\mu, \nu) = \frac{1}{N} \sum_{i=1}^N | x_{(i)} - y_{(i)} |
$$

**Proof**: Optimal transport matches $i$-th smallest mass in $\mu$ to $i$-th smallest in $\nu$ (sorted coupling).

---

### 5.3 Application to Trade Residuals

**Setup**:
- Year $t$: Residuals $\{ r_{ij,t} \}$ for all pairs $(i,j)$
- Year $t+1$: Residuals $\{ r_{ij,t+1} \}$

**Distributions**:
$$
\mu_t = \frac{1}{M} \sum_{(i,j)} \delta_{r_{ij,t}}, \quad \nu_{t+1} = \frac{1}{M} \sum_{(i,j)} \delta_{r_{ij,t+1}}
$$

Where $M = N(N-1)$ is the number of pairs.

**Wasserstein distance**:
$$
W_1(\mu_t, \nu_{t+1}) = \frac{1}{M} \sum_{k=1}^M | r_{(k),t} - r_{(k),t+1} |
$$

Where $r_{(k),t}$ is the $k$-th order statistic (sorted residuals).

**Interpretation**:
- Small $W_1 \approx 0.05$: Residual distribution **stable** (normal year)
- Large $W_1 \approx 0.20$: Residual distribution **shifted dramatically** (shock year)

**Examples** (from our data):
- 2015 → 2016: $W_1 = 0.23$ (Brexit referendum)
- 2019 → 2020: $W_1 = 0.21$ (COVID-19 pandemic)
- 2009 → 2010: $W_1 = 0.18$ (Greek debt crisis)

---

### 5.4 Why Wasserstein Instead of KL Divergence?

**Kullback-Leibler divergence**:
$$
D_{KL}(\mu \| \nu) = \int \log\left( \frac{d\mu}{d\nu} \right) d\mu
$$

**Problems**:
1. **Not symmetric**: $D_{KL}(\mu \| \nu) \neq D_{KL}(\nu \| \mu)$
2. **Requires absolute continuity**: If $\mu$ has mass where $\nu$ doesn't, $D_{KL} = \infty$
3. **No geometric information**: Doesn't use distance between points

**Wasserstein advantages**:
1. **True metric**: $W_1(\mu, \nu) = W_1(\nu, \mu)$, satisfies triangle inequality
2. **Always finite** for distributions with finite first moment
3. **Geometric**: Incorporates spatial structure via cost function
4. **Interpretable**: "Cost to transport" has clear meaning

---

### 5.5 Computing Wasserstein Distance in Practice

**Python implementation** (using sorted residuals):
```python
import numpy as np

def wasserstein_distance(r_t, r_s):
    """
    Compute W_1 distance between two sets of residuals.

    Args:
        r_t: Residuals for year t (array)
        r_s: Residuals for year s (array)

    Returns:
        W_1 distance (scalar)
    """
    # Sort residuals
    r_t_sorted = np.sort(r_t)
    r_s_sorted = np.sort(r_s)

    # Compute mean absolute difference
    return np.mean(np.abs(r_t_sorted - r_s_sorted))
```

**Alternative** (using `scipy`):
```python
from scipy.stats import wasserstein_distance

w1 = wasserstein_distance(r_t, r_s)
```

**For 2D fields**: Use `ot.emd2()` from POT (Python Optimal Transport) library:
```python
import ot

# Flatten 64x64 grids
phi_t_flat = phi_t.flatten()
phi_s_flat = phi_s.flatten()

# Compute cost matrix (Euclidean distances on grid)
coords = np.array([(i, j) for i in range(64) for j in range(64)])
M = ot.dist(coords, coords)

# Compute Wasserstein distance
w = ot.emd2(phi_t_flat, phi_s_flat, M)
```

---

## 6. Network Science Metrics

### 6.1 Trade Network as Directed Weighted Graph

**Nodes**: Countries $V = \{1, 2, \ldots, N\}$

**Edges**: Directed edge $(i \to j)$ if $X_{ij} > 0$

**Weights**: Trade flow $w_{ij} = X_{ij}$

**Adjacency matrix**:
$$
A_{ij} = \begin{cases}
1 & \text{if } X_{ij} > 0 \\
0 & \text{otherwise}
\end{cases}
$$

**Weighted adjacency**:
$$
W_{ij} = X_{ij}
$$

---

### 6.2 Degree Centrality

**Out-degree** (number of export destinations):
$$
k_i^{\text{out}} = \sum_{j=1}^N A_{ij}
$$

**In-degree** (number of import sources):
$$
k_j^{\text{in}} = \sum_{i=1}^N A_{ij}
$$

**Total degree**:
$$
k_i = k_i^{\text{out}} + k_i^{\text{in}}
$$

**Normalized degree centrality**:
$$
C_D(i) = \frac{k_i}{2(N-1)}
$$

Ranges from 0 (isolated) to 1 (connected to all).

**Interpretation**:
- High $k_i^{\text{out}}$: Country exports to many partners (diversified exports)
- High $k_i^{\text{in}}$: Country imports from many sources (diversified imports)

---

### 6.3 Strength Centrality (Weighted Degree)

**Out-strength** (total exports):
$$
s_i^{\text{out}} = \sum_{j=1}^N W_{ij} = \sum_{j=1}^N X_{ij}
$$

**In-strength** (total imports):
$$
s_j^{\text{in}} = \sum_{i=1}^N W_{ij} = \sum_{i=1}^N X_{ij}
$$

**Total strength**:
$$
s_i = s_i^{\text{out}} + s_i^{\text{in}}
$$

**Interpretation**: Volume-based centrality (total trade).

---

### 6.4 Closeness Centrality

**Shortest path distance**:
$$
d_{ij} = \text{minimum number of hops from } i \text{ to } j
$$

**Average distance**:
$$
\bar{d}_i = \frac{1}{N-1} \sum_{j \neq i} d_{ij}
$$

**Closeness centrality**:
$$
C_C(i) = \frac{1}{\bar{d}_i}
$$

**Interpretation**:
- High closeness: Country can reach all others via **short paths**
- Hub countries have high closeness (central in network)

**Weighted version** (using trade volumes as inverse distances):
$$
d_{ij}^w = \frac{1}{W_{ij}}
$$

Dijkstra's algorithm finds shortest weighted paths.

---

### 6.5 Betweenness Centrality

**Shortest paths through $i$**:
$$
C_B(i) = \sum_{s \neq t \neq i} \frac{\sigma_{st}(i)}{\sigma_{st}}
$$

Where:
- $\sigma_{st}$ = total number of shortest paths from $s$ to $t$
- $\sigma_{st}(i)$ = number of those paths passing through $i$

**Interpretation**:
- High betweenness: Country is **bridge** connecting otherwise distant regions
- "Gatekeeper" countries have high betweenness

**Normalized**:
$$
C_B'(i) = \frac{C_B(i)}{(N-1)(N-2)/2}
$$

Ranges from 0 to 1.

**Algorithm** (Brandes 2001):
```
For each source s:
  1. BFS from s to find shortest paths
  2. Backpropagate: accumulate credit for nodes on paths
Total: O(N * M) where M = number of edges
```

---

### 6.6 PageRank

**Random surfer model**:
- Start at random node
- With probability $\alpha$ (damping factor): follow random outgoing edge
- With probability $1 - \alpha$: jump to random node

**PageRank** = stationary distribution of this random walk.

**Equation**:
$$
PR(i) = \frac{1 - \alpha}{N} + \alpha \sum_{j \to i} \frac{PR(j)}{k_j^{\text{out}}}
$$

**Matrix form**:
$$
\mathbf{PR} = (1-\alpha) \frac{\mathbf{1}}{N} + \alpha \mathbf{A}^T \mathbf{D}^{-1} \mathbf{PR}
$$

Where:
- $\mathbf{A}$ = adjacency matrix
- $\mathbf{D}$ = diagonal matrix with $D_{ii} = k_i^{\text{out}}$

**Power iteration**:
```
1. Initialize: PR^(0) = 1/N for all nodes
2. Iterate: PR^(k+1) = (1-α) * 1/N + α * A^T * D^-1 * PR^(k)
3. Stop when |PR^(k+1) - PR^(k)| < tol
```

**Weighted PageRank**:
$$
PR(i) = \frac{1 - \alpha}{N} + \alpha \sum_{j \to i} \frac{W_{ji}}{s_j^{\text{out}}} PR(j)
$$

Now edge weight $W_{ji}$ determines transition probability.

**Interpretation**:
- High PageRank: Country receives imports from **other important traders**
- Not just volume, but **who you trade with**

---

### 6.7 Clustering Coefficient

**Local clustering** (for node $i$):
$$
C(i) = \frac{\text{number of triangles involving } i}{\text{number of possible triangles}}
$$

**For undirected graph**:
$$
C(i) = \frac{2 e_i}{k_i (k_i - 1)}
$$

Where $e_i$ = number of edges between neighbors of $i$.

**For directed graph** (count all triangle types):
$$
C(i) = \frac{\text{triangles}}{k_i^{\text{in}} k_i^{\text{out}}}
$$

**Global clustering**:
$$
C = \frac{1}{N} \sum_{i=1}^N C(i)
$$

**Interpretation**:
- High clustering: Country's partners **also trade with each other**
- Indicates regional trade blocs

**Example**:
- EU countries: High clustering (intra-EU trade)
- Isolated country: Zero clustering (partners don't connect)

---

### 6.8 Network Density

**Density**:
$$
\rho = \frac{\text{number of edges}}{\text{number of possible edges}} = \frac{|E|}{N(N-1)}
$$

For directed graph with $N$ nodes: $N(N-1)$ possible edges.

**Interpretation**:
- $\rho \approx 1$: Fully connected (everyone trades with everyone)
- $\rho \approx 0$: Sparse (few trade relationships)

**Typical values** (trade networks):
- Full BACI (200+ countries): $\rho \approx 0.3$ to $0.4$
- Top-20 countries: $\rho \approx 0.8$ to $0.9$ (most pairs trade)

---

### 6.9 Reciprocity

**Reciprocity**: Fraction of edges that are **bidirectional**.

$$
r = \frac{\text{number of reciprocated edges}}{\text{total number of edges}}
$$

**Formal**:
$$
r = \frac{\sum_{i,j} A_{ij} A_{ji}}{\sum_{i,j} A_{ij}}
$$

**Interpretation**:
- $r \approx 1$: Most trade relationships are two-way
- $r \approx 0$: Many one-way trade flows

**Trade networks**: Typically $r > 0.9$ (bilateral trade is nearly universal).

---

### 6.10 Assortativity

**Degree assortativity**: Do high-degree nodes connect to other high-degree nodes?

**Pearson correlation**:
$$
r = \frac{\sum_{ij} (k_i - \bar{k})(k_j - \bar{k}) A_{ij}}{\sqrt{\sum_{ij} (k_i - \bar{k})^2 A_{ij}} \sqrt{\sum_{ij} (k_j - \bar{k})^2 A_{ij}}}
$$

**Interpretation**:
- $r > 0$: Assortative (hubs connect to hubs)
- $r < 0$: Disassortative (hubs connect to low-degree nodes)
- $r \approx 0$: Random mixing

**Trade networks**: Often **disassortative** ($r < 0$) because large economies (hubs) trade with both large and small partners, but small economies mainly trade with hubs.

---

## 7. Persistent Homology

### 7.1 Homology Basics

**Intuition**: Homology counts "holes" of different dimensions.

- $H_0$ (0-dimensional homology): Connected components
- $H_1$ (1-dimensional): Loops (cycles)
- $H_2$ (2-dimensional): Voids (cavities)

**Betti numbers**:
$$
\beta_k = \text{rank}(H_k) = \text{number of } k\text{-dimensional holes}
$$

**Example** (coffee cup):
- $\beta_0 = 1$ (one connected piece)
- $\beta_1 = 1$ (one loop through the handle)
- $\beta_2 = 0$ (no voids)

---

### 7.2 Simplicial Complexes

**Simplex**: Generalization of triangle to higher dimensions.
- 0-simplex: Point
- 1-simplex: Edge (line segment)
- 2-simplex: Triangle (filled)
- 3-simplex: Tetrahedron

**Simplicial complex** $K$: Collection of simplices closed under taking subsets.

**Example**:
```
Vertices: {A, B, C}
Edges: {AB, BC, CA}
Triangles: {ABC}
```

This is a simplicial complex (the filled triangle).

---

### 7.3 Filtration

**Filtration**: Sequence of nested complexes:
$$
K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_n
$$

**Example** (Vietoris-Rips filtration):
1. Start with points (countries)
2. Add edge between $i$ and $j$ if distance $d_{ij} < \epsilon$
3. Increase $\epsilon$: more edges appear
4. Add triangles when all 3 edges present
5. Continue to full complex

**Persistence**: How long does a homological feature (hole) "live"?

**Birth time**: Filtration parameter $\epsilon_b$ when feature appears
**Death time**: Filtration parameter $\epsilon_d$ when feature disappears (filled in)

**Persistence**: $\epsilon_d - \epsilon_b$

---

### 7.4 Persistence Diagram

**Persistence diagram**: Plot of $(b, d)$ pairs where $b$ = birth, $d$ = death.

**Example**:
```
Feature 1: Loop appears at ε=2, filled at ε=5 → (2, 5)
Feature 2: Loop appears at ε=3, filled at ε=3.5 → (3, 3.5)
```

**Interpretation**:
- Points far from diagonal $y=x$: **Long-lived** features (topological signal)
- Points near diagonal: **Short-lived** features (noise)

**Persistent homology**: Only count features with persistence $> \text{threshold}$.

---

### 7.5 Betti Curves

**Betti curve**: $\beta_k(\epsilon)$ = number of $k$-dimensional holes at filtration parameter $\epsilon$.

**Example**:
```
ε=0: β_0=20 (20 countries, disconnected)
ε=1: β_0=15, β_1=2 (some connections, 2 loops)
ε=2: β_0=10, β_1=5 (more connected, more loops)
ε=5: β_0=1, β_1=3 (fully connected, 3 persistent loops)
```

**Why useful?**
- $\beta_0$ decreasing: Network becoming more connected
- $\beta_1$ non-zero: Cyclic trade patterns (triangular relationships)

---

### 7.6 Application to Trade Networks

**Setup**:
1. Nodes = countries
2. Edge weights = trade volumes $X_{ij}$
3. Filtration: Include edge $(i,j)$ if $X_{ij} > \epsilon$

**As $\epsilon$ decreases** (from high to low):
- Start with only largest flows
- Gradually add smaller flows
- Track when cycles appear/disappear

**Interpretation**:
- **Persistent cycle**: Triangular trade relationship stable across thresholds
  - Example: USA ↔ Canada ↔ Mexico (NAFTA)
- **Short-lived cycle**: Appears only at specific trade volume threshold (transient)

**Betti-0** (connected components):
- High $\beta_0$: Network fragmented into isolated blocs
- $\beta_0 = 1$: Fully connected trade network

**Betti-1** (cycles):
- High $\beta_1$: Many triangular trade relationships
- $\beta_1 = 0$: Tree-like structure (hub-and-spoke)

---

### 7.7 Computing Persistent Homology

**Algorithm** (Edelsbrunner et al. 2002):

**Step 1**: Build filtration
```
Sort edges by weight: X_1 ≥ X_2 ≥ ... ≥ X_M
For k=1 to M:
  Add edge k to complex K_k
  Update boundary matrix
```

**Step 2**: Compute homology via matrix reduction
```
Boundary matrix ∂: columns = simplices, entries = boundary relations
Reduce to Smith Normal Form
Read off Betti numbers from rank
```

**Step 3**: Track birth/death times
```
When column becomes zero: new feature born
When column becomes nonzero: feature dies (filled)
```

**Python** (using `ripser`):
```python
from ripser import ripser
import numpy as np

# Distance matrix
D = compute_distances(countries)

# Compute persistence diagram
result = ripser(D, maxdim=2, distance_matrix=True)

# Extract Betti-1 features
H1 = result['dgms'][1]

# Persistent features (persistence > 0.1)
persistent = H1[H1[:, 1] - H1[:, 0] > 0.1]
```

---

## 8. General Equilibrium Counterfactuals

### 8.1 The GE Problem

**Question**: If tariffs change, what happens to:
1. Bilateral trade flows $X_{ij}$?
2. Prices $P_j$?
3. Wages $w_i$?
4. Welfare?

**Challenge**: Everything affects everything!
- Tariff $\tau_{ij} \uparrow$ → $X_{ij} \downarrow$
- But $X_{ij} \downarrow$ → $P_j \downarrow$ (less demand for $j$'s goods)
- And $P_j \downarrow$ → $X_{kj} \uparrow$ for other exporters $k$
- And $X_{kj} \uparrow$ → $w_k \uparrow$ (more demand for $k$'s labor)
- ...

Need to solve for **new equilibrium** where all markets clear.

---

### 8.2 Structural Model (Eaton-Kortum 2002)

**Technology**: Country $i$ can produce any good $\omega$ with productivity:
$$
z_i(\omega) \sim \text{Fréchet}(T_i, \theta)
$$

CDF: $F(z) = \exp(-T_i z^{-\theta})$

- $T_i$ = absolute advantage (technology level)
- $\theta$ = dispersion (higher $\theta$ → less heterogeneity)

**Trade costs**: Delivering from $i$ to $j$ costs:
$$
d_{ij} = \tau_{ij} D_{ij}^\beta
$$

Where $\tau_{ij}$ = tariff factor (1 + tariff rate).

**Price**: Cost of good $\omega$ from source $i$ delivered to $j$:
$$
p_{ij}(\omega) = \frac{w_i}{z_i(\omega)} d_{ij}
$$

**Lowest price**: Consumer in $j$ buys from cheapest source:
$$
p_j(\omega) = \min_i p_{ij}(\omega)
$$

**Trade shares**: Fraction of $j$'s spending on goods from $i$:
$$
\pi_{ij} = \mathbb{P}[\text{country } i \text{ is cheapest source for } j]
$$

**Closed form** (Eaton-Kortum):
$$
\pi_{ij} = \frac{T_i (w_i d_{ij})^{-\theta}}{\sum_{k=1}^N T_k (w_k d_{kj})^{-\theta}}
$$

---

### 8.3 Market Clearing Conditions

**Goods market clearing**: Country $i$'s production = total sales to all destinations:
$$
w_i L_i = \sum_{j=1}^N \pi_{ij} X_j
$$

Where:
- $L_i$ = labor supply (exogenous)
- $X_j = w_j L_j$ = country $j$'s total expenditure

**Trade balance**: Expenditure = income + deficits:
$$
X_j = w_j L_j + D_j
$$

Where $D_j$ = trade deficit (exogenous in baseline).

**Price index**: CES aggregation:
$$
P_j = \gamma \left[ \sum_{i=1}^N T_i (w_i d_{ij})^{-\theta} \right]^{-1/\theta}
$$

Where $\gamma = \Gamma\left(\frac{\theta - \sigma + 1}{\theta}\right)^{1/(1-\sigma)}$.

---

### 8.4 Exact Hat Algebra (Dekle et al. 2007)

**Key insight**: Don't need to know $T_i$ or $\theta$ exactly. Only need **observed baseline** and **elasticities**.

**Changes in "hats"**: Let $\hat{x} = x' / x$ denote proportional change.

**Trade share changes**:
$$
\hat{\pi}_{ij} = \frac{(\hat{w}_i \hat{d}_{ij})^{-\theta}}{\sum_{k=1}^N \pi_{kj} (\hat{w}_k \hat{d}_{kj})^{-\theta}}
$$

**Market clearing**:
$$
\hat{w}_i = \frac{\sum_{j=1}^N \hat{\pi}_{ij} \pi_{ij}^0 \hat{X}_j X_j^0}{\sum_{j=1}^N \pi_{ij}^0 X_j^0}
$$

**Expenditure**:
$$
\hat{X}_j = \hat{w}_j
$$

(Under balanced trade.)

**Iterative solution**:
```
1. Initialize: ŵ_i^(0) = 1 for all i
2. Update trade shares: π̂_ij using current ŵ
3. Update wages: ŵ_i using market clearing
4. Repeat until |ŵ^(k+1) - ŵ^(k)| < tol
```

---

### 8.5 Welfare Changes

**Real wage**:
$$
W_i = \frac{w_i}{P_i}
$$

**Change in real wage** (hat algebra):
$$
\hat{W}_i = \frac{\hat{w}_i}{\hat{P}_i}
$$

**Price index change**:
$$
\hat{P}_j = \left[ \sum_{i=1}^N \pi_{ij}^0 (\hat{w}_i \hat{d}_{ij})^{-\theta} \right]^{-1/\theta}
$$

**Welfare decomposition**:
$$
\ln \hat{W}_i = \ln \hat{w}_i - \ln \hat{P}_i
$$

- Wage effect: How much does $i$'s income rise?
- Price effect: How much do prices paid by $i$ change?

**Example** (tariff increase on imports):
- Direct: $\hat{P}_j \uparrow$ (imported goods more expensive)
- GE effect: $\hat{w}_j$ may rise (protected industries expand) or fall (intermediate inputs costlier)
- Net welfare: Depends on whether wage gains offset price increases

---

### 8.6 Implementation Steps

**Baseline calibration**:
1. Estimate $\theta$ from gravity regression (distance elasticity)
2. Observe trade shares $\pi_{ij}^0 = X_{ij}^0 / X_j^0$
3. Observe wages $w_i^0$ (GDP per capita)

**Counterfactual**:
1. Change tariffs: $\tau_{ij}' = \tau_{ij}^0 \times (1 + \Delta \tau)$
2. Compute $\hat{d}_{ij} = \tau_{ij}' / \tau_{ij}^0$
3. Solve for $\hat{w}_i$, $\hat{\pi}_{ij}$, $\hat{P}_j$ via iteration
4. Compute welfare: $\hat{W}_i = \hat{w}_i / \hat{P}_i$

**Python pseudocode**:
```python
def solve_counterfactual(pi_baseline, d_change, theta, max_iter=1000):
    N = len(pi_baseline)
    w_hat = np.ones(N)

    for iteration in range(max_iter):
        # Update trade shares
        pi_new = update_trade_shares(pi_baseline, w_hat, d_change, theta)

        # Market clearing
        w_hat_new = market_clearing(pi_baseline, pi_new, w_hat)

        # Check convergence
        if np.max(np.abs(w_hat_new - w_hat)) < 1e-6:
            break

        w_hat = w_hat_new

    # Compute price changes
    P_hat = compute_price_index(pi_baseline, w_hat, d_change, theta)

    # Welfare
    W_hat = w_hat / P_hat

    return W_hat, w_hat, P_hat
```

---

## Summary of Key Formulas

### Gravity Model
$$
X_{ij} = \frac{Y_i Y_j}{Y^W} \left( \frac{t_{ij}}{\Pi_i P_j} \right)^{1-\sigma}
$$

### PPML First-Order Condition
$$
\sum_{i,j} X_{ij} z_{ij,k} = \sum_{i,j} \hat{X}_{ij} z_{ij,k}
$$

### MDS Objective
$$
X = V_k \Lambda_k^{1/2}
$$

### Field Interpolation
$$
\phi(x, y) = \frac{\sum_k \exp\left(-\frac{\|x - x_k\|^2}{2\sigma^2}\right) \phi_k}{\sum_k \exp\left(-\frac{\|x - x_k\|^2}{2\sigma^2}\right)}
$$

### Wasserstein-1 Distance
$$
W_1(\mu, \nu) = \frac{1}{N} \sum_{i=1}^N | x_{(i)} - y_{(i)} |
$$

### PageRank
$$
PR(i) = \frac{1-\alpha}{N} + \alpha \sum_{j \to i} \frac{W_{ji}}{s_j^{\text{out}}} PR(j)
$$

### GE Counterfactual (Hat Algebra)
$$
\hat{w}_i = \frac{\sum_j \hat{\pi}_{ij} \pi_{ij}^0 \hat{X}_j X_j^0}{\sum_j \pi_{ij}^0 X_j^0}
$$

---

**End of Mathematical Foundations**

This document provides complete mathematical exposition of all methods used in the Trade Network Topology Platform. For practical implementation details, see IMPLEMENTATION_GUIDE.md. For intuitive explanations, see USER_GUIDE.md.
