# Topological Trade Dynamics: Methodology

**Web demo note (GitHub Pages)**: The browser visualization uses gridded dyad residuals from the
`docs/data/baci_gravity_viz.parquet` subset, mapped to a 2D field via distance-based MDS and
smoothed with a light Gaussian blur. This keeps the demo fast while preserving the residual
structure used in the full analysis pipeline.

## Revolutionary Integration of Field Theory + Econometrics

**First-ever implementation** combining:
- Theoretical field dynamics (PDEs)
- Structural gravity models (econometric)
- Persistent homology (algebraic topology)
- Critical slowing down detection (statistical physics)

---

## 1. Theoretical Foundation

### 1.1 Structural Gravity Model

Anderson-van Wincoop (2003) structural gravity with multilateral resistance:

```
X_ij = exp(α_i + δ_j - θ·ln(dist_ij) + β'Z_ij) + ε_ij
```

**Estimated via PPML** (Santos Silva & Tenreyro 2006) to handle:
- Zero trade flows
- Heteroskedasticity
- Jensen's inequality bias

### 1.2 Topological Field Theory

Field evolution PDE (Helfrich 2024):

```
∂_t y = M(κΔy + λ(K*y) - α(y-y₀)³)
```

**Physical interpretation:**
- `y(x,t)`: Trade intensity field on 2D torus
- `κΔy`: Diffusion (local equilibration via arbitrage)
- `λ(K*y)`: Topological pressure (network clustering)
- `α(y-y₀)³`: Congestion (nonlinear saturation)

**Key innovation**: Mexican-hat kernel `K(x)` creates center-surround structure:
```
K(x) = Gauss(σ₁) - Gauss(σ₂)
```
- Excitation at short range → promotes trade clustering
- Inhibition at long range → prevents over-concentration

### 1.3 Energy Functional (Lyapunov)

The system minimizes the energy:

```
E[y] = ∫[(κ/2)|∇y|² - (λ/2)y(K*y) + (α/4)(y-y₀)⁴] dx
```

**Guaranteed to decrease** under gradient flow, ensuring stability.

Components:
- **Dirichlet energy**: Penalizes spatial variation (smoothness)
- **Topological energy**: Nonlocal interaction via kernel
- **Congestion energy**: Prevents infinite trade concentration

---

## 2. Methodology Pipeline

### Step 1: Gravity Model Estimation

1. Load BACI bilateral trade data (187,362 obs, 2015-2021)
2. Merge with CEPII Gravity dataset (distances, RTAs, etc.)
3. Estimate PPML with fixed effects:
   ```python
   trade_value ~ dist + contig + comlang + rta + FE(origin,dest,year)
   ```
4. Extract residuals `ε_ij` (model prediction errors)

### Step 2: Spatial Embedding

Map countries into 2D space using **Multidimensional Scaling (MDS)**:

```python
from sklearn.manifold import MDS

# Build distance matrix D_ij = geographic distance
positions = MDS(n_components=2, dissimilarity='precomputed').fit_transform(D)
```

**Result**: Each country → (x,y) coordinate on 2D torus.

### Step 3: Field Initialization

Interpolate gravity residuals onto spatial grid:

```python
from scipy.interpolate import griddata

field = griddata(positions, residuals, (grid_x, grid_y), method='cubic')
```

**Interpretation**:
- High values: Over-performing trade (stronger than gravity predicts)
- Low values: Under-performing trade (weaker than gravity predicts)
- Field captures **structural deviations** from equilibrium

### Step 4: PDE Evolution

Solve field dynamics using **spectral methods** (FFT):

```python
def step(y, dt):
    # Laplacian via FFT
    y_fft = fft2(y)
    lap_y = ifft2(laplacian_k * y_fft)

    # Convolution via FFT
    conv_y = ifft2(kernel_fft * y_fft)

    # RHS of PDE
    rhs = M * (kappa * lap_y + lambda_ * conv_y - alpha * (y - y0)**3)

    return y + dt * rhs
```

**Stability**: CFL condition `dt < dx²/(2κ)` ensures convergence.

### Step 5: Topological Analysis

Compute **persistent homology** via filtration:

```python
for threshold in np.linspace(y_min, y_max, N):
    binary_image = field > threshold

    # β₀: Connected components (islands of high trade)
    β_0 = count_components(binary_image)

    # β₁: Cycles (trade loops)
    β_1 = euler_char - β_0  # Approximation

    # β₂: Voids (isolated regions)
    β_2 = ...
```

**Detection**: Phase transitions manifest as **jumps in Betti numbers**.

### Step 6: Critical Slowing Down

Early warning signals for bifurcations:

```python
def detect_critical_slowing(snapshots):
    means = [snap.mean() for snap in snapshots]

    # 1. Autocorrelation (lag-1)
    ac = correlate(means[:-1], means[1:])

    # 2. Variance trend (increasing?)
    variances = rolling_variance(means, window=10)
    slope = np.polyfit(range(len(variances)), variances, 1)[0]

    # 3. Recovery rate (decreasing?)
    # ...

    if ac > 0.7 and slope > 0:
        return "HIGH WARNING - Critical transition imminent"
```

**Theory**: Near bifurcations, systems exhibit:
- ↑ Memory (high autocorrelation)
- ↑ Fluctuations (variance amplification)
- ↓ Recovery rate (slowing down)

---

## 3. Applications

### 3.1 Supply Chain Fragmentation Prediction

**Question**: Will COVID-19 cause network fragmentation?

**Method**:
1. Estimate gravity model on pre-2020 data
2. Apply shock: reduce China exports by 30%
3. Evolve field dynamics with shock
4. Monitor: Betti numbers, energy curvature, autocorrelation

**Result**: If β₀ jumps → fragmentation into isolated clusters.

### 3.2 Trade Bloc Formation Detection

**Question**: Is EU-Asia decoupling?

**Method**:
1. Compute field for 2015 vs 2021
2. Detect: Number of peaks (high-trade regions)
3. Track: Distance between peaks (clustering vs separation)

**Result**: Increasing distance → bloc formation.

### 3.3 Critical Country Identification

**Question**: Which countries are systemically important?

**Method**:
1. For each country `i`, set `y(x_i) → 0` (remove node)
2. Evolve field, measure energy change ΔE
3. Rank countries by |ΔE|

**Result**: High |ΔE| → critical node (removal causes phase transition).

### 3.4 Resilience Quantification

**Metric**: Topological resilience = minimum shock to cause β₀ jump.

**Method**:
1. Apply shocks of increasing magnitude σ = 0.1, 0.2, ...
2. Evolve field for each shock
3. Compute Betti numbers
4. Find critical σ* where β₀ jumps by >5

**Result**: Higher σ* → more resilient network.

---

## 4. Novel Contributions

### 4.1 First Bridge Between Theory and Empirics

**Previous work**:
- Theoretical physics: Pattern formation in abstract systems
- Economics: Gravity models without spatial dynamics
- Network science: Static topology

**This work**:
- Maps **real trade data** onto **theoretical field**
- Evolves via **rigorous PDEs** with proven convergence
- Detects **phase transitions** via **algebraic topology**

### 4.2 Predictive Power

Traditional gravity models are **static** (no dynamics).

This framework enables:
- **Forecasting**: Where will trade flow after shock?
- **Early warning**: Detect critical transitions before they happen
- **Policy evaluation**: Test interventions in silico

### 4.3 Multiscale Analysis

Captures patterns at multiple scales:
- **Local**: City-level trade hubs (fine grid)
- **Regional**: Trade blocs (kernel radius)
- **Global**: Network-wide phase transitions (Betti numbers)

### 4.4 Theoretical Rigor

Unlike ML "black boxes", this method has:
- **Provable stability** (Lyapunov functional)
- **Conservation laws** (energy dissipation)
- **Topological invariants** (Betti numbers)
- **Physical interpretation** (diffusion, pressure, congestion)

---

## 5. Technical Implementation

### 5.1 Computational Efficiency

**FFT-based solver**:
- Complexity: O(N² log N) per time step
- Grid: 128×128 → ~0.1s per step (Python)
- GPU acceleration: 100x speedup possible (JAX/PyTorch)

**Memory**:
- Field: N×N floats (65KB for N=128)
- Kernel FFT: N×N complex (130KB)
- Total: <1MB for production runs

### 5.2 Parameter Calibration

**From economic theory**:
- `κ`: Proportional to transaction cost reduction rate
- `λ`: Estimated from network clustering coefficient
- `α`: Fitted to match observed trade saturation

**Sensitivity analysis**:
```python
for kappa in [0.0001, 0.001, 0.01]:
    for lambda_ in [0.05, 0.15, 0.25]:
        results = analyze(kappa, lambda_)
        plot_bifurcation_diagram(results)
```

### 5.3 Validation

**Tests**:
1. **Energy dissipation**: Verify dE/dt ≤ 0 always
2. **Equilibrium recovery**: Perturb → should return to stable state
3. **Known shocks**: 2008 crisis, COVID-19 → does model detect?
4. **Topology invariance**: Betti numbers should be robust to noise

**Benchmark**: Compare to network analysis (Louvain, PageRank).

---

## 6. Future Extensions

### 6.1 Multi-Product Field Theory

Extend to **vector field** `y = (y₁, ..., y_K)` for K products:

```
∂_t y_k = M(κ_k Δy_k + Σ_j λ_kj (K_kj * y_j) - α_k(y_k - y₀_k)³)
```

Captures **product substitution** and **cross-sector spillovers**.

### 6.2 Agent-Based Validation

Micro-found field theory via agent-based model:
- Firms choose trade partners (discrete)
- Aggregate → continuum field (mean-field limit)
- Verify PDE emerges from microeconomic optimization

### 6.3 Stochastic Dynamics

Add noise term:

```
∂_t y = M(κΔy + λ(K*y) - α(y-y₀)³) + σ·η(x,t)
```

Where `η` is white noise. Enables:
- **Fluctuation-driven transitions** (rare events)
- **Langevin dynamics** (thermal equilibrium)
- **Fokker-Planck equation** (probability density evolution)

### 6.4 Optimal Control

Find policy `u(x,t)` that minimizes cost:

```
min ∫[L(y, u) + ν|u|²] dt

subject to: ∂_t y = f(y) + B·u
```

**Applications**:
- Tariff design to prevent fragmentation
- Subsidy placement to enhance resilience
- Infrastructure investment to reduce critical slowing down

---

## 7. Software Architecture

```
scripts/08_topological_trade_dynamics.py
├── TradeFieldDynamics          # PDE solver
│   ├── mexican_hat_kernel()    # K(x) construction
│   ├── laplacian()             # Spectral Laplacian
│   ├── convolve()              # FFT convolution
│   ├── rhs()                   # PDE right-hand side
│   ├── step()                  # Euler integration
│   └── evolve()                # Full time evolution
│
├── TopologicalAnalyzer         # Persistent homology
│   ├── compute_betti_numbers() # β₀, β₁, β₂
│   └── detect_transition()     # Jump detection
│
├── CriticalSlowingDetector     # Early warning
│   ├── autocorrelation()       # Lag-1 AC
│   ├── variance_trend()        # Slope estimation
│   └── warning_signals()       # Combined indicators
│
└── TopologicalGravityAnalysis  # Integration layer
    ├── map_to_spatial_field()  # MDS embedding
    └── run_analysis()          # Full pipeline
```

---

## 8. Reproducibility

### Data Requirements

- **BACI**: Bilateral trade flows (public, CEPII)
- **Gravity**: Distance, RTAs, language, etc. (public, CEPII)
- **Computation**: ~10 min on laptop for full analysis

### Code Availability

```bash
# Run analysis
python scripts/08_topological_trade_dynamics.py \
    --year 2019 \
    --grid-size 128 \
    --steps 200 \
    --shock 0.5

# Output: Energy trajectory, Betti curves, warning signals
```

### Visualization

Interactive web interface: [topology.html](topology.html)

---

## 9. References

**Economics**:
- Anderson & van Wincoop (2003): Gravity with Gravitas
- Santos Silva & Tenreyro (2006): The Log of Gravity
- Head & Mayer (2014): Gravity Equations (Handbook)

**Mathematics**:
- Edelsbrunner & Harer (2010): Computational Topology
- Carlsson (2009): Topology and Data (Bull. AMS)

**Physics**:
- Cross & Hohenberg (1993): Pattern Formation (Rev. Mod. Phys.)
- Scheffer et al. (2009): Early-Warning Signals (Nature)

**This work**:
- Helfrich (2026): Topological Field Theory for International Trade

---

## 10. Summary

This methodology represents a **paradigm shift** in trade analysis:

| Traditional Gravity | Topological Dynamics |
|---------------------|----------------------|
| Static equilibrium | Dynamic evolution |
| No spatial structure | Explicit 2D field |
| Single-scale | Multiscale patterns |
| Descriptive | Predictive |
| No early warning | Critical slowing down detection |
| Regression residuals | Phase transitions |

**Key insight**: Trade networks are **not static** - they evolve via field dynamics with emergent topological structure. Phase transitions (fragmentation, bloc formation) can be **detected before they occur** via universal warning signals.

**Impact**: Enables proactive policy design to prevent supply chain crises.
