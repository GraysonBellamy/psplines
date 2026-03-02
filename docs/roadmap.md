# psplines Roadmap

Future additions based on Eilers & Marx (2021), *Practical Smoothing: The Joys of P-Splines*.

Reference sections (§) refer to `refs/joy_psplines.md`.

---

## Tier 1 — Core Foundations

These are small changes that unlock many downstream features.

### ~~Observation Weights~~ — Done

Implemented: optional `weights` parameter on `PSpline`, weighted normal equations
`(B'WB + λD'D)α = B'Wy`, propagated into ED, σ², SE, bootstrap, and all optimizers.
Zero weights give missing-data interpolation for free.

### HFS Automatic λ Selection

The Harville-Fellner-Schall algorithm estimates λ via the mixed model identity `λ = σ²/τ²`. Converges in ~7 iterations with no grid search or bounds required.

Algorithm (§3.4, p. 952–961):
1. Start with λ = 1
2. Solve `(B'B + λD'D)α = B'y`
3. Compute `ED = trace((B'B + λD'D)⁻¹ B'B)`
4. `τ² = ||Dα||² / (ED - d)` where d = penalty order
5. `σ² = ||y - Bα||² / (m - ED)`
6. Update `λ = σ² / τ²`, repeat until convergence

Can be accelerated with *regula falsi* root-finding on `f(λ) = log(λ) - log(σ²) + log(τ²)` (§3.4, p. 997).

**Reference**: §3.4, Appendix E.2

---

## ~~Tier 2 — GLM P-Splines~~ — Done

Implemented: full GLM P-spline framework via IRLS (Iteratively Reweighted Least Squares).

### ~~Poisson P-Splines~~ — Done

Implemented: `family="poisson"` parameter on `PSpline`. IRLS loop with log link (eq. 2.21–2.22),
working weights `W = diag(μ)`, working response `z = η + W⁻¹(y - μ)`, convergence control
via `max_iter` and `tol`. Supports exposure offsets via `offset` parameter for rate/hazard modeling.

### ~~Binomial P-Splines~~ — Done

Implemented: `family="binomial"` with logit link (eq. 2.23–2.25). Supports both Bernoulli (`trials=None`)
and grouped binomial (`trials` vector). Starting values `π₀ = (y+1)/(t+2)`. Fitted values bounded in [0, t].

### ~~GLM Standard Errors and ED~~ — Done

Implemented: upon IRLS convergence, ED computed with converged weights `Ŵ` (eq. 2.26–2.27).
`Cov(α̂) = φ(B'ŴB + λD'D)⁻¹` with `φ = 1` for Poisson/Binomial. SE bands on link scale via
delta method; response-scale CI via inverse-link transform (`predict(scale="response", return_se=True)`
returns `(mu_hat, lower, upper)`). User prior weights combine with IRLS weights per §2.12.3.

### ~~Density Estimation~~ — Done

Implemented: `density_estimate()` convenience function. Bins raw data into a histogram, fits a
Poisson P-spline on counts, selects λ via AIC, normalizes to a proper density. Conservation of
moments preserved via penalty order (`penalty_order=3` preserves mean + variance).

---

## Tier 3 — The Whittaker Smoother

A degenerate P-spline where `B = I` (identity). Extremely fast for evenly-spaced data — only requires solving `(I + λD'D)μ = y`. No knots, no degree, no segments to choose.

Applications: time series, spectra, life tables, signal processing.

Missing data via zero weights: `(W + λD'D)μ = Wy`.

**Reference**: §2.10

---

## Tier 4 — Multidimensional Smoothing

### 2D Tensor Product P-Splines

Basis: `B ⊗ B̌` (Kronecker product of marginal bases).

Row and column penalties (eq. 4.19–4.21):
```
Pen = λ||(Ǐ ⊗ D)α||² + λ̌||(Ď ⊗ I)α||²
```

Efficient array algorithms avoid forming the full Kronecker product (Appendix D). For gridded data, the normal equations can be solved via GLAM (Generalized Linear Array Models) using only marginal basis matrices.

**Reference**: §4.3–4.9, Appendix D

### Generalized Additive Models

Additive structure `μ = f₁(x₁) + f₂(x₂)` with block-diagonal penalty (eq. 4.2–4.4). Each component gets its own λ.

**Reference**: §4.1 eq. 4.1–4.4

---

## Tier 5 — Special Penalties and Bases

### Circular / Periodic Smoothing

Modified B-spline basis where boundary splines wrap around, plus a circular penalty matrix connecting the ends. Needed for directional data, time-of-day effects. Heavy smoothing approaches the von Mises distribution.

**Reference**: §8.3

### Harmonic Penalty

Replace standard second-order difference penalty with `Σ(α_{j-1} - 2ψα_j + α_{j+1})²` where `ψ = cos(2π/p)`. Interpolates gaps with sine/cosine of period p rather than polynomials.

**Reference**: §8.2 eq. 8.1–8.3

### ~~Shape Constraints (Monotonicity, Convexity)~~ — Done

Implemented: asymmetric penalty approach from §8.7 (eq. 8.14–8.15). Iterative fitting with
`V = diag(v)` where `v_j = I(constraint violated)`, scaled by large κ (default 1e8).
Supports all constraint types: `"increasing"`, `"decreasing"`, `"convex"`, `"concave"`, `"nonneg"`.
Selective domain constraints via `"domain": (lo, hi)`. Combined constraints via multiple specs.
Works with both Gaussian and GLM families. Flat-slope subdomain penalty via
`slope_zero=SlopeZeroConstraint(domain=(lo, hi))`.

**Reference**: §8.7

### ~~Adaptive / Variable Penalties~~ — Done

Implemented: two approaches from §8.8. (1) Exponential variable penalty via `penalty_gamma`
parameter: `v_j = exp(γ j/m)` with 2D grid search optimizer `variable_penalty_cv()`.
(2) Nonparametric adaptive penalty via `adaptive=True`: local roughness estimation with
secondary P-spline basis over index space, alternating optimisation of weights and coefficients.
Per-difference weights normalised to mean 1; global λ controls overall level.

**Reference**: §8.8

---

## Tier 6 — Specialized Applications

| Feature | Reference | Notes |
|---------|-----------|-------|
| Quantile smoothing | §5.1 | Asymmetric check loss + penalty |
| Expectile smoothing | §5.2 | Asymmetric squared loss + penalty |
| Varying coefficient models | §4.2 eq. 4.8–4.10 | Coefficients that vary with a covariate |
| Signal regression | §7 | High-dim coefficient function on spectra/signals |
| Composite links | §6 | Grouped/censored histogram data |
| Survival / hazard modeling | §8.9 | Poisson P-spline with exposure offsets |
