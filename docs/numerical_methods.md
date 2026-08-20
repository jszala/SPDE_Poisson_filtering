# Numerical methods and calibration

## State equation

The state is stored as two float64 arrays with trailing dimensions `(nx, ny)`.
For batched propagation, the leading dimension is the particle index. V2 uses
the stochastic FitzHugh–Nagumo system

$$
du = [D_u\Delta u + s(u-\alpha_1)(u-\alpha_2)(\alpha_3-u)-v+p]dt
     + \sigma_u dW_u,
$$

$$
dv = [D_v\Delta v + \gamma(\beta u-v)]dt + \sigma_v dW_v.
$$

The profile fixes $\alpha=(0.5,0.75,1)$ and $\beta=10$. The tuned actin-wave
parameters are $D_u=0.02$, $D_v=0.1$, $s=40$, $p=0.1$, $\gamma=0.02$, and
$\sigma_u=\sigma_v=0.002$. Both profiles use $\Delta x=0.1$ and
$\Delta t=0.002$.

## Spatial and temporal discretization

The five-point finite-volume Laplacian is symmetric and implements zero normal
flux by giving a boundary cell one neighbor contribution in the outward
coordinate rather than two. Consequently, constants are in its null space and
the discrete operator conserves spatial mass.

With $L$ denoting that Laplacian, one Euler–Maruyama step is

$$
(I-\Delta t D_uL)u^{n+1}
=u^n+\Delta t\,f(u^n,v^n)+\sigma_u\frac{\sqrt{\Delta t}}{\Delta x}\xi_u^n,
$$

$$
(I-\Delta t D_vL)v^{n+1}
=v^n+\Delta t\,g(u^n,v^n)+\sigma_v\frac{\sqrt{\Delta t}}{\Delta x}\xi_v^n.
$$

The two sparse matrices are factorized once with SuperLU and reused. All
reaction and coupling terms are explicit; notably, the inhibitor update uses
`u[n]`, not the freshly computed `u[n+1]`. The factor
$\sqrt{\Delta t}/\Delta x$ is the two-dimensional finite-volume scaling for
space-time white noise. Non-finite results raise `FloatingPointError`; neither
field is clamped.

## Seeded warm-up and actin-wave preset

The initial state is a seeded Gaussian random field smoothed with a small
reflecting Gaussian kernel and centered near the upper excitable state. It is
then advanced with exactly the same stochastic solver used after observations
begin. Warm-up stops only after its minimum duration and when the activator has
an admissible active fraction, adequate 5–95% spatial contrast, and 0.5–99.5%
quantiles inside `[-0.1, 1.1]`.

For `configs/quick.yaml` and master seed `20260819`, warm-up stops at step 100.
The fixed diagnostic regression values are approximately:

| Diagnostic | Value |
|---|---:|
| Active fraction | 0.3682 |
| Warm-up spatial contrast | 0.3592 |
| Trajectory 0.5% quantile | 0.5061 |
| Trajectory 99.5% quantile | 0.9542 |
| Median trajectory spatial contrast | 0.3692 |
| RMS one-step temporal change | 0.000950 |

Thus more than 99% of activator samples are within the target envelope, while
spatial contrast and nonzero movement persist. The wider 64×64 profile uses the
same physical parameters and seed with a longer minimum warm-up. These are
calibration criteria, not signal clipping or post-selection of a simulated
trajectory.

## Observation model and aggregation

The intensity density is

$$
\lambda(t,x)=\operatorname{clip}
\left(e^{-at}(c\max(u(t,x),0))^2,\lambda_{\min},\lambda_{\max}\right).
$$

On the equal square grid, the cell rate is approximated by
$r_i=\lambda(t,x_i)\Delta x^2$ and the next count increment has distribution
$Y_i\sim\operatorname{Poisson}(\Delta t\,r_i)$. The complete fine count cube is
sampled once. Rates and counts at resolution `m` are sums over disjoint
`(nx/m) × (ny/m)` blocks, preserving both Poisson additivity and the total event
count at every timestep.

## Likelihood and filter

For count $y_i$ and event rate $r_i$, the full log likelihood is

$$
\ell(r;y)=\sum_i[y_i\log(\Delta t r_i)-\Delta t r_i-\log(y_i!)].
$$

The filter uses the equivalent reference-measure increment

$$
\Delta\ell(r;y)=\sum_i
[y_i\log(r_i/r_{0,i})-\Delta t(r_i-r_{0,i})].
$$

Their difference is independent of the particle. The standalone likelihood
module exposes both formulas so it can be extracted into a future likelihood
publication without changing formulas, axes, or tests.

At each observation time, particles are propagated, the relative increments
are added to float64 log weights, weights are normalized with log-sum-exp, and
$\operatorname{ESS}=1/\sum_l(w^l)^2$ is recorded. When ESS is below $L/2$,
corrected residual resampling makes `floor(L*w)` deterministic copies, samples
the residual mass, and resets weights uniformly. Posterior means and variances
are computed before resampling. Reported skill is compared with a seeded,
unobserved stochastic trajectory drawn from the same initial prior; this avoids
giving the open-loop forecast the synthetic truth's exact initial condition.

## Local parallel execution

Only propagation is delegated to a persistent `ProcessPoolExecutor`. Particles
are divided into contiguous chunks. The worker initializer constructs its
Laplacian and LU factorizations once, results are collected in submission order,
and resolution experiments remain sequential. Every stochastic draw derives
from a `SeedSequence` keyed by `(master_seed, stage, timestep, particle_index)`;
worker identity and scheduling do not affect the result.
