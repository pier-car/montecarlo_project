# NEPHELE — Theory

This document derives the equations implemented in the NEPHELE compute core.

## 1. Lagrangian stochastic dispersion

Each marker particle carries a position $\mathbf{x}(t)$ and a turbulent velocity
fluctuation $\mathbf{u}(t)$. Transport obeys the generalized **Langevin
equation**

$$
dx_i = \big( U_i(\mathbf{x}) + u_i \big)\, dt,
\qquad
du_i = a_i(\mathbf{x}, \mathbf{u})\, dt + \sqrt{C_0\,\varepsilon}\; dW_i ,
$$

where $U_i$ is the resolved mean wind, $\varepsilon$ the turbulent-kinetic-energy
dissipation rate, $C_0 \approx 3\text{–}6$ the universal Lagrangian constant, and
$dW_i$ are increments of a Wiener process with
$\langle dW_i\, dW_j \rangle = \delta_{ij}\, dt$.

### 1.1 Thomson's well-mixed criterion

The drift $a_i$ is not free: Thomson (1987) showed it is fixed by the
**well-mixed condition** — if particles are initially distributed according to
the ambient turbulence PDF $p_a(\mathbf{x}, \mathbf{u})$, they must remain so for
all time. For inhomogeneous Gaussian turbulence with position-dependent
variances $\sigma_i^2(\mathbf{x})$ this yields, in the separable case,

$$
a_i = -\frac{u_i}{T_{L,i}(\mathbf{x})}
      + \frac{1}{2}\left(1 + \frac{u_i^2}{\sigma_i^2}\right)
        \frac{\partial \sigma_i^2}{\partial x_i},
\qquad
T_{L,i} = \frac{2\,\sigma_i^2}{C_0\,\varepsilon}.
$$

The first term is an **Ornstein–Uhlenbeck** relaxation over the Lagrangian
timescale $T_{L,i}$; the second is the drift correction that guarantees the
well-mixed condition in non-uniform turbulence. In a homogeneous field the
inhomogeneity term vanishes and the velocity reduces to a stationary OU process
with variance $\sigma_i^2$ — the invariant verified in the test suite.

## 2. From particles to concentration

The mean concentration field is recovered by ensemble binning over a Eulerian
grid of cell volume $V_c$:

$$
C(\mathbf{x}_c, t) = \frac{q}{N_p\, V_c}\sum_{p=1}^{N_p}
  m_p(t)\,\mathbb{1}\!\left[\mathbf{x}_p(t) \in \text{cell } c\right],
\qquad
m_p(t) = e^{-\lambda t},
$$

with $q$ the released mass/activity and $\lambda$ the decay/deposition rate.
Ground and building faces are treated as **reflecting boundaries**
($u_n \to -u_n$).

## 3. Data assimilation (Ensemble Kalman Filter)

For an ensemble $\{\mathbf{X}^{(k)}\}_{k=1}^{N_e}$ and observation operator $H$,
the stochastic EnKF analysis update is

$$
\mathbf{X}^{(k)}_a = \mathbf{X}^{(k)}_f
  + \mathbf{K}\big(\mathbf{y} + \boldsymbol{\eta}^{(k)} - H\,\mathbf{X}^{(k)}_f\big),
\qquad
\mathbf{K} = \mathbf{P}_f H^\top\!\left(H \mathbf{P}_f H^\top + \mathbf{R}\right)^{-1},
$$

with $\mathbf{P}_f$ the ensemble forecast covariance, $\mathbf{R}$ the
sensor-noise covariance, and $\boldsymbol{\eta}^{(k)} \sim \mathcal{N}(0, \mathbf{R})$
perturbed observations. NEPHELE forms $\mathbf{P}_f$ directly from ensemble
anomalies and solves the innovation system by Cholesky factorisation.

## 4. Information-optimal sensor placement

Given candidate sites $\mathcal{V}$ and budget $B$, choose
$\mathcal{A} \subseteq \mathcal{V}$, $|\mathcal{A}| \le B$, maximizing the mutual
information between observed and unobserved locations:

$$
\mathcal{A}^\star = \arg\max_{|\mathcal{A}| \le B}\;
   I(\mathcal{A};\, \mathcal{V} \setminus \mathcal{A}).
$$

$I(\cdot)$ is **monotone submodular**, so the **lazy-greedy** algorithm attains
the $(1 - 1/e) \approx 0.63$ approximation guarantee. The marginal gain of adding
site $i$ is

$$
\Delta_i(\mathcal{A}) = \frac{1}{2}\log
   \frac{\operatorname{Var}(y_i \mid y_\mathcal{A})}
        {\operatorname{Var}(y_i \mid y_{\mathcal{V}\setminus(\mathcal{A}\cup\{i\})})},
$$

with each conditional variance evaluated as a Schur complement of the belief
covariance.

## References

- D. J. Thomson, "Criteria for the selection of stochastic models of particle
  trajectories in turbulent flows," *J. Fluid Mech.* **180**, 529–556 (1987).
- G. Evensen, "The Ensemble Kalman Filter," *Ocean Dynamics* **53** (2003).
- A. Krause, A. Singh, C. Guestrin, "Near-Optimal Sensor Placements in Gaussian
  Processes," *JMLR* **9** (2008).
