<div align="center">

# NEPHELE
### Networked Ensemble Plume-Hazard Estimation & Lagrangian Engine

**Real-time Lagrangian stochastic CBRN dispersion modelling, online data assimilation, and information-optimal sensor tasking for dense urban environments.**

[![CI (C++)](https://github.com/pier-car/montecarlo_project/actions/workflows/ci-cpp.yml/badge.svg)](.github/workflows/ci-cpp.yml)
[![CI (Python)](https://github.com/pier-car/montecarlo_project/actions/workflows/ci-python.yml/badge.svg)](.github/workflows/ci-python.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)

</div>

---

## 1. Overview

**NEPHELE** forecasts the dispersion of chemical, biological, radiological, and
nuclear (CBRN) agents through urban canopies in **faster-than-real-time**, and
tells you **where to move your sensors next**.

Legacy hazard-prediction tools rely on **Gaussian plume** approximations that
assume flat terrain and homogeneous turbulence — assumptions that collapse
inside a street canyon. Full CFD is accurate but far too slow for a tactical
decision loop. NEPHELE occupies the missing middle: a **Lagrangian Stochastic
Model (LSM)** that captures canyon channelling and turbulent intermittency at
Monte-Carlo cost, wrapped in an **Ensemble Kalman / particle-filter**
assimilation loop and an **information-theoretic sensor-placement** optimizer.

| Capability | NEPHELE | Gaussian plume | CFD (LES) |
|---|:--:|:--:|:--:|
| Urban canyon channelling | ✅ | ❌ | ✅ |
| Faster-than-real-time | ✅ | ✅ | ❌ |
| Online sensor assimilation | ✅ | ❌ | ⚠️ |
| Optimal sensor tasking | ✅ | ❌ | ❌ |
| Runs at the edge (no cluster) | ✅ | ✅ | ❌ |

---

## 2. Theory

### 2.1 Lagrangian stochastic dispersion

Each of the $N_p$ marker particles carries a position $\mathbf{x}(t)$ and a
turbulent velocity $\mathbf{u}(t)$. Transport is governed by the generalized
**Langevin equation**

$$
dx_i = \big( U_i(\mathbf{x}) + u_i \big)\, dt,
\qquad
du_i = a_i(\mathbf{x}, \mathbf{u})\, dt + \sqrt{C_0\,\varepsilon}\; dW_i ,
$$

where $U_i$ is the resolved mean wind, $\varepsilon$ the turbulent-kinetic-energy
dissipation rate, $C_0 \approx 3\text{–}6$ the universal Lagrangian constant, and
$dW_i$ increments of a Wiener process with
$\langle dW_i\, dW_j\rangle = \delta_{ij}\,dt$.

The drift $a_i$ is fixed by **Thomson's (1987) well-mixed criterion**: if the
particles are initially distributed according to the ambient turbulence PDF, they
must remain so. For inhomogeneous Gaussian turbulence with position-dependent
variances $\sigma_i^2(\mathbf{x})$ this yields, in the separable case,

$$
a_i = -\frac{u_i}{T_{L,i}(\mathbf{x})}
      + \frac{1}{2}\left(1 + \frac{u_i^2}{\sigma_i^2}\right)
        \frac{\partial \sigma_i^2}{\partial x_i},
\qquad
T_{L,i} = \frac{2\,\sigma_i^2}{C_0\,\varepsilon}.
$$

The first term is an **Ornstein–Uhlenbeck** relaxation over the Lagrangian
timescale $T_{L,i}$; the second guarantees the well-mixed condition in non-uniform
turbulence.

### 2.2 From particles to concentration

The mean concentration field is recovered by ensemble binning over a Eulerian
grid of cell volume $V_c$:

$$
C(\mathbf{x}_c, t) = \frac{q}{N_p\, V_c}\sum_{p=1}^{N_p}
  m_p(t)\,\mathbb{1}\!\left[\mathbf{x}_p(t) \in \text{cell } c\right],
\qquad
m_p(t) = e^{-\lambda t},
$$

where $q$ is the released mass/activity and $\lambda$ the decay/deposition rate.
Ground and building faces are reflecting boundaries ($u_n \to -u_n$).

### 2.3 Data assimilation (Ensemble Kalman Filter)

For an ensemble $\{\mathbf{X}^{(k)}\}_{k=1}^{N_e}$ and observation operator $H$,
the stochastic EnKF analysis update is

$$
\mathbf{X}^{(k)}_a = \mathbf{X}^{(k)}_f
  + \mathbf{K}\big(\mathbf{y} + \boldsymbol{\eta}^{(k)} - H\,\mathbf{X}^{(k)}_f\big),
\qquad
\mathbf{K} = \mathbf{P}_f H^\top\!\left(H \mathbf{P}_f H^\top + \mathbf{R}\right)^{-1},
$$

with $\mathbf{P}_f$ the ensemble forecast covariance, $\mathbf{R}$ the sensor-noise
covariance, and $\boldsymbol{\eta}^{(k)}\sim\mathcal{N}(0,\mathbf{R})$ perturbed
observations.

### 2.4 Information-optimal sensor placement

Given candidate sites $\mathcal{V}$ and budget $B$, choose
$\mathcal{A}\subseteq\mathcal{V}$, $|\mathcal{A}|\le B$, maximizing the mutual
information between observed and unobserved locations:

$$
\mathcal{A}^\star = \arg\max_{|\mathcal{A}|\le B}\;
   I(\mathcal{A};\,\mathcal{V}\setminus\mathcal{A}).
$$

$I(\cdot)$ is **monotone submodular**, so the **lazy-greedy** algorithm attains
the $(1-1/e)\approx 0.63$ approximation guarantee while exploiting submodularity
for near-linear practical runtime.

Full derivations are in [`docs/theory.md`](docs/theory.md).

---

## 3. System architecture

```
                         ┌──────────────────────────────────────────┐
   QGIS / GDAL geodata   │              PYTHON / JULIA               │
   (DEM, footprints) ───►│  geo ingest │ assimilation │ optimization │
   met data              └───────┬───────────────┬──────────────┬────┘
                                 │ MeteoField     │ sensor counts│ tasking
                                 ▼                ▼              ▼
                         ┌──────────────────────────────────────────┐
                         │            C++17 COMPUTE CORE             │
                         │  Langevin integrator (OpenMP / CUDA)     │
                         │  concentration grid │ EnKF gain          │
                         └───────────────────┬──────────────────────┘
                                             │ C(x,t), hazard polygons
                                             ▼
                              FastAPI service ► GeoJSON / QGIS
```

| Layer | Language | Responsibility |
|---|---|---|
| Compute core | C++17 (OpenMP, optional CUDA) | Particle advection, grid binning, EnKF gain, reflecting boundaries. Zero per-step heap allocation. |
| Bindings | pybind11 | Zero-copy NumPy ↔ C++ buffer exchange (`nephele._core`). |
| Orchestration | Python 3.10+ | Config, QGIS/GDAL ingest, met pre-processing, assimilation driver, sensor optimizer, REST service. |
| Verification | Julia | Reference SDE integrator + well-mixed oracle for differential testing. |
| Presentation | Python (FastAPI) + GeoJSON/QGIS | Hazard polygons, sensor tasking. |

See [`docs/architecture.md`](docs/architecture.md) for the full component map.

---

## 4. Quick start

### 4.1 Build the C++ core

```bash
git clone https://github.com/pier-car/montecarlo_project.git
cd montecarlo_project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNEPHELE_USE_OPENMP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 4.2 Install the Python package

```bash
# Reproducible toolchain (GDAL/QGIS + build tools)
conda env create -f environment.yml
conda activate nephele

# Editable install (builds the pybind11 module via scikit-build-core)
pip install -e .[dev]
```

The pure-Python reference engine works without compiling the extension — handy
for quick experiments and CI:

```bash
pip install numpy
PYTHONPATH=api python -c "import nephele; print('native core:', nephele.has_native_core())"
```

### 4.3 Run the demo scenario

```bash
bash scripts/fetch_demo_data.sh
python scripts/run_scenario.py --config data/demo/release.yaml --horizon 120 \
    --output hazard.geojson
```

Visualise `hazard.geojson` over `data/demo/buildings.geojson` in QGIS, or serve
it live:

```bash
pip install -e .[service]
python scripts/run_scenario.py --config data/demo/release.yaml --serve
# POST a scenario to http://localhost:8000/hazard
```

### 4.4 Benchmark throughput

```bash
python scripts/benchmark_throughput.py --particles 100000 --steps 100
```

---

## 5. Repository layout

```
include/nephele/     Public C++ headers
core/src/            C++ compute core (engine, EnKF, grid, meteo)
core/cuda/           Optional CUDA advection kernel
bindings/            pybind11 module
api/nephele/         Python package (geo, assimilation, optimization, service)
julia/               Reference oracle for differential testing
scripts/             Scenario runner, benchmark, demo data
tests/cpp, tests/python   GoogleTest + pytest suites
docs/                Theory, architecture, validation
.github/workflows/   CI: C++, Python, wheels, docs
```

---

## 6. Validation

NEPHELE is verified against (i) the analytic **well-mixed** invariant, (ii) the
Julia reference SDE integrator (differential testing), and (iii) mass/advection
conservation laws. See [`docs/validation.md`](docs/validation.md).

```bash
ctest --test-dir build --output-on-failure   # C++
PYTHONPATH=api pytest tests/python -q          # Python
julia julia/WellMixed.jl                       # Julia oracle
```

---

## 7. License & citation

Released under the **Apache-2.0** license (see [`LICENSE`](LICENSE)). If you use
NEPHELE in academic work, please cite it via [`CITATION.cff`](CITATION.cff).

> **Scope note.** NEPHELE is an open-science civil-protection / hazard-modelling
> toolkit built on published atmospheric-dispersion and estimation theory, using
> only synthetic demo data.
