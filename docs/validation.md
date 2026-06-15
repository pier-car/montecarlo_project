# NEPHELE — Validation

NEPHELE is verified at three levels.

## 1. Analytic well-mixed invariant

In homogeneous Gaussian turbulence the turbulent-velocity variance of the
ensemble must remain equal to $\sigma^2$ for all time (Thomson's well-mixed
condition). This is checked statistically:

- C++: `tests/cpp/test_wellmixed.cpp`
- Julia oracle: `julia/WellMixed.jl` (`well_mixed_residual < 0.05`)

## 2. Differential testing against the Julia oracle

The Julia reference integrator in `julia/WellMixed.jl` implements the same
generalized Langevin equation independently of the C++ core. Matching ensemble
statistics (variance, mean displacement) within Monte-Carlo error confirms the
optimised port is faithful.

## 3. Conservation and physical sanity

- **Mass conservation:** with zero decay, the total binned mass equals the
  released mass (`tests/cpp/test_grid.cpp`, `tests/python/test_reference_engine.py`).
- **Mean advection:** the ensemble centroid tracks $U\,t$ within sampling error.
- **Reflecting boundary:** no particle penetrates below the ground surface.
- **Decay law:** total airborne mass follows $e^{-\lambda t}$.

## 4. Assimilation behaviour

- **EnKF:** a scalar observation pulls the ensemble mean toward the measurement,
  bounded by the prior/observation precision ratio
  (`tests/python/test_enkf.py`, `tests/cpp/test_enkf.cpp`).
- **Particle filter:** the weighted estimate converges to the true source under
  repeated informative updates (`tests/python/test_particle_filter.py`).
- **Sensor placement:** the lazy-greedy selector picks one site per correlated
  cluster and yields monotone, non-negative information gains
  (`tests/python/test_sensor_placement.py`).

## Reproducing locally

```bash
# C++
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# Python
PYTHONPATH=api pytest tests/python -q

# Julia oracle
julia julia/WellMixed.jl
```

## Future field-trial validation

The roadmap includes quantitative comparison against published wind-tunnel
street-canyon datasets and tracer field trials, scored with the standard
fractional-bias and figure-of-merit-in-space (FMS) metrics.
