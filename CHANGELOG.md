# Changelog

All notable changes to NEPHELE are documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-15

### Added
- C++17 compute core: Lagrangian stochastic dispersion engine implementing the
  Thomson (1987) well-mixed generalized Langevin equation with OpenMP
  parallelism and an optional CUDA advection kernel.
- Stochastic Ensemble Kalman Filter (C++ and NumPy reference) with a
  Cholesky-based analysis update.
- Bootstrap particle filter with systematic resampling.
- Information-optimal sensor placement via lazy-greedy submodular mutual-
  information maximization.
- Geospatial ingestion: building-footprint rasterisation and Pasquill-stability
  turbulence closure.
- End-to-end assimilation driver and FastAPI hazard service.
- Julia reference oracle for differential testing of the well-mixed invariant.
- C++ (GoogleTest) and Python (pytest) test suites; CI workflows.
