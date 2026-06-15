cmake_minimum_required(VERSION 3.18)
project(NEPHELE VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NEPHELE_USE_OPENMP "Enable OpenMP parallelism in the compute core" ON)
option(NEPHELE_USE_CUDA   "Build the optional CUDA advection kernel"       OFF)
option(NEPHELE_BUILD_TESTS  "Build the C++ test suite"                     ON)
option(NEPHELE_BUILD_PYTHON "Build the pybind11 Python extension module"   OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compute core library
# ---------------------------------------------------------------------------
add_library(nephele_core
  core/src/lagrangian_engine.cpp
  core/src/ensemble_kalman.cpp
  core/src/concentration_grid.cpp
  core/src/meteo_field.cpp
)
target_include_directories(nephele_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_options(nephele_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

if(NEPHELE_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nephele_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(NEPHELE_USE_CUDA)
  enable_language(CUDA)
  target_sources(nephele_core PRIVATE core/cuda/advect_kernel.cu)
  set_target_properties(nephele_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---------------------------------------------------------------------------
# Python extension (built by scikit-build-core via pyproject.toml)
# ---------------------------------------------------------------------------
if(NEPHELE_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core bindings/pybind_module.cpp)
  target_link_libraries(_core PRIVATE nephele_core)
  install(TARGETS _core LIBRARY DESTINATION nephele)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(NEPHELE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
