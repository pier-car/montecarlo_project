#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "nephele/concentration_grid.hpp"
#include "nephele/ensemble_kalman.hpp"
#include "nephele/lagrangian_engine.hpp"
#include "nephele/meteo_field.hpp"

namespace py = pybind11;
using namespace nephele;

namespace {

// Zero-copy view of a contiguous std::vector<double> owned by the engine.
py::array_t<double> as_readonly_array(const std::vector<double>& v,
                                      py::object owner) {
    return py::array_t<double>({static_cast<py::ssize_t>(v.size())},
                               {static_cast<py::ssize_t>(sizeof(double))}, v.data(),
                               std::move(owner));
}

}  // namespace

PYBIND11_MODULE(_core, mod) {
    mod.doc() = "NEPHELE C++ compute core (Lagrangian dispersion + EnKF).";

    py::class_<MeteoField, std::shared_ptr<MeteoField>>(mod, "MeteoField");

    py::class_<UniformMeteoField, MeteoField, std::shared_ptr<UniformMeteoField>>(
        mod, "UniformMeteoField")
        .def(py::init<std::array<double, 3>, std::array<double, 3>, double, double>(),
             py::arg("mean_wind"), py::arg("sigma2"), py::arg("epsilon"),
             py::arg("ground") = 0.0);

    py::class_<EngineConfig>(mod, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("num_particles", &EngineConfig::num_particles)
        .def_readwrite("dt", &EngineConfig::dt)
        .def_readwrite("c0", &EngineConfig::c0)
        .def_readwrite("decay_rate", &EngineConfig::decay_rate)
        .def_readwrite("source_pos", &EngineConfig::source_pos)
        .def_readwrite("source_mass", &EngineConfig::source_mass)
        .def_readwrite("seed", &EngineConfig::seed);

    py::class_<LagrangianDispersionEngine>(mod, "LagrangianDispersionEngine")
        .def(py::init<EngineConfig, std::shared_ptr<const MeteoField>>(),
             py::arg("config"), py::arg("meteo"))
        .def("step", &LagrangianDispersionEngine::step)
        .def("advance", &LagrangianDispersionEngine::advance, py::arg("n"))
        .def("concentration", &LagrangianDispersionEngine::concentration,
             py::arg("origin"), py::arg("cell_size"), py::arg("dims"))
        .def_property_readonly("time", &LagrangianDispersionEngine::time)
        .def("__len__", &LagrangianDispersionEngine::size)
        .def_property_readonly(
            "x", [](py::object self) {
                return as_readonly_array(self.cast<LagrangianDispersionEngine&>().x(),
                                         self);
            })
        .def_property_readonly(
            "y", [](py::object self) {
                return as_readonly_array(self.cast<LagrangianDispersionEngine&>().y(),
                                         self);
            })
        .def_property_readonly(
            "z", [](py::object self) {
                return as_readonly_array(self.cast<LagrangianDispersionEngine&>().z(),
                                         self);
            })
        .def_property_readonly("mass", [](py::object self) {
            return as_readonly_array(self.cast<LagrangianDispersionEngine&>().mass(),
                                     self);
        });

    py::class_<EnsembleKalmanFilter>(mod, "EnsembleKalmanFilter")
        .def(py::init<std::size_t, std::size_t>(), py::arg("state_dim"),
             py::arg("ensemble_size"))
        .def("set_ensemble", &EnsembleKalmanFilter::set_ensemble, py::arg("members"))
        .def("update", &EnsembleKalmanFilter::update, py::arg("obs_operator"),
             py::arg("observations"), py::arg("obs_var"), py::arg("seed") = 0xA11CEULL)
        .def("mean", &EnsembleKalmanFilter::mean)
        .def_property_readonly("ensemble", &EnsembleKalmanFilter::ensemble)
        .def_property_readonly("state_dim", &EnsembleKalmanFilter::state_dim)
        .def_property_readonly("ensemble_size", &EnsembleKalmanFilter::ensemble_size);
}
