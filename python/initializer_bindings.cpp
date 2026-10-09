// Backlog M4: explicit initializers in the NumPy bindings.
#include "kan/initializers.hpp"
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_initializers(py::module_& module) {
    py::native_enum<kan::Distribution>(module, "Distribution", "enum.Enum", "Initializer draw distribution")
        .value("UNIFORM", kan::Distribution::Uniform, "symmetric uniform")
        .value("NORMAL", kan::Distribution::Normal, "Gaussian (Marsaglia polar method)")
        .finalize();
    py::class_<kan::DenominatorInit>(module, "DenominatorInit",
                                     "Rational denominators with |S(z)| <= bound for |z| <= radius")
        .def(py::init([](double bound, double radius) { return kan::DenominatorInit{bound, radius}; }),
             py::arg("bound") = 0.5, py::arg("radius") = 1.0)
        .def_readwrite("bound", &kan::DenominatorInit::bound)
        .def_readwrite("radius", &kan::DenominatorInit::radius)
        .def(py::self == py::self)
        .def(py::self != py::self);
    py::class_<kan::VarianceScaling>(module, "VarianceScaling", "Variance-preserving initialization")
        .def(py::init([](double gain, kan::Distribution distribution, std::uint64_t seed, kan::DenominatorInit d) {
            return kan::VarianceScaling{gain, distribution, seed, d};
        }), py::arg("gain") = 1.0, py::arg("distribution") = kan::Distribution::Uniform, py::arg("seed") = 0,
            py::arg("denominators") = kan::DenominatorInit{})
        .def_readwrite("gain", &kan::VarianceScaling::gain)
        .def_readwrite("distribution", &kan::VarianceScaling::distribution)
        .def_readwrite("seed", &kan::VarianceScaling::seed)
        .def_readwrite("denominators", &kan::VarianceScaling::denominators)
        .def(py::self == py::self)
        .def(py::self != py::self);
    py::class_<kan::NoiseInit>(module, "NoiseInit", "pykan-style small coefficient noise")
        .def(py::init([](double scale, kan::Distribution distribution, std::uint64_t seed, kan::DenominatorInit d,
                         double residual_mean, double residual_spread) {
            return kan::NoiseInit{scale, distribution, seed, d, residual_mean, residual_spread};
        }), py::arg("scale") = 0.3, py::arg("distribution") = kan::Distribution::Uniform, py::arg("seed") = 0,
            py::arg("denominators") = kan::DenominatorInit{}, py::arg("residual_mean") = 0.0,
            py::arg("residual_spread") = 1.0)
        .def_readwrite("scale", &kan::NoiseInit::scale)
        .def_readwrite("distribution", &kan::NoiseInit::distribution)
        .def_readwrite("seed", &kan::NoiseInit::seed)
        .def_readwrite("denominators", &kan::NoiseInit::denominators)
        .def_readwrite("residual_mean", &kan::NoiseInit::residual_mean)
        .def_readwrite("residual_spread", &kan::NoiseInit::residual_spread)
        .def(py::self == py::self)
        .def(py::self != py::self);
    module.def("reference_moments", [](const kan::BasisConfig& config) {
        kan::BasisMoments m;
        {
            py::gil_scoped_release release;
            m = kan::reference_moments(config);
        }
        py::array_t<double> moments(static_cast<py::ssize_t>(m.second_moments.size()), m.second_moments.data());
        return py::make_tuple(m.variance, moments);
    }, py::arg("config"), "(variance, second_moments) of the family's reference measure");
    module.def("initialize", [](kan::Layer& layer, const kan::Initializer& initializer) {
        kan::initialize(layer, initializer);
    }, py::arg("layer"), py::arg("initializer"), py::call_guard<py::gil_scoped_release>());
    module.def("initialize", [](kan::Network& network, const kan::Initializer& initializer) {
        kan::initialize(network, initializer);
    }, py::arg("network"), py::arg("initializer"), py::call_guard<py::gil_scoped_release>());
    module.def("layer_seed", &kan::layer_seed, py::arg("seed"), py::arg("position"));
}
