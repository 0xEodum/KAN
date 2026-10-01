#include "kan/network.hpp"
#ifdef KAN_PYTHON_CUDA
#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;
namespace {
using Shape = std::vector<py::ssize_t>;

py::ssize_t axis(std::size_t value) {
    if (value > static_cast<std::size_t>(PY_SSIZE_T_MAX))
        throw py::value_error("array dimension exceeds Python index range");
    return static_cast<py::ssize_t>(value);
}

std::span<const double> array_data(const py::array& array) {
    if (!array.dtype().is(py::dtype::of<double>()))
        throw py::type_error("expected a native-endian float64 NumPy array");
    if (!(array.flags() & py::array::c_style))
        throw py::value_error("expected a C-contiguous NumPy array");
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(double) != 0)
        throw py::value_error("expected a double-aligned NumPy array");
    return {static_cast<const double*>(array.data()), static_cast<std::size_t>(array.size())};
}

std::span<const double> shaped(const py::array& array, const Shape& shape) {
    const auto data = array_data(array);
    if (array.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw py::value_error("array rank mismatch");
    for (py::ssize_t i = 0; i < array.ndim(); ++i)
        if (array.shape(i) != shape[static_cast<std::size_t>(i)])
            throw py::value_error("array shape mismatch");
    return data;
}

std::size_t input_batch(const py::array& input, std::size_t inputs,
                        std::optional<std::size_t> requested) {
    array_data(input);
    if (input.ndim() != 2 || input.shape(1) != axis(inputs))
        throw py::value_error("input must have shape (batch, inputs)");
    const auto batch = static_cast<std::size_t>(input.shape(0));
    if (requested && *requested != batch) throw py::value_error("explicit batch mismatch");
    return batch;
}

py::array_t<double> owned(std::span<const double> values, const Shape& shape) {
    py::array_t<double> result(shape);
    if (static_cast<std::size_t>(result.size()) != values.size())
        throw std::logic_error("internal binding array shape mismatch");
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

Shape coefficient_shape(const kan::Layer& layer) {
    return {axis(layer.outputs()), axis(layer.inputs()), axis(layer.basis().size)};
}

struct LayerGradient {
    kan::LayerGradients value;
    std::size_t batch, inputs, outputs, basis_size;
};
struct NetworkGradient {
    kan::NetworkGradients value;
    std::size_t batch;
    std::vector<kan::Layer> topology;
};

LayerGradient wrap(kan::LayerGradients value, std::size_t batch, const kan::Layer& layer) {
    return {std::move(value), batch, layer.inputs(), layer.outputs(), layer.basis().size};
}

template <typename Model> std::size_t input_count(const Model& model) {
    if constexpr (std::is_same_v<Model, kan::Layer>) return model.inputs();
    else return model.layers().front().inputs();
}
template <typename Model> std::size_t output_count(const Model& model) {
    if constexpr (std::is_same_v<Model, kan::Layer>) return model.outputs();
    else return model.layers().back().outputs();
}

template <typename Model> auto forward(const Model& model, py::array input,
                                       std::optional<std::size_t> requested) {
    const auto batch = input_batch(input, input_count(model), requested);
    const auto data = array_data(input);
    std::vector<double> output;
    {
        py::gil_scoped_release release;
        output = model.forward(data, batch);
    }
    return owned(output, {axis(batch), axis(output_count(model))});
}

template <typename Model> auto backward(const Model& model, py::array input,
                                        py::array upstream, std::optional<std::size_t> requested) {
    const auto batch = input_batch(input, input_count(model), requested);
    const auto data = array_data(input);
    const auto gradient_data = shaped(upstream, {axis(batch), axis(output_count(model))});
    using Gradient = decltype(model.backward(data, batch, gradient_data));
    Gradient gradient;
    {
        py::gil_scoped_release release;
        gradient = model.backward(data, batch, gradient_data);
    }
    if constexpr (std::is_same_v<Model, kan::Layer>) return wrap(std::move(gradient), batch, model);
    else return NetworkGradient{std::move(gradient), batch,
                               {model.layers().begin(), model.layers().end()}};
}

#ifdef KAN_PYTHON_CUDA
struct Resident {
    std::vector<kan::Layer> topology;
    kan::cuda::ResidentNetwork value;
    Resident(const kan::Network& network, std::size_t capacity)
        : topology(network.layers().begin(), network.layers().end()), value(network, capacity) {}
};
#endif
} // namespace

PYBIND11_MODULE(_kan, module) {
    module.doc() = "Optional strict float64 NumPy bindings for KAN";
#ifdef KAN_PYTHON_CUDA
    module.attr("cuda_enabled") = true;
#else
    module.attr("cuda_enabled") = false;
#endif
    py::enum_<kan::BasisKind>(module, "BasisKind")
        .value("Chebyshev", kan::BasisKind::Chebyshev)
        .value("Legendre", kan::BasisKind::Legendre)
        .value("Jacobi", kan::BasisKind::Jacobi)
        .value("Hermite", kan::BasisKind::Hermite)
        .value("Fourier", kan::BasisKind::Fourier)
        .value("GaussianRbf", kan::BasisKind::GaussianRbf);
    py::class_<kan::BasisConfig>(module, "BasisConfig")
        .def(py::init<>())
        .def_readwrite("kind", &kan::BasisConfig::kind)
        .def_readwrite("size", &kan::BasisConfig::size)
        .def_readwrite("alpha", &kan::BasisConfig::alpha)
        .def_readwrite("beta", &kan::BasisConfig::beta)
        .def_readwrite("frequency", &kan::BasisConfig::frequency)
        .def_readwrite("centers", &kan::BasisConfig::centers)
        .def_readwrite("width", &kan::BasisConfig::width);
    module.def("cuda_available", [] {
#ifdef KAN_PYTHON_CUDA
        return kan::cuda::available();
#else
        return false;
#endif
    });
    module.def("evaluate_basis", [](kan::BasisConfig config, double x) {
        kan::BasisValues basis;
        {
            py::gil_scoped_release release;
            basis = kan::evaluate_basis(config, x);
        }
        return py::make_tuple(owned(basis.values, {axis(basis.values.size())}),
                              owned(basis.derivatives, {axis(basis.derivatives.size())}));
    }, py::arg("config"), py::arg("x"));

    py::class_<LayerGradient>(module, "LayerGradients")
        .def_property_readonly("input", [](const LayerGradient& g) {
            return owned(g.value.input, {axis(g.batch), axis(g.inputs)});
        })
        .def_property_readonly("coefficients", [](const LayerGradient& g) {
            return owned(g.value.coefficients, {axis(g.outputs), axis(g.inputs), axis(g.basis_size)});
        })
        .def_property_readonly("bias", [](const LayerGradient& g) {
            return owned(g.value.bias, {axis(g.outputs)});
        });
    py::class_<NetworkGradient>(module, "NetworkGradients")
        .def_property_readonly("input", [](const NetworkGradient& g) {
            return owned(g.value.input, {axis(g.batch), axis(g.topology.front().inputs())});
        })
        .def_property_readonly("layers", [](const NetworkGradient& g) {
            std::vector<LayerGradient> layers;
            for (std::size_t i = 0; i < g.topology.size(); ++i)
                layers.push_back(wrap(g.value.layers[i], g.batch, g.topology[i]));
            return layers;
        });
    py::class_<kan::Layer>(module, "Layer")
        .def(py::init<std::size_t, std::size_t, kan::BasisConfig>(),
             py::arg("inputs"), py::arg("outputs"), py::arg("basis"))
        .def_property_readonly("inputs", &kan::Layer::inputs)
        .def_property_readonly("outputs", &kan::Layer::outputs)
        .def_property_readonly("basis", [](const kan::Layer& layer) { return layer.basis(); })
        .def_property_readonly("coefficients", [](const kan::Layer& layer) {
            return owned(layer.coefficients(), coefficient_shape(layer));
        })
        .def_property_readonly("bias", [](const kan::Layer& layer) {
            return owned(layer.bias(), {axis(layer.outputs())});
        })
        .def("set_parameters", [](kan::Layer& layer, py::array coefficients, py::array bias) {
            const auto c = shaped(coefficients, coefficient_shape(layer));
            const auto b = shaped(bias, {axis(layer.outputs())});
            py::gil_scoped_release release;
            layer.set_parameters(c, b);
        }, py::arg("coefficients").noconvert(), py::arg("bias").noconvert())
        .def("forward", &forward<kan::Layer>, py::arg("input").noconvert(), py::arg("batch") = py::none())
        .def("backward", &backward<kan::Layer>, py::arg("input").noconvert(),
             py::arg("output_gradient").noconvert(), py::arg("batch") = py::none())
        .def("sgd", [](kan::Layer& layer, const LayerGradient& gradient, double learning_rate) {
            if (gradient.inputs != layer.inputs() || gradient.outputs != layer.outputs() ||
                gradient.basis_size != layer.basis().size)
                throw py::value_error("gradient topology mismatch");
            py::gil_scoped_release release;
            layer.sgd(gradient.value, learning_rate);
        }, py::arg("gradients"), py::arg("learning_rate"));
    py::class_<kan::Network>(module, "Network")
        .def(py::init<std::vector<kan::Layer>>(), py::arg("layers"))
        .def_property_readonly("layers", [](const kan::Network& model) {
            return std::vector<kan::Layer>(model.layers().begin(), model.layers().end());
        })
        .def("forward", &forward<kan::Network>, py::arg("input").noconvert(), py::arg("batch") = py::none())
        .def("backward", &backward<kan::Network>, py::arg("input").noconvert(),
             py::arg("output_gradient").noconvert(), py::arg("batch") = py::none())
        .def("sgd", [](kan::Network& model, const NetworkGradient& gradient, double learning_rate) {
            if (model.layers().size() != gradient.topology.size())
                throw py::value_error("gradient topology mismatch");
            for (std::size_t i = 0; i < gradient.topology.size(); ++i) {
                const auto& actual = model.layers()[i];
                const auto& expected = gradient.topology[i];
                if (actual.inputs() != expected.inputs() || actual.outputs() != expected.outputs() ||
                    actual.basis().size != expected.basis().size)
                    throw py::value_error("gradient topology mismatch");
            }
            py::gil_scoped_release release;
            model.sgd(gradient.value, learning_rate);
        }, py::arg("gradients"), py::arg("learning_rate"));
#ifdef KAN_PYTHON_CUDA
    py::class_<Resident>(module, "ResidentNetwork")
        .def(py::init([](const kan::Network& network, std::size_t capacity) {
            py::gil_scoped_release release;
            return std::make_unique<Resident>(network, capacity);
        }), py::arg("network"), py::arg("capacity"))
        .def_property_readonly("capacity", [](const Resident& model) { return model.value.capacity(); })
        .def_property_readonly("batch", [](const Resident& model) { return model.value.batch(); })
        .def_property_readonly("workspace_allocations", [](const Resident& model) {
            return model.value.workspace_allocations();
        })
        .def("upload_input", [](Resident& model, py::array input, std::optional<std::size_t> requested) {
            const auto batch = input_batch(input, model.topology.front().inputs(), requested);
            const auto data = array_data(input);
            py::gil_scoped_release release;
            model.value.upload_input(data, batch);
        }, py::arg("input").noconvert(), py::arg("batch") = py::none())
        .def("upload_output_gradient", [](Resident& model, py::array gradient) {
            const auto data = shaped(gradient, {axis(model.value.batch()), axis(model.topology.back().outputs())});
            py::gil_scoped_release release;
            model.value.upload_output_gradient(data);
        }, py::arg("output_gradient").noconvert())
        .def("forward", [](Resident& model) { model.value.forward(); }, py::call_guard<py::gil_scoped_release>())
        .def("backward", [](Resident& model) { model.value.backward(); }, py::call_guard<py::gil_scoped_release>())
        .def("sgd", [](Resident& model, double learning_rate) { model.value.sgd(learning_rate); },
             py::arg("learning_rate"), py::call_guard<py::gil_scoped_release>())
        .def("synchronize", [](Resident& model) { model.value.synchronize(); }, py::call_guard<py::gil_scoped_release>())
        .def("download_output", [](Resident& model) {
            std::vector<double> output;
            {
                py::gil_scoped_release release;
                output = model.value.download_output();
            }
            return owned(output, {axis(model.value.batch()), axis(model.topology.back().outputs())});
        })
        .def("download_gradients", [](Resident& model) {
            kan::NetworkGradients gradient;
            {
                py::gil_scoped_release release;
                gradient = model.value.download_gradients();
            }
            return NetworkGradient{std::move(gradient), model.value.batch(), model.topology};
        })
        .def("download_parameters", [](Resident& model) { return model.value.download_parameters(); },
             py::call_guard<py::gil_scoped_release>());
#endif
}
