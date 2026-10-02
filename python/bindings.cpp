#include "kan/families.hpp"
#include "kan/network.hpp"
#ifdef KAN_PYTHON_CUDA
#include "kan/resident.hpp"
#include "kan/cuda.hpp"
#endif
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <variant>

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
    return {axis(layer.outputs()), axis(layer.inputs()), axis(layer.terms())};
}

std::size_t denominator_degree(const kan::Layer& layer) {
    const auto* rational = std::get_if<kan::RationalEdges>(&layer.carrier());
    return rational ? rational->config.denominator_degree : 0;
}

// Value-semantics Python class for one basis configuration type.
template <typename Config, typename... Extra>
py::class_<Config> config_class(py::module_& module, const char* name, Extra&&... extra) {
    py::class_<Config> result(module, name, std::forward<Extra>(extra)...);
    result.def(py::self == py::self);
    return result;
}

// Localized families derive their term count from their parameter vectors.
template <typename Config>
void derived_size(py::class_<Config>& config) {
    config.def_property_readonly("size", [](const Config& c) { return kan::basis_size(c); });
}
Shape denominator_shape(const kan::Layer& layer) {
    if (!std::holds_alternative<kan::RationalEdges>(layer.carrier())) return {0};
    return {axis(layer.outputs()), axis(layer.inputs()), axis(denominator_degree(layer))};
}

// What a gradient snapshot was computed for; sgd requires the same.
struct Topology {
    std::size_t inputs, outputs, terms, carrier, denominator_size;
    bool operator==(const Topology&) const = default;
};
Topology topology(const kan::Layer& layer) {
    return {layer.inputs(), layer.outputs(), layer.terms(), layer.carrier().index(), denominator_degree(layer)};
}

// Read-only snapshot of one carrier alternative with its layer's dimensions,
// so parameter arrays keep their (outputs, inputs, terms) shape.
template <typename Edges> struct CarrierView {
    Edges value;
    std::size_t inputs, outputs;
    py::array_t<double> coefficients() const {
        const auto terms = value.coefficients.size() / (inputs * outputs);
        return owned(value.coefficients, {axis(outputs), axis(inputs), axis(terms)});
    }
};
py::object carrier_view(const kan::Layer& layer) {
    return std::visit([&](const auto& edges) -> py::object {
        using Edges = std::decay_t<decltype(edges)>;
        return py::cast(CarrierView<Edges>{edges, layer.inputs(), layer.outputs()});
    }, layer.carrier());
}

struct LayerGradient {
    kan::LayerGradients value;
    std::size_t batch;
    Topology topology;
};
struct NetworkGradient {
    kan::NetworkGradients value;
    std::size_t batch;
    std::vector<kan::Layer> topology;
};

LayerGradient wrap(kan::LayerGradients value, std::size_t batch, const kan::Layer& layer) {
    return {std::move(value), batch, topology(layer)};
}

// A nonlinear gradient vector of one carrier; empty for the other carriers.
template <typename Gradients>
std::span<const double> nonlinear_field(const LayerGradient& g, std::vector<double> Gradients::*field) {
    const auto* nonlinear = std::get_if<Gradients>(&g.value.nonlinear);
    return nonlinear ? std::span<const double>(nonlinear->*field) : std::span<const double>();
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
    py::class_<kan::RationalConfig>(module, "RationalConfig")
        .def(py::init<>())
        .def_readwrite("numerator_degree", &kan::RationalConfig::numerator_degree)
        .def_readwrite("denominator_degree", &kan::RationalConfig::denominator_degree)
        .def_readwrite("center", &kan::RationalConfig::center)
        .def_readwrite("scale", &kan::RationalConfig::scale)
        .def_readwrite("epsilon", &kan::RationalConfig::epsilon);
#ifdef KAN_PYTHON_CUDA
    module.attr("cuda_enabled") = true;
#else
    module.attr("cuda_enabled") = false;
#endif
    // One class per family: each holds only its own parameters. Global
    // families have an explicit size; localized families derive it.
    config_class<kan::ChebyshevConfig>(module, "ChebyshevConfig")
        .def(py::init([](std::size_t size) { return kan::ChebyshevConfig{size}; }), py::arg("size") = 4)
        .def_readwrite("size", &kan::ChebyshevConfig::size);
    config_class<kan::LegendreConfig>(module, "LegendreConfig")
        .def(py::init([](std::size_t size) { return kan::LegendreConfig{size}; }), py::arg("size") = 4)
        .def_readwrite("size", &kan::LegendreConfig::size);
    config_class<kan::HermiteConfig>(module, "HermiteConfig")
        .def(py::init([](std::size_t size) { return kan::HermiteConfig{size}; }), py::arg("size") = 4)
        .def_readwrite("size", &kan::HermiteConfig::size);
    config_class<kan::JacobiConfig>(module, "JacobiConfig")
        .def(py::init([](std::size_t size, double alpha, double beta) { return kan::JacobiConfig{size, alpha, beta}; }),
             py::arg("size") = 4, py::arg("alpha") = 0.0, py::arg("beta") = 0.0)
        .def_readwrite("size", &kan::JacobiConfig::size)
        .def_readwrite("alpha", &kan::JacobiConfig::alpha)
        .def_readwrite("beta", &kan::JacobiConfig::beta);
    config_class<kan::FourierConfig>(module, "FourierConfig")
        .def(py::init([](std::size_t size, double frequency) { return kan::FourierConfig{size, frequency}; }),
             py::arg("size") = 3, py::arg("frequency") = 1.0)
        .def_readwrite("size", &kan::FourierConfig::size)
        .def_readwrite("frequency", &kan::FourierConfig::frequency);
    auto gaussian = config_class<kan::GaussianRbfConfig>(module, "GaussianRbfConfig");
    derived_size(gaussian);
    gaussian
        .def(py::init([](std::vector<double> centers, double width) {
            return kan::GaussianRbfConfig{std::move(centers), width}; }),
             py::arg("centers") = std::vector<double>{}, py::arg("width") = 1.0)
        .def_readwrite("centers", &kan::GaussianRbfConfig::centers)
        .def_readwrite("width", &kan::GaussianRbfConfig::width);
    auto trainable = config_class<kan::TrainableRbfConfig>(module, "TrainableRbfConfig");
    derived_size(trainable);
    trainable
        .def(py::init([](std::vector<double> centers, std::vector<double> log_widths) {
            return kan::TrainableRbfConfig{std::move(centers), std::move(log_widths)}; }),
             py::arg("centers") = std::vector<double>{}, py::arg("log_widths") = std::vector<double>{})
        .def_readwrite("centers", &kan::TrainableRbfConfig::centers)
        .def_readwrite("log_widths", &kan::TrainableRbfConfig::log_widths);
    auto spline = config_class<kan::BSplineConfig>(module, "BSplineConfig");
    derived_size(spline);
    spline
        .def(py::init([](std::size_t degree, std::vector<double> knots) {
            return kan::BSplineConfig{degree, std::move(knots)}; }),
             py::arg("degree") = 3, py::arg("knots") = std::vector<double>{})
        .def_readwrite("degree", &kan::BSplineConfig::degree)
        .def_readwrite("knots", &kan::BSplineConfig::knots);
    auto wavelet = config_class<kan::MexicanHatConfig>(module, "MexicanHatConfig");
    derived_size(wavelet);
    wavelet
        .def(py::init([](std::vector<double> centers, std::vector<double> scales) {
            return kan::MexicanHatConfig{std::move(centers), std::move(scales)}; }),
             py::arg("centers") = std::vector<double>{}, py::arg("scales") = std::vector<double>{})
        .def_readwrite("centers", &kan::MexicanHatConfig::centers)
        .def_readwrite("scales", &kan::MexicanHatConfig::scales);
    module.def("basis_size", [](const kan::BasisConfig& config) { return kan::basis_size(config); },
               py::arg("config"));
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
    module.def("evaluate_rational", [](kan::RationalConfig config, double x, py::array numerator, py::array denominator) {
        kan::validate_rational(config);
        const auto a = shaped(numerator, {axis(config.numerator_degree+1)});
        const auto b = shaped(denominator, {axis(config.denominator_degree)});
        kan::RationalEvaluation r;
        { py::gil_scoped_release release; r=kan::evaluate_rational(config,x,a,b); }
        return py::make_tuple(r.value,r.input_derivative,
            owned(r.numerator_derivatives,{axis(r.numerator_derivatives.size())}),
            owned(r.denominator_derivatives,{axis(r.denominator_derivatives.size())}));
    }, py::arg("config"), py::arg("x"), py::arg("numerator").noconvert(), py::arg("denominator").noconvert());

    // Read-only carrier snapshots returned by Layer.carrier, one class per carrier.
    py::class_<CarrierView<kan::BasisEdges>>(module, "BasisEdges")
        .def_property_readonly("basis", [](const CarrierView<kan::BasisEdges>& c) { return c.value.basis; })
        .def_property_readonly("coefficients", &CarrierView<kan::BasisEdges>::coefficients);
    py::class_<CarrierView<kan::TrainableRbfEdges>>(module, "TrainableRbfEdges")
        .def_property_readonly("basis", [](const CarrierView<kan::TrainableRbfEdges>& c) { return c.value.basis; })
        .def_property_readonly("coefficients", &CarrierView<kan::TrainableRbfEdges>::coefficients);
    py::class_<CarrierView<kan::RationalEdges>>(module, "RationalEdges")
        .def_property_readonly("config", [](const CarrierView<kan::RationalEdges>& c) { return c.value.config; })
        .def_property_readonly("coefficients", &CarrierView<kan::RationalEdges>::coefficients)
        .def_property_readonly("denominators", [](const CarrierView<kan::RationalEdges>& c) {
            return owned(c.value.denominators, {axis(c.outputs), axis(c.inputs), axis(c.value.config.denominator_degree)});
        });
    // Family-specific operations (kan/families.hpp).
    module.def("insert_knot", &kan::insert_knot, py::arg("layer"), py::arg("x"),
               py::call_guard<py::gil_scoped_release>());
    module.def("adapt_grid", [](kan::Layer& layer, py::array samples) {
        const auto data = array_data(samples);
        if (samples.ndim() != 1) throw py::value_error("samples must be a vector");
        py::gil_scoped_release release;
        return kan::adapt_grid(layer, data);
    }, py::arg("layer"), py::arg("samples").noconvert());
    module.def("set_rbf_parameters", [](kan::Layer& layer, py::array centers, py::array log_widths) {
        if (!std::holds_alternative<kan::TrainableRbfEdges>(layer.carrier()))
            throw std::invalid_argument("RBF parameters require a trainable Gaussian basis");
        const auto c = shaped(centers, {axis(layer.terms())});
        const auto w = shaped(log_widths, {axis(layer.terms())});
        py::gil_scoped_release release;
        kan::set_rbf_parameters(layer, c, w);
    }, py::arg("layer"), py::arg("centers").noconvert(), py::arg("log_widths").noconvert());
    module.def("set_rational_parameters", [](kan::Layer& layer, py::array coefficients, py::array denominators, py::array bias) {
        if (!std::holds_alternative<kan::RationalEdges>(layer.carrier()))
            throw std::invalid_argument("rational parameter shape or layer type mismatch");
        const auto c = shaped(coefficients, coefficient_shape(layer));
        const auto d = shaped(denominators, denominator_shape(layer));
        const auto b = shaped(bias, {axis(layer.outputs())});
        py::gil_scoped_release release;
        kan::set_rational_parameters(layer, c, d, b);
    }, py::arg("layer"), py::arg("coefficients").noconvert(), py::arg("denominators").noconvert(),
       py::arg("bias").noconvert());

    py::class_<LayerGradient>(module, "LayerGradients")
        .def_property_readonly("input", [](const LayerGradient& g) {
            return owned(g.value.input, {axis(g.batch), axis(g.topology.inputs)});
        })
        .def_property_readonly("coefficients", [](const LayerGradient& g) {
            const auto& t = g.topology;
            return owned(g.value.coefficients, {axis(t.outputs), axis(t.inputs), axis(t.terms)});
        })
        .def_property_readonly("bias", [](const LayerGradient& g) {
            return owned(g.value.bias, {axis(g.topology.outputs)});
        })
        .def_property_readonly("centers", [](const LayerGradient& g) {
            const auto v = nonlinear_field(g, &kan::TrainableRbfGradients::centers);
            return owned(v, {axis(v.size())});
        })
        .def_property_readonly("log_widths", [](const LayerGradient& g) {
            const auto v = nonlinear_field(g, &kan::TrainableRbfGradients::log_widths);
            return owned(v, {axis(v.size())});
        })
        .def_property_readonly("denominators", [](const LayerGradient& g) {
            const auto& t = g.topology;
            const auto v = nonlinear_field(g, &kan::RationalGradients::denominators);
            return owned(v, std::holds_alternative<kan::RationalGradients>(g.value.nonlinear) ?
                Shape{axis(t.outputs), axis(t.inputs), axis(t.denominator_size)} : Shape{0});
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
        .def(py::init<std::size_t, std::size_t, kan::RationalConfig>(),
             py::arg("inputs"), py::arg("outputs"), py::arg("rational"))
        .def_property_readonly("inputs", &kan::Layer::inputs)
        .def_property_readonly("outputs", &kan::Layer::outputs)
        .def_property_readonly("carrier", &carrier_view,
            "Owned read-only snapshot of the layer's carrier (BasisEdges, TrainableRbfEdges or "
            "RationalEdges); every access copies the parameters.")
        .def_property_readonly("terms", &kan::Layer::terms)
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
        .def("regularization", [](const kan::Layer& layer, double lambda) {
            kan::RegularizationResult r;
            {py::gil_scoped_release release;r=layer.regularization(lambda);}
            return py::make_tuple(r.value,wrap(std::move(r.gradients),0,layer));
        }, py::arg("coefficient_l2"))
        .def("forward", &forward<kan::Layer>, py::arg("input").noconvert(), py::arg("batch") = py::none())
        .def("backward", &backward<kan::Layer>, py::arg("input").noconvert(),
             py::arg("output_gradient").noconvert(), py::arg("batch") = py::none())
        .def("sgd", [](kan::Layer& layer, const LayerGradient& gradient, double learning_rate) {
            if (gradient.topology != topology(layer)) throw py::value_error("gradient topology mismatch");
            py::gil_scoped_release release;
            layer.sgd(gradient.value, learning_rate);
        }, py::arg("gradients"), py::arg("learning_rate"));
    py::class_<kan::Network>(module, "Network")
        .def(py::init<std::vector<kan::Layer>>(), py::arg("layers"))
        .def_property_readonly("layers", [](const kan::Network& model) {
            return std::vector<kan::Layer>(model.layers().begin(), model.layers().end());
        })
        .def("insert_knot", &kan::Network::insert_knot, py::arg("layer_index"), py::arg("x"),
             py::call_guard<py::gil_scoped_release>())
        .def("adapt_grid", [](kan::Network& model, std::size_t index, py::array samples) {
            const auto data=array_data(samples);
            if(samples.ndim()!=1)throw py::value_error("samples must be a vector");
            py::gil_scoped_release release;return model.adapt_grid(index,data);
        }, py::arg("layer_index"), py::arg("samples").noconvert())
        .def("regularization", [](const kan::Network& model, double lambda) {
            kan::NetworkRegularizationResult r;
            {py::gil_scoped_release release;r=model.regularization(lambda);}
            return py::make_tuple(r.value,NetworkGradient{std::move(r.gradients),0,
                {model.layers().begin(),model.layers().end()}});
        }, py::arg("coefficient_l2"))
        .def("forward", &forward<kan::Network>, py::arg("input").noconvert(), py::arg("batch") = py::none())
        .def("backward", &backward<kan::Network>, py::arg("input").noconvert(),
             py::arg("output_gradient").noconvert(), py::arg("batch") = py::none())
        .def("sgd", [](kan::Network& model, const NetworkGradient& gradient, double learning_rate) {
            if (model.layers().size() != gradient.topology.size())
                throw py::value_error("gradient topology mismatch");
            for (std::size_t i = 0; i < gradient.topology.size(); ++i) {
                if (topology(model.layers()[i]) != topology(gradient.topology[i]))
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
        .def("backward", [](Resident& model, double lambda) { model.value.backward(lambda); },
             py::arg("coefficient_l2")=0.0, py::call_guard<py::gil_scoped_release>())
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
