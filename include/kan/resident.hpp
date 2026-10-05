#pragma once

#include "kan/cuda_runtime.hpp" // kan::cuda::available()
#include "kan/network.hpp"
#include <memory>

namespace kan::cuda {

// Storage and arithmetic precision of a resident executor (backlog C1). Host
// data stays double in both: uploads are rounded to the executor precision
// and downloads are exact conversions of the device values.
enum class Precision {
    Float64, // default: double storage and kernels, the parity reference
    Float32, // float storage and kernels, cuBLAS SGEMM; opt-in for training
    // Float32 whose cuBLAS contractions use TF32 tensor cores (10-bit
    // mantissa operands, FP32 accumulation): faster large GEMMs, looser tolerance.
    TensorFloat32,
};

// Move-only GPU network. Construction reserves all device workspaces for the
// maximum batch; execution and SGD never allocate device storage.
class ResidentNetwork {
public:
    ResidentNetwork(const Network& network, std::size_t capacity, Precision precision = Precision::Float64);
    ~ResidentNetwork();
    ResidentNetwork(ResidentNetwork&&) noexcept;
    ResidentNetwork& operator=(ResidentNetwork&&) noexcept;
    ResidentNetwork(const ResidentNetwork&) = delete;
    ResidentNetwork& operator=(const ResidentNetwork&) = delete;

    void upload_input(std::span<const double> input, std::size_t batch);
    void upload_output_gradient(std::span<const double> gradient);
    void forward();
    // Add lambda*c to coefficient/numerator VJPs on device. RBF nonlinear
    // parameters and rational denominators are unpenalized.
    void backward(double coefficient_l2 = 0.0);
    void sgd(double learning_rate);
    // Replaces every trainable parameter (coefficients, biases, trainable RBF
    // centers and log widths, rational denominators, LayerNorm gain and bias)
    // with those of `network` (backlog R9), e.g. after CPU training or to
    // restore a model. `network` must have the structure of the network the
    // executor was built from: the same layer kinds and dimensions, carriers
    // and map kinds, and equal fixed configuration (basis configuration
    // including knots and fixed centers, rational configuration, fixed input
    // maps, LayerNorm epsilon); a network returned by download_parameters()
    // always qualifies. Everything is validated on the host first, with the
    // construction rules of this precision (FP32: representability); any
    // violation raises std::invalid_argument and leaves the executor
    // unchanged. Success invalidates the output and gradients (backward needs
    // a new forward) and keeps the uploaded input and upstream. No device
    // allocation; one host-to-device copy.
    void upload_parameters(const Network& network);
    std::vector<double> download_output();
    NetworkGradients download_gradients();
    Network download_parameters();
    void synchronize();
    std::size_t capacity() const;
    std::size_t batch() const;
    std::size_t workspace_allocations() const;
    Precision precision() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Impl& state() const;
};

} // namespace kan::cuda
