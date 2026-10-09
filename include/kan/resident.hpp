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

// Loss whose gradient a training step computes on the device (backlog C9).
enum class Loss {
    // The upstream uploaded by upload_output_gradient(): the gradient of the
    // caller's own loss, kept resident across steps (what backward() uses).
    OutputGradient,
    // Mean squared error against the resident target (upload_target() or the
    // staging train_step): L = sum (y - t)^2 / (batch*outputs), upstream
    // 2(y - t)/(batch*outputs), as torch mse_loss with reduction "mean".
    MeanSquaredError,
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
    // Add lambda*c to coefficient/numerator VJPs and lambda*w to the SiLU
    // residual-branch weight VJPs (backlog M3) on device. Biases, RBF
    // nonlinear parameters and rational denominators are unpenalized.
    void backward(double coefficient_l2 = 0.0);
    void sgd(double learning_rate);
    // Replaces every trainable parameter (coefficients, biases, trainable RBF
    // centers and log widths, rational denominators, SiLU residual-branch
    // weights, LayerNorm gain and bias) with those of `network` (backlog R9),
    // e.g. after CPU training or to restore a model. `network` must have the
    // structure of the network the executor was built from: the same layer
    // kinds and dimensions, carriers and map kinds, the residual branch on the
    // same layers, and equal fixed configuration (basis configuration
    // including knots and fixed centers, rational configuration, fixed input
    // maps, LayerNorm epsilon); a network returned by download_parameters()
    // always qualifies. Everything is validated on the host first, with the
    // construction rules of this precision (FP32: representability); any
    // violation raises std::invalid_argument and leaves the executor
    // unchanged. Success invalidates the output and gradients (backward needs
    // a new forward) and keeps the uploaded input and upstream. No device
    // allocation; one host-to-device copy, or for FP64 parameters above 1 MiB
    // one per parameter tensor (copied without host staging).
    void upload_parameters(const Network& network);

    // Training steps (backlog C9; contract in docs/CONTRACT.md). One step is
    // forward, the loss gradient, backward(coefficient_l2) and
    // sgd(learning_rate), replayed as a captured CUDA graph without host
    // synchronization; with Loss::OutputGradient it computes bitwise what the
    // eager forward(); backward(l2); sgd(rate) sequence computes. Arguments and
    // lifecycle are checked like those calls (OutputGradient needs an input and
    // an upstream, MeanSquaredError an input, a target and a nonempty batch).
    // Outputs and gradients are stale afterwards, as after sgd(); an MSE step
    // consumes the uploaded upstream (its own gradient replaces it).
    void train_step(double learning_rate, double coefficient_l2 = 0.0, Loss loss = Loss::OutputGradient);
    // Trains one MSE step on a new host batch: `input` (batch x inputs) and
    // `target` (batch x outputs) are validated (std::invalid_argument, nothing
    // changes) and converted into page-locked staging before the call returns,
    // so the caller may reuse them at once; the copy runs on a separate stream,
    // overlapping earlier steps still executing. The batch becomes the
    // executor's input and target.
    void train_step(std::span<const double> input, std::span<const double> target, std::size_t batch,
                    double learning_rate, double coefficient_l2 = 0.0);
    // Resident target for Loss::MeanSquaredError (batch x outputs, like
    // upload_output_gradient). Input upload invalidates it.
    void upload_target(std::span<const double> target);
    // Loss value of the most recent training step, which must have used
    // MeanSquaredError (std::logic_error otherwise): its forward pass, i.e.
    // before that step's update.
    double download_loss();
    // Deferred status: the device status of training steps is checked once
    // every `steps` training steps (default 1: every step reports its own
    // failure), when check_status() is called, and before every other call
    // except the const queries. The first failing step of an interval is
    // reported with the usual exception (std::overflow_error for nonfinite
    // results, std::domain_error for rational poles), naming the step; that
    // step and the later steps of the interval commit no update, so the
    // parameters are those after the last good step, outputs, gradients and
    // the loss are stale, and the executor stays usable. trained_steps() then
    // equals the failing step's index. steps = 0 raises std::invalid_argument.
    void set_status_interval(std::size_t steps);
    std::size_t status_interval() const;
    void check_status();
    // Training steps whose update was committed, as confirmed by the last
    // status check (exact whenever no step is pending, e.g. after check_status()).
    std::size_t trained_steps() const;

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
