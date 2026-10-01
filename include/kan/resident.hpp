#pragma once

#include "kan/network.hpp"
#include <memory>

namespace kan::cuda {

// Move-only GPU network. Construction reserves all device workspaces for the
// maximum batch; execution and SGD never allocate device storage.
class ResidentNetwork {
public:
    ResidentNetwork(const Network& network, std::size_t capacity);
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
    std::vector<double> download_output();
    NetworkGradients download_gradients();
    Network download_parameters();
    void synchronize();
    std::size_t capacity() const;
    std::size_t batch() const;
    std::size_t workspace_allocations() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Impl& state() const;
};

} // namespace kan::cuda
