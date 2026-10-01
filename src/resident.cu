#include "kan/resident.hpp"
#include <stdexcept>
namespace kan::cuda {
struct ResidentNetwork::Impl {};
ResidentNetwork::ResidentNetwork(const Network&, std::size_t) { throw std::logic_error("M2 resident CUDA not implemented"); }
ResidentNetwork::~ResidentNetwork() = default;
ResidentNetwork::ResidentNetwork(ResidentNetwork&&) noexcept = default;
ResidentNetwork& ResidentNetwork::operator=(ResidentNetwork&&) noexcept = default;
ResidentNetwork::Impl& ResidentNetwork::state() const { throw std::logic_error("M2 resident CUDA not implemented"); }
void ResidentNetwork::upload_input(std::span<const double>, std::size_t) { state(); }
void ResidentNetwork::upload_output_gradient(std::span<const double>) { state(); }
void ResidentNetwork::forward() { state(); }
void ResidentNetwork::backward() { state(); }
void ResidentNetwork::sgd(double) { state(); }
std::vector<double> ResidentNetwork::download_output() { state(); return {}; }
NetworkGradients ResidentNetwork::download_gradients() { state(); return {}; }
Network ResidentNetwork::download_parameters() { state(); return Network({}); }
void ResidentNetwork::synchronize() { state(); }
std::size_t ResidentNetwork::capacity() const { state(); return 0; }
std::size_t ResidentNetwork::batch() const { state(); return 0; }
std::size_t ResidentNetwork::workspace_allocations() const { state(); return 0; }
}
