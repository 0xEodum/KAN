// PQC carrier (M5 phase 1): configuration validation, single-circuit
// evaluation and the family setter. RED stub: no validation, zero evaluation.
#include "kan/families.hpp"
#include "kan/pqc.hpp"
#include <stdexcept>

namespace kan {

std::size_t pqc_terms(const PqcConfig& config) noexcept {
    return config.readout == PqcReadout::Parity ? 1 : config.qubits;
}
std::size_t pqc_angles(const PqcConfig& config) noexcept {
    return (config.repetitions + 1) * config.qubits * config.block.size();
}
std::size_t pqc_encodings(const PqcConfig& config) noexcept { return config.repetitions * config.qubits; }

void validate_pqc(const PqcConfig&) {}

PqcEvaluation evaluate_pqc(const PqcConfig& config, double, std::span<const double>, std::span<const double>,
                           std::span<const double>, GradientMethod) {
    const auto terms = pqc_terms(config);
    return {std::vector<double>(terms), std::vector<double>(terms), std::vector<double>(terms * pqc_angles(config)),
            std::vector<double>(terms * pqc_encodings(config)), std::vector<double>(terms * pqc_encodings(config))};
}

std::vector<std::complex<double>> pqc_state(const PqcConfig& config, double, std::span<const double>,
                                            std::span<const double>, std::span<const double>) {
    return std::vector<std::complex<double>>(std::size_t{1} << (config.qubits % 6));
}

void set_pqc_parameters(Layer& layer, std::span<const double> coefficients, std::span<const double> angles,
                        std::span<const double> encoding_weights, std::span<const double> encoding_biases,
                        std::span<const double> bias) {
    const auto* edges = std::get_if<PqcEdges>(&layer.carrier());
    if (!edges) throw std::invalid_argument("PQC parameters require a PQC layer");
    layer.set_carrier(PqcEdges{edges->config,
                               {coefficients.begin(), coefficients.end()},
                               {angles.begin(), angles.end()},
                               {encoding_weights.begin(), encoding_weights.end()},
                               {encoding_biases.begin(), encoding_biases.end()},
                               edges->shot_seed},
                      bias);
}

} // namespace kan
