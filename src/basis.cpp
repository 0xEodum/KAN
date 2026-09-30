#include "kan/basis.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace kan {

namespace {
double finite_result(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("basis result is not finite");
    return value;
}

void positive_finite(double value, const char* message) {
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument(message);
}
}

void validate_basis(const BasisConfig& config) {
    if (config.size == 0) throw std::invalid_argument("basis size must be positive");
    if (config.size > std::vector<double>{}.max_size())
        throw std::overflow_error("basis size exceeds vector capacity");
    switch (config.kind) {
    case BasisKind::Chebyshev:
    case BasisKind::Legendre:
    case BasisKind::Hermite:
        break;
    case BasisKind::Jacobi:
        if (!std::isfinite(config.alpha) || !std::isfinite(config.beta) ||
            config.alpha <= -1 || config.beta <= -1)
            throw std::invalid_argument("Jacobi alpha and beta must be finite and greater than -1");
        break;
    case BasisKind::Fourier:
        if (config.size % 2 == 0) throw std::invalid_argument("Fourier size must be odd");
        positive_finite(config.frequency, "Fourier frequency must be finite and positive");
        break;
    case BasisKind::GaussianRbf:
        if (config.centers.size() != config.size)
            throw std::invalid_argument("Gaussian center count must match basis size");
        for (double center : config.centers)
            if (!std::isfinite(center)) throw std::invalid_argument("Gaussian centers must be finite");
        positive_finite(config.width, "Gaussian width must be finite and positive");
        break;
    default:
        throw std::invalid_argument("unknown basis kind");
    }
}

BasisValues evaluate_basis(const BasisConfig& config, double x) {
    validate_basis(config);
    if (!std::isfinite(x)) throw std::invalid_argument("basis input must be finite");
    BasisValues result{std::vector<double>(config.size), std::vector<double>(config.size)};

    if (config.kind == BasisKind::GaussianRbf) {
        for (std::size_t k = 0; k < config.size; ++k) {
            const double distance = x - config.centers[k];
            // Overflow in subtraction is possible only for opposite signs. Divide
            // before subtracting in that case, retaining a wide Gaussian's tail.
            const double q = std::isfinite(distance) ? distance / config.width :
                             x / config.width - config.centers[k] / config.width;
            const double value = std::exp(-q*q);
            result.values[k] = value;
            double derivative = 0;
            if (q != 0 && std::isfinite(q)) {
                if (value < std::numeric_limits<double>::min()) {
                    // A tiny width can amplify a value that underflows to zero
                    // into a representable derivative. Preserve it in log space.
                    const double log_magnitude = std::log(2.0) + std::log(std::abs(q)) -
                                                 q*q - std::log(config.width);
                    derivative = -std::copysign(std::exp(log_magnitude), q);
                } else {
                    derivative = (-2*q*value) / config.width;
                }
            }
            result.derivatives[k] = finite_result(derivative);
        }
        return result;
    }

    result.values[0] = 1;
    if (config.kind == BasisKind::Fourier) {
        for (std::size_t k = 1; k <= config.size / 2; ++k) {
            const double angular = finite_result(static_cast<double>(k) * config.frequency);
            const double phase = finite_result(angular*x);
            const double cosine = std::cos(phase);
            const double sine = std::sin(phase);
            result.values[2*k-1] = cosine;
            result.values[2*k] = sine;
            result.derivatives[2*k-1] = finite_result(-angular*sine);
            result.derivatives[2*k] = finite_result(angular*cosine);
        }
        return result;
    }
    if (config.size == 1) return result;

    // Half-sums avoid overflow in Jacobi's valid, very large parameters.
    const double half_sum = config.kind == BasisKind::Jacobi ?
                            0.5*config.alpha + 0.5*config.beta : 0;
    // Preserve distance from the admissible boundary alpha,beta > -1.
    // Adding one after summing the parameters can erase that distance.
    const double shifted_half_sum = config.kind == BasisKind::Jacobi ?
                                   0.5*(config.alpha+1) + 0.5*(config.beta+1) : 0;
    const double half_difference = config.kind == BasisKind::Jacobi ?
                                   0.5*config.alpha - 0.5*config.beta : 0;
    if (config.kind == BasisKind::Jacobi && (x == -1 || x == 1)) {
        // DLMF 18.6.T1: endpoint rising-factorial values. The general
        // three-term recurrence can cancel a small endpoint against huge terms.
        const double parameter = x == 1 ? config.alpha : config.beta;
        double endpoint_value = 1;
        double shifted_endpoint_value = 1;
        for (std::size_t k = 1; k < config.size; ++k) {
            const double n = static_cast<double>(k);
            endpoint_value = finite_result(endpoint_value*((parameter+n)/n));
            result.values[k] = x < 0 && k % 2 ? -endpoint_value : endpoint_value;
            if (k > 1)
                shifted_endpoint_value = finite_result(shifted_endpoint_value*((parameter+n)/(n-1)));
            // DLMF 18.9.E15: P'_n = (n+alpha+beta+1)/2 *
            // P_(n-1)^(alpha+1,beta+1). Half-sums keep a finite slope
            // representable even when alpha+beta would overflow.
            const double derivative = finite_result((shifted_half_sum+0.5*(n-1))*shifted_endpoint_value);
            result.derivatives[k] = x < 0 && k % 2 == 0 ? -derivative : derivative;
        }
        return result;
    }
    double first_slope = 1;
    double first_offset = 0;
    if (config.kind == BasisKind::Hermite) first_slope = 2;
    if (config.kind == BasisKind::Jacobi) {
        first_slope = shifted_half_sum;
        first_offset = half_difference;
    }
    result.values[1] = finite_result(first_slope*x + first_offset);
    result.derivatives[1] = finite_result(first_slope);

    for (std::size_t k = 1; k < config.size-1; ++k) {
        const double n = static_cast<double>(k);
        double a = 0;
        double b = 0;
        double c = 0;
        switch (config.kind) {
        case BasisKind::Chebyshev:
            a = 2;
            c = 1;
            break;
        case BasisKind::Legendre:
            a = (2*n+1)/(n+1);
            c = n/(n+1);
            break;
        case BasisKind::Hermite:
            a = 2;
            c = 2*n;
            break;
        case BasisKind::Jacobi: {
            // NIST DLMF 18.9.2, rearranged into ratios to avoid squaring
            // large parameters. P1 above handles alpha+beta = -1 or 0.
            const double t = shifted_half_sum + (n-1);
            const double denominator_half = shifted_half_sum + 0.5*(n-1);
            a = ((t+0.5)/(n+1))*((t+1)/denominator_half);
            b = (half_difference/(n+1))*(half_sum/t)*((t+0.5)/denominator_half);
            c = 0.5*((n+config.alpha)/(n+1))*((n+config.beta)/t)*((t+1)/denominator_half);
            break;
        }
        default:
            throw std::logic_error("validated polynomial kind expected");
        }
        const double factor = a*x+b;
        result.values[k+1] = finite_result(factor*result.values[k] - c*result.values[k-1]);
        result.derivatives[k+1] = finite_result(a*result.values[k] +
            factor*result.derivatives[k] - c*result.derivatives[k-1]);
    }
    return result;
}
}
