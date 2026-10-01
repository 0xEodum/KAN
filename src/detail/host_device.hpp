#pragma once

// Shared numerical formulas are compiled by the host C++ compiler for the CPU
// backend and by nvcc for device kernels. Each backend keeps its own math
// library: host code calls <cmath>, device code calls CUDA's global overloads.

#include <cmath>

#if defined(__CUDACC__)
#define KAN_HOST_DEVICE __host__ __device__
#else
#define KAN_HOST_DEVICE
#endif

// Per-edge formulas are called once per sample and edge; host compilers do not
// reliably inline the nested templates, which costs ~30% on low-degree edges.
#if defined(__CUDACC__)
#define KAN_FORCE_INLINE __forceinline__
#elif defined(_MSC_VER)
#define KAN_FORCE_INLINE __forceinline
#else
#define KAN_FORCE_INLINE inline __attribute__((always_inline))
#endif

#if defined(__CUDA_ARCH__)
#define KAN_MATH_NAMESPACE
#else
#define KAN_MATH_NAMESPACE std
#endif

namespace kan::detail::math {

// Smallest positive normal double; numeric_limits is not callable on device.
inline constexpr double min_normal = 2.2250738585072014e-308;
// ln 2 rounded to double (0x3FE62E42FEFA39EF), equal to a correctly rounded
// log(2.0). A literal keeps device code from evaluating log at run time.
inline constexpr double ln2 = 0.693147180559945309417232121458;

KAN_HOST_DEVICE inline bool finite(double v) { return KAN_MATH_NAMESPACE::isfinite(v); }
KAN_HOST_DEVICE inline bool signbit(double v) { return KAN_MATH_NAMESPACE::signbit(v); }
KAN_HOST_DEVICE inline bool tiny(double v) { return KAN_MATH_NAMESPACE::fabs(v) < min_normal; }
KAN_HOST_DEVICE inline double abs(double v) { return KAN_MATH_NAMESPACE::fabs(v); }
KAN_HOST_DEVICE inline double exp(double v) { return KAN_MATH_NAMESPACE::exp(v); }
KAN_HOST_DEVICE inline double log(double v) { return KAN_MATH_NAMESPACE::log(v); }
KAN_HOST_DEVICE inline double sqrt(double v) { return KAN_MATH_NAMESPACE::sqrt(v); }
KAN_HOST_DEVICE inline double acos(double v) { return KAN_MATH_NAMESPACE::acos(v); }
KAN_HOST_DEVICE inline double cos(double v) { return KAN_MATH_NAMESPACE::cos(v); }
KAN_HOST_DEVICE inline double sin(double v) { return KAN_MATH_NAMESPACE::sin(v); }
KAN_HOST_DEVICE inline double copysign(double magnitude, double sign) {
    return KAN_MATH_NAMESPACE::copysign(magnitude, sign);
}

} // namespace kan::detail::math

#undef KAN_MATH_NAMESPACE
