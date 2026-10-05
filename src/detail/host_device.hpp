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

// The same constants per scalar type (backlog C1): the shared formulas are
// instantiated for double (CPU, FP64 resident) and float (FP32 resident).
template<class Scalar> struct constants;
template<> struct constants<double> {
    static constexpr double min_normal = math::min_normal;
    static constexpr double ln2 = math::ln2;
};
template<> struct constants<float> {
    static constexpr float min_normal = 1.17549435082228751e-38f; // FLT_MIN
    static constexpr float ln2 = 0.693147180559945309417232121458f;
};

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
KAN_HOST_DEVICE inline double tanh(double v) { return KAN_MATH_NAMESPACE::tanh(v); }
KAN_HOST_DEVICE inline double cosh(double v) { return KAN_MATH_NAMESPACE::cosh(v); }
KAN_HOST_DEVICE inline double copysign(double magnitude, double sign) {
    return KAN_MATH_NAMESPACE::copysign(magnitude, sign);
}

// Single-precision overloads: device code calls the CUDA f-suffixed
// functions explicitly, so no float argument is promoted to double.
#if defined(__CUDA_ARCH__)
#define KAN_FLOAT_MATH(name) ::name##f
#else
#define KAN_FLOAT_MATH(name) std::name
#endif
KAN_HOST_DEVICE inline bool finite(float v) { return KAN_MATH_NAMESPACE::isfinite(v); }
KAN_HOST_DEVICE inline bool signbit(float v) { return KAN_MATH_NAMESPACE::signbit(v); }
KAN_HOST_DEVICE inline bool tiny(float v) { return KAN_FLOAT_MATH(fabs)(v) < constants<float>::min_normal; }
KAN_HOST_DEVICE inline float abs(float v) { return KAN_FLOAT_MATH(fabs)(v); }
KAN_HOST_DEVICE inline float exp(float v) { return KAN_FLOAT_MATH(exp)(v); }
KAN_HOST_DEVICE inline float log(float v) { return KAN_FLOAT_MATH(log)(v); }
KAN_HOST_DEVICE inline float sqrt(float v) { return KAN_FLOAT_MATH(sqrt)(v); }
KAN_HOST_DEVICE inline float acos(float v) { return KAN_FLOAT_MATH(acos)(v); }
KAN_HOST_DEVICE inline float cos(float v) { return KAN_FLOAT_MATH(cos)(v); }
KAN_HOST_DEVICE inline float sin(float v) { return KAN_FLOAT_MATH(sin)(v); }
KAN_HOST_DEVICE inline float tanh(float v) { return KAN_FLOAT_MATH(tanh)(v); }
KAN_HOST_DEVICE inline float cosh(float v) { return KAN_FLOAT_MATH(cosh)(v); }
KAN_HOST_DEVICE inline float copysign(float magnitude, float sign) {
    return KAN_FLOAT_MATH(copysign)(magnitude, sign);
}
#undef KAN_FLOAT_MATH

} // namespace kan::detail::math

#undef KAN_MATH_NAMESPACE
