#pragma once
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test {
inline std::vector<std::pair<std::string, std::function<void()>>>& cases() {
    static std::vector<std::pair<std::string, std::function<void()>>> value;
    return value;
}
struct Register {
    Register(const char* name, std::function<void()> fn) { cases().emplace_back(name, fn); }
};
inline void require(bool condition, const char* expression) {
    if (!condition) throw std::runtime_error(expression);
}
inline void near(double actual, double expected, double tolerance = 1e-10) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected)))
        throw std::runtime_error("actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
template<class Exception, class Fn> void throws(Fn fn) {
    try { fn(); } catch (const Exception&) { return; }
    throw std::runtime_error("expected exception was not thrown");
}
inline int run() {
    int failed = 0;
    for (const auto& [name, fn] : cases()) {
        try { fn(); std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& ex) { ++failed; std::cout << "FAIL " << name << ": " << ex.what() << '\n'; }
    }
    std::cout << cases().size() - static_cast<std::size_t>(failed) << '/' << cases().size() << " passed\n";
    return failed == 0 ? 0 : 1;
}
} // namespace test
#define TEST(name) void name(); static test::Register reg_##name(#name, name); void name()
#define REQUIRE(expression) test::require(static_cast<bool>(expression), #expression)
