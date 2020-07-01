#pragma once

#include <limits>
#include <array>
#include <cstdint>

#include "future_compat.h"

const constexpr int8_t VALUES_BELOW_ZERO = 44;
const constexpr int8_t VALUES_ABOVE_ZERO = 38;

static constexpr std::array<float, VALUES_BELOW_ZERO + 1 + VALUES_ABOVE_ZERO> pow_10_lookup_table = {
    1e-44f,
    1e-43f,
    1e-42f,
    1e-41f,
    1e-40f,
    1e-39f,
    1e-38f,
    1e-37f,
    1e-36f,
    1e-35f,
    1e-34f,
    1e-33f,
    1e-32f,
    1e-31f,
    1e-30f,
    1e-29f,
    1e-28f,
    1e-27f,
    1e-26f,
    1e-25f,
    1e-24f,
    1e-23f,
    1e-22f,
    1e-21f,
    1e-20f,
    1e-19f,
    1e-18f,
    1e-17f,
    1e-16f,
    1e-15f,
    1e-14f,
    1e-13f,
    1e-12f,
    1e-11f,
    1e-10f,
    1e-9f,
    1e-8f,
    1e-7f,
    1e-6f,
    1e-5f,
    1e-4f,
    1e-3f,
    1e-2f,
    1e-1f,
    1e0f,
    1e1f,
    1e2f,
    1e3f,
    1e4f,
    1e5f,
    1e6f,
    1e7f,
    1e8f,
    1e9f,
    1e10f,
    1e11f,
    1e12f,
    1e13f,
    1e14f,
    1e15f,
    1e16f,
    1e17f,
    1e18f,
    1e19f,
    1e20f,
    1e21f,
    1e22f,
    1e23f,
    1e24f,
    1e25f,
    1e26f,
    1e27f,
    1e28f,
    1e29f,
    1e30f,
    1e31f,
    1e32f,
    1e33f,
    1e34f,
    1e35f,
    1e36f,
    1e37f,
    1e38f,
};

namespace VW
{
// std::array::operator[] is made constexpr in C++17, so this can only be guaranteed to be constexpr when this standard
// is used.
VW_STD17_CONSTEXPR inline float fast_pow10(int8_t pow)
{
  // If the power would be above the range float can represent, return inf.
  return pow > VALUES_ABOVE_ZERO ? std::numeric_limits<float>::infinity()
                                 : (pow < -1 * VALUES_BELOW_ZERO) ? 0.f : pow_10_lookup_table[static_cast<size_t>(pow) + static_cast<size_t>(VALUES_BELOW_ZERO)];
}

// std::array::operator[] is made constexpr in C++17, so this can only be guaranteed to be constexpr when this standard
// is used. Undefined behavior if pow is > 38 or < -45
VW_STD17_CONSTEXPR inline float fast_pow10_unsafe(int8_t pow)
{
  return pow_10_lookup_table[static_cast<size_t>(pow) + static_cast<size_t>(VALUES_BELOW_ZERO)];
}
}  // namespace VW
