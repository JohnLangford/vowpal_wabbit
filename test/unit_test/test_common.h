#pragma once

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include <vector>

#include "action_score.h"

constexpr float FLOAT_TOL = 0.0001f;

inline void compare(float l, float r, float tol) { BOOST_CHECK_CLOSE(l, r, tol); }
inline void compare(const ACTION_SCORE::action_score& l, const ACTION_SCORE::action_score& r, float float_tolerance)
{
  BOOST_CHECK_EQUAL(l.action, r.action);
  BOOST_CHECK_CLOSE(l.score, r.score, float_tolerance);
}

template <typename ContainerOneT, typename ContainerTwoT>
void check_collections_with_float_tolerance(const ContainerOneT& lhs, const ContainerTwoT& rhs, float float_tolerance)
{
  BOOST_CHECK_EQUAL(lhs.size(), rhs.size());
  auto l = std::begin(lhs);
  auto r = std::begin(rhs);
  for (; l < std::end(lhs); ++l, ++r)
  {
    compare(*l, *r, float_tolerance);
  }
}

template <template <typename...> class ContainerOneT, template <typename...> class ContainerTwoT, typename T>
void check_collections_exact(const ContainerOneT<T>& lhs, const ContainerTwoT<T>& rhs)
{
  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs));
}
