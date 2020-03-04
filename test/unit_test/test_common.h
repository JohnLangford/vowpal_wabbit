#pragma once

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include <vector>

#include "action_score.h"

constexpr float FLOAT_TOL = 0.0001f;

template <template<typename...> typename ContainerOneT, template<typename...> typename ContainerTwoT>
void inline check_collections(const ContainerOneT<float>& lhs, const ContainerTwoT<float>& rhs, float tolerance_percent) {
  BOOST_CHECK_EQUAL(lhs.size(), rhs.size());
  for (auto l = std::begin(lhs), r = std::begin(rhs); l < std::end(lhs); ++l, ++r) {
    BOOST_CHECK_CLOSE(*l, *r, tolerance_percent);
  }
}

template <template<typename...> typename ContainerOneT, template<typename...> typename ContainerTwoT>
void inline check_collections(const ContainerOneT<ACTION_SCORE::action_score>& lhs, const ContainerTwoT<ACTION_SCORE::action_score>& rhs, float tolerance_percent) {
  BOOST_CHECK_EQUAL(lhs.size(), rhs.size());
  for (auto l = std::begin(lhs), r = std::begin(rhs); l < std::end(lhs); ++l, ++r) {
    BOOST_CHECK_EQUAL(l->action, r->action);
    BOOST_CHECK_CLOSE(l->score, r->score, tolerance_percent);
  }
}

template <template <typename...> typename ContainerOneT, template <typename...> typename ContainerTwoT, typename T>
void check_collections(const ContainerOneT<T>& lhs, const ContainerTwoT<T>& rhs) {
  BOOST_CHECK_EQUAL_COLLECTIONS(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs));
}
