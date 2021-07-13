#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include "namespaced_feature_store.h"

BOOST_AUTO_TEST_CASE(namespaced_features_test)
{
  VW::namespaced_feature_store feature_groups;
  BOOST_CHECK(feature_groups.empty());

  BOOST_CHECK(feature_groups.indices().empty());

  feature_groups.get_or_create('a', 123);
  feature_groups.get_or_create('a', 1234);

  BOOST_CHECK_EQUAL(feature_groups.indices().size(), 1);

  BOOST_CHECK(feature_groups.indices()[0] == 'a');

  feature_groups.remove('a', 123);
  BOOST_CHECK_EQUAL(feature_groups.indices().size(), 1);

  BOOST_CHECK(feature_groups.indices()[0] == 'a');
  feature_groups.remove('a', 1234);
  BOOST_CHECK(feature_groups.indices().size() == 0);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_test)
{
  VW::namespaced_feature_store feature_groups;

  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create('a', 1234);
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);

  auto it = feature_groups.index_flat_begin('a');
  BOOST_CHECK(it != feature_groups.index_flat_end('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 1);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.index_flat_end('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 2);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.index_flat_end('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 3);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.index_flat_end('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 4);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it == feature_groups.index_flat_end('a'));
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_loop_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create('a', 1234);
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create('a', 12345);
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  size_t counter = 0;
  auto it = feature_groups.index_flat_begin('a');
  auto end = feature_groups.index_flat_end('a');
  for (; it != end; ++it) { counter++; }
  BOOST_REQUIRE_EQUAL(counter, 8);

  counter = 0;
  it = feature_groups.index_flat_begin('b');
  end = feature_groups.index_flat_end('b');
  for (; it != end; ++it) { counter++; }
  BOOST_REQUIRE_EQUAL(counter, 0);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_empty_test)
{
  VW::namespaced_feature_store feature_groups;

  auto it = feature_groups.index_flat_begin('a');
  BOOST_CHECK(it == feature_groups.index_flat_end('a'));
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_difference_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create('a', 1234);
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create('a', 12345);
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  auto it = feature_groups.index_flat_begin('a');
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 0);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 1);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 2);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 3);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 4);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 5);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 6);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.index_flat_begin('a'), 7);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_difference_end_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  fs1.push_back(1.0, 3);
  fs1.push_back(1.0, 4);

  auto it = feature_groups.index_flat_begin('a');
  auto it_end = feature_groups.index_flat_end('a');
  BOOST_REQUIRE_EQUAL(it_end - it, 4);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 3);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 2);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 1);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 0);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_difference_empty_group_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  fs1.push_back(1.0, 3);
  fs1.push_back(1.0, 4);
  auto& fs2 = feature_groups.get_or_create('a', 1234);

  auto it = feature_groups.index_flat_begin('a');
  auto it_end = feature_groups.index_flat_end('a');
  BOOST_REQUIRE_EQUAL(it_end - it, 4);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 3);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 2);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 1);
  ++it;
  BOOST_REQUIRE_EQUAL(it_end - it, 0);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_advance_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create('a', 1234);
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create('a', 12345);
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  auto it = feature_groups.index_flat_begin('a');
  BOOST_REQUIRE_EQUAL((*it).index(), 1);
  it += 4;
  BOOST_REQUIRE_EQUAL((*it).index(), 5);
  it += 3;
  BOOST_REQUIRE_EQUAL((*it).index(), 8);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_advance_to_end_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);

  auto it = feature_groups.index_flat_begin('a');
  it += 2;
  BOOST_REQUIRE(it == feature_groups.index_flat_end('a'));
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_advance_to_end_two_groups_test)
{
  VW::namespaced_feature_store feature_groups;
  auto& fs1 = feature_groups.get_or_create('a', 123);
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create('a', 1234);
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);

  auto it = feature_groups.index_flat_begin('a');
  it += 6;
  BOOST_REQUIRE(it == feature_groups.index_flat_end('a'));
}
