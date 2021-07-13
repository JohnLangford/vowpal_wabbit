#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include "test_common.h"

#include "namespaced_feature_store.h"

BOOST_AUTO_TEST_CASE(namespaced_features_test)
{
  VW::namespaced_features feature_groups;
  BOOST_CHECK(feature_groups.empty());

  auto begin_end = feature_groups.get_namespace_index_groups('a');
  BOOST_CHECK(begin_end.second - begin_end.first == 0);

  feature_groups.get_or_create_feature_group(123, 'a');
  feature_groups.get_or_create_feature_group(1234, 'a');

  begin_end = feature_groups.get_namespace_index_groups('a');
  BOOST_CHECK(begin_end.second - begin_end.first == 2);

  BOOST_REQUIRE_THROW(feature_groups[1], VW::vw_exception);
  BOOST_REQUIRE_NO_THROW(feature_groups[123]);

  check_collections_exact(feature_groups.get_indices(), std::set<namespace_index>{'a'});

  feature_groups.remove_feature_group(123);
  begin_end = feature_groups.get_namespace_index_groups('a');
  BOOST_CHECK(begin_end.second - begin_end.first == 1);

  check_collections_exact(feature_groups.get_indices(), std::set<namespace_index>{'a'});
  feature_groups.remove_feature_group(1234);
  check_collections_exact(feature_groups.get_indices(), std::set<namespace_index>{});
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_test)
{
  VW::namespaced_features feature_groups;

  auto& fs1 = feature_groups.get_or_create_feature_group(123, 'a');
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create_feature_group(1234, 'a');
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);

  auto it = feature_groups.namespace_index_begin_proxy('a');
  BOOST_CHECK(it != feature_groups.namespace_index_end_proxy('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 1);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.namespace_index_end_proxy('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 2);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.namespace_index_end_proxy('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 3);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it != feature_groups.namespace_index_end_proxy('a'));
  BOOST_REQUIRE_EQUAL((*it).index(), 4);
  BOOST_REQUIRE_CLOSE((*it).value(), 1.f, FLOAT_TOL);
  ++it;

  BOOST_CHECK(it == feature_groups.namespace_index_end_proxy('a'));
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_loop_test)
{
  VW::namespaced_features feature_groups;
  auto& fs1 = feature_groups.get_or_create_feature_group(123, 'a');
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create_feature_group(1234, 'a');
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create_feature_group(12345, 'a');
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  size_t counter = 0;
  auto it = feature_groups.namespace_index_begin_proxy('a');
  auto end = feature_groups.namespace_index_end_proxy('a');
  for (; it != end; ++it) { counter++; }
  BOOST_REQUIRE_EQUAL(counter, 8);

  counter = 0;
  it = feature_groups.namespace_index_begin_proxy('b');
  end = feature_groups.namespace_index_end_proxy('b');
  for (; it != end; ++it) { counter++; }
  BOOST_REQUIRE_EQUAL(counter, 0);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_empty_test)
{
  VW::namespaced_features feature_groups;

  auto it = feature_groups.namespace_index_begin_proxy('a');
  BOOST_CHECK(it == feature_groups.namespace_index_end_proxy('a'));
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_difference_test)
{
  VW::namespaced_features feature_groups;
  auto& fs1 = feature_groups.get_or_create_feature_group(123, 'a');
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create_feature_group(1234, 'a');
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create_feature_group(12345, 'a');
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  auto it = feature_groups.namespace_index_begin_proxy('a');
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 0);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 1);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 2);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 3);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 4);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 5);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 6);
  ++it;
  BOOST_REQUIRE_EQUAL(it - feature_groups.namespace_index_begin_proxy('a'), 7);
}

BOOST_AUTO_TEST_CASE(namespaced_features_proxy_iterator_difference_end_test)
{
  VW::namespaced_features feature_groups;
  auto& fs1 = feature_groups.get_or_create_feature_group(123, 'a');
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  fs1.push_back(1.0, 3);
  fs1.push_back(1.0, 4);

  auto it = feature_groups.namespace_index_begin_proxy('a');
  auto it_end = feature_groups.namespace_index_end_proxy('a');
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
  VW::namespaced_features feature_groups;
  auto& fs1 = feature_groups.get_or_create_feature_group(123, 'a');
  fs1.push_back(1.0, 1);
  fs1.push_back(1.0, 2);
  auto& fs2 = feature_groups.get_or_create_feature_group(1234, 'a');
  fs2.push_back(1.0, 3);
  fs2.push_back(1.0, 4);
  fs2.push_back(1.0, 5);
  fs2.push_back(1.0, 6);
  auto& fs3 = feature_groups.get_or_create_feature_group(12345, 'a');
  fs3.push_back(1.0, 7);
  fs3.push_back(1.0, 8);

  auto it = feature_groups.namespace_index_begin_proxy('a');
  BOOST_REQUIRE_EQUAL((*it).index(), 1);
  it += 4;
  BOOST_REQUIRE_EQUAL((*it).index(), 5);
  it += 3;
  BOOST_REQUIRE_EQUAL((*it).index(), 8);
}