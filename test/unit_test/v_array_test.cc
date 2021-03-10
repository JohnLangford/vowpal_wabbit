#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include <cstddef>
#include <algorithm>

#include "v_array.h"

BOOST_AUTO_TEST_CASE(v_array_size_is_const)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);

  const auto& const_list = list;
  BOOST_CHECK_EQUAL(std::size_t(2), const_list.size());
}

BOOST_AUTO_TEST_CASE(v_array_empty_is_const)
{
  const v_array<int> list;
  BOOST_CHECK_EQUAL(true, list.empty());
}

BOOST_AUTO_TEST_CASE(v_array_dereference)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);

  BOOST_CHECK_EQUAL(std::size_t(2), list[1]);
}

BOOST_AUTO_TEST_CASE(v_array_clear)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
  list.clear();
  BOOST_CHECK_EQUAL(std::size_t(0), list.size());
}

BOOST_AUTO_TEST_CASE(v_array_copy)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());

  v_array<int> list2 = list;
  BOOST_CHECK_EQUAL(std::size_t(2), list2.size());

  v_array<int> list3(list);
  BOOST_CHECK_EQUAL(std::size_t(2), list3.size());
}

BOOST_AUTO_TEST_CASE(v_array_move)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());

  v_array<int> list2 = std::move(list);
  BOOST_CHECK_EQUAL(std::size_t(2), list2.size());

  v_array<int> list3(std::move(list2));
  BOOST_CHECK_EQUAL(std::size_t(2), list3.size());
}

BOOST_AUTO_TEST_CASE(v_array_pop_back)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
  list.pop_back();
  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  list.pop_back();
  BOOST_CHECK_EQUAL(std::size_t(0), list.size());
  BOOST_CHECK_EQUAL(true, list.empty());
}

BOOST_AUTO_TEST_CASE(v_array_find_exists)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  list.push_back(6);
  list.push_back(4);

  auto it = std::find(list.begin(), list.end(), 2);
  BOOST_CHECK_NE(it, list.end());
  BOOST_CHECK_EQUAL(it, list.begin() + 1);
}

BOOST_AUTO_TEST_CASE(v_array_find_not_exists)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  list.push_back(6);
  list.push_back(4);

  auto it = std::find(list.begin(), list.end(), 11);
  BOOST_CHECK_EQUAL(it, list.end());
}

BOOST_AUTO_TEST_CASE(v_array_back)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(2, list.back());
}

BOOST_AUTO_TEST_CASE(v_array_erase_single_element_single_element_array)
{
  v_array<int> list;
  list.push_back(1);
  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  list.erase(list.begin());
  BOOST_CHECK_EQUAL(std::size_t(0), list.size());
}

BOOST_AUTO_TEST_CASE(v_array_erase_single_element_reuse_array)
{
  v_array<int> list;
  list.push_back(1);
  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  list.erase(list.begin());
  BOOST_CHECK_EQUAL(std::size_t(0), list.size());
  list.push_back(2);
  list.push_back(3);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
}

BOOST_AUTO_TEST_CASE(v_array_erase_range)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  list.push_back(3);
  list.push_back(4);
  BOOST_CHECK_EQUAL(std::size_t(4), list.size());
  auto it = list.erase(list.begin() + 1, list.begin() + 3);
  BOOST_CHECK_EQUAL(*it, 4);
  BOOST_CHECK_EQUAL(list[0], 1);
  BOOST_CHECK_EQUAL(list[1], 4);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
}

BOOST_AUTO_TEST_CASE(v_array_erase_range_zero_width)
{
  v_array<int> list;
  list.push_back(1);
  list.push_back(2);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
  list.erase(list.begin() + 1, list.begin() + 1);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
}

BOOST_AUTO_TEST_CASE(v_array_erase_last_element)
{
  v_array<int> list;
  list.push_back(5);
  list.push_back(3);
  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
  auto it = list.erase(list.begin() + 1);
  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  BOOST_CHECK_EQUAL(list[0], 5);
  BOOST_CHECK_EQUAL(it, list.end());
}

BOOST_AUTO_TEST_CASE(v_array_insert_from_empty)
{
  v_array<int> list;
  list.insert(list.begin(), 1);

  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  BOOST_CHECK_EQUAL(1, list[0]);
}

BOOST_AUTO_TEST_CASE(v_array_insert_check_iterator)
{
  v_array<int> list;
  auto it = list.insert(list.begin(), 47);

  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  BOOST_CHECK_EQUAL(47, list[0]);
  BOOST_CHECK_EQUAL(47, *it);
}

BOOST_AUTO_TEST_CASE(v_array_insert_end_iterator)
{
  v_array<int> list;
  auto it = list.insert(list.end(), 22);

  BOOST_CHECK_EQUAL(std::size_t(1), list.size());
  BOOST_CHECK_EQUAL(22, list[0]);
  BOOST_CHECK_EQUAL(22, *it);
}

BOOST_AUTO_TEST_CASE(v_array_insert_multiple_insert)
{
  v_array<int> list;
  list.insert(list.begin(), 1);
  list.insert(list.end(), 2);

  BOOST_CHECK_EQUAL(std::size_t(2), list.size());
  BOOST_CHECK_EQUAL(1, list[0]);
  BOOST_CHECK_EQUAL(2, list[1]);

  list.insert(list.begin() + 1, 33);
  BOOST_CHECK_EQUAL(std::size_t(3), list.size());
  BOOST_CHECK_EQUAL(1, list[0]);
  BOOST_CHECK_EQUAL(33, list[1]);
  BOOST_CHECK_EQUAL(2, list[2]);

  list.insert(list.begin(), 8);
  BOOST_CHECK_EQUAL(std::size_t(4), list.size());
  BOOST_CHECK_EQUAL(8, list[0]);
  BOOST_CHECK_EQUAL(1, list[1]);
  BOOST_CHECK_EQUAL(33, list[2]);
  BOOST_CHECK_EQUAL(2, list[3]);

  list.insert(list.begin() + 2, 13);
  BOOST_CHECK_EQUAL(std::size_t(5), list.size());
  BOOST_CHECK_EQUAL(8, list[0]);
  BOOST_CHECK_EQUAL(1, list[1]);
  BOOST_CHECK_EQUAL(13, list[2]);
  BOOST_CHECK_EQUAL(33, list[3]);
  BOOST_CHECK_EQUAL(2, list[4]);

  list.insert(list.begin() + 1, 41);
  BOOST_CHECK_EQUAL(std::size_t(6), list.size());
  BOOST_CHECK_EQUAL(8, list[0]);
  BOOST_CHECK_EQUAL(41, list[1]);
  BOOST_CHECK_EQUAL(1, list[2]);
  BOOST_CHECK_EQUAL(13, list[3]);
  BOOST_CHECK_EQUAL(33, list[4]);
  BOOST_CHECK_EQUAL(2, list[5]);
}

BOOST_AUTO_TEST_CASE(v_array_insert_in_loop)
{
  v_array<int> list;
  const auto num_values_to_insert = 1000;
  for (auto i = 0; i < num_values_to_insert; i++) { list.insert(list.begin(), i); }

  BOOST_CHECK_EQUAL(std::size_t(num_values_to_insert), list.size());

  for (auto i = 0; i < num_values_to_insert; i++) { BOOST_CHECK_EQUAL(num_values_to_insert - i - 1, list[i]); }
}
