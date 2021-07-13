#ifndef STATIC_LINK_VW
#  define BOOST_TEST_DYN_LINK
#endif

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <array>
#include <vector>

#include "test_common.h"
#include "vw.h"
#include "gd_predict.h"
#include "gd.h"
#include "interactions.h"

struct eval_gen_data
{
  size_t& new_features_cnt;
  float& new_features_value;
  eval_gen_data(size_t& features_cnt, float& features_value)
      : new_features_cnt(features_cnt), new_features_value(features_value)
  {
  }
};

void ft_cnt(eval_gen_data& dat, const float fx, const uint64_t)
{
  ++dat.new_features_cnt;
  dat.new_features_value += fx * fx;
}

// eval_count_of_generated_ft_naive() is an alternative way of implementation of
// eval_count_of_generated_ft() it just calls generate_interactions() with small
// function which counts generated features and sums their squared values. We
// use it to validate the with more fast (?) analytic solution
template <INTERACTIONS::generate_func_t generate_func, bool leave_duplicate_interactions>
void eval_count_of_generated_ft_naive(vw& all, example_predict& ec, size_t& new_features_cnt, float& new_features_value)
{
  // Only makes sense to do this when not in permutations mode.
  assert(!all.permutations);

  new_features_cnt = 0;
  new_features_value = 0.;

  auto interactions = INTERACTIONS::compile_interactions<generate_func, leave_duplicate_interactions>(
      all.interactions, std::set<namespace_index>(ec.indices.begin(), ec.indices.end()));

  v_array<float> results;

  eval_gen_data dat(new_features_cnt, new_features_value);
  size_t ignored = 0;
  ec.interactions = &interactions;
  INTERACTIONS::generate_interactions<eval_gen_data, uint64_t, ft_cnt, false, nullptr>(all, ec, dat, ignored);
  ec.interactions = &all.interactions;
}

inline void noop_func(float& unused_dat, const float ft_weight, const uint64_t ft_idx) {}

BOOST_AUTO_TEST_CASE(eval_count_of_generated_ft_test)
{
  auto& vw = *VW::initialize("--quiet -q :: --noconstant", nullptr, false, nullptr, nullptr);
  auto* ex = VW::read_example(vw, "3 |f a b c |e x y z");

  size_t naive_features_count;
  float naive_features_value;
  eval_count_of_generated_ft_naive<INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
      vw, *ex, naive_features_count, naive_features_value);

  auto interactions =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          vw.interactions, std::set<namespace_index>(ex->indices.begin(), ex->indices.end()));
  ex->interactions = &interactions;
  size_t fast_features_count;
  float fast_features_value;
  INTERACTIONS::eval_count_of_generated_ft(
      vw.permutations, *ex->interactions, ex->feature_space, fast_features_count, fast_features_value);
  ex->interactions = &vw.interactions;

  BOOST_CHECK_EQUAL(naive_features_count, fast_features_count);
  BOOST_CHECK_CLOSE(naive_features_value, fast_features_value, FLOAT_TOL);

  // Prediction will count the interacted features, so we can compare that too.
  vw.predict(*ex);
  BOOST_CHECK_EQUAL(naive_features_count, ex->num_features_from_interactions);
  VW::finish_example(vw, *ex);
  VW::finish(vw);
}

BOOST_AUTO_TEST_CASE(eval_count_of_generated_ft_permuations_test)
{
  auto& vw = *VW::initialize(
      "--quiet -q :: --leave_duplicate_interactions --permutations --noconstant", nullptr, false, nullptr, nullptr);
  auto* ex = VW::read_example(vw, "3 |f a b c |e x y z");

  std::set<namespace_index> indices(ex->indices.begin(), ex->indices.end());
  auto interactions =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_permutations_with_repetition, true>(
          vw.interactions, indices);
  ex->interactions = &interactions;
  size_t fast_features_count;
  float fast_features_value;
  INTERACTIONS::eval_count_of_generated_ft(
      vw.permutations, *ex->interactions, ex->feature_space, fast_features_count, fast_features_value);
  ex->interactions = &vw.interactions;

  vw.predict(*ex);
  BOOST_CHECK_EQUAL(fast_features_count, ex->num_features_from_interactions);

  VW::finish_example(vw, *ex);
  VW::finish(vw);
}

BOOST_AUTO_TEST_CASE(interaction_generic_expand_wildcard_only)
{
  std::set<namespace_index> namespaces = {'a', 'b'};
  auto result = INTERACTIONS::generate_namespace_combinations_with_repetition(namespaces, 2);

  std::vector<std::vector<namespace_index>> compare_set = {{'b', 'a'}, {'a', 'a'}, {'b', 'b'}};

  std::sort(compare_set.begin(), compare_set.end());
  std::sort(result.begin(), result.end());
  check_vector_of_vectors_exact(result, compare_set);
}

BOOST_AUTO_TEST_CASE(interaction_generic_with_duplicates_expand_wildcard_only)
{
  std::set<namespace_index> namespaces = {'a', 'b'};
  auto result = INTERACTIONS::generate_namespace_permutations_with_repetition(namespaces, 2);

  std::vector<std::vector<namespace_index>> compare_set = {{'b', 'a'}, {'a', 'b'}, {'a', 'a'}, {'b', 'b'}};

  std::sort(compare_set.begin(), compare_set.end());
  std::sort(result.begin(), result.end());
  check_vector_of_vectors_exact(result, compare_set);
}

BOOST_AUTO_TEST_CASE(sort_and_filter_interactions)
{
  std::vector<std::vector<namespace_index>> input = {{'b', 'a'}, {'a', 'b', 'a'}, {'a', 'a'}, {'b', 'b'}};

  size_t removed_count = 0;
  size_t sorted_count = 0;
  INTERACTIONS::sort_and_filter_duplicate_interactions(input, false, removed_count, sorted_count);

  std::vector<std::vector<namespace_index>> compare_set = {{'b', 'a'}, {'a', 'a', 'b'}, {'a', 'a'}, {'b', 'b'}};
  check_vector_of_vectors_exact(input, compare_set);
}

void sort_all(std::vector<std::vector<namespace_index>>& interactions)
{
  for (auto& interaction : interactions) { std::sort(interaction.begin(), interaction.end()); }
  std::sort(interactions.begin(), interactions.end());
}

BOOST_AUTO_TEST_CASE(compile_interactions_quadratic_permutations_and_combinations_same)
{
  std::set<namespace_index> indices = {'a', 'b', 'c', 'd'};
  std::vector<std::vector<namespace_index>> interactions = {{':', 'a'}};

  // Permutations implies leave duplicate interactions (second template arg)
  auto result_perms =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_permutations_with_repetition, true>(
          interactions, indices);
  auto result_combs =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          interactions, indices);

  std::vector<std::vector<namespace_index>> compare_set = {{'a', 'a'}, {'b', 'a'}, {'c', 'a'}, {'d', 'a'}};

  sort_all(compare_set);
  sort_all(result_perms);
  sort_all(result_combs);
  check_vector_of_vectors_exact(result_perms, compare_set);
  check_vector_of_vectors_exact(result_combs, compare_set);
}

BOOST_AUTO_TEST_CASE(compile_interactions_quadratic_combinations)
{
  std::set<namespace_index> indices = {'a', 'b', 'c', 'd'};
  std::vector<std::vector<namespace_index>> interactions = {{':', ':'}};

  auto result =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          interactions, indices);

  std::vector<std::vector<namespace_index>> compare_set = {{'a', 'a'}, {'a', 'b'}, {'a', 'c'}, {'a', 'd'}, {'b', 'b'},
      {'b', 'c'}, {'b', 'd'}, {'c', 'c'}, {'c', 'd'}, {'d', 'd'}};

  sort_all(compare_set);
  sort_all(result);
  check_vector_of_vectors_exact(result, compare_set);
}

BOOST_AUTO_TEST_CASE(compile_interactions_quadratic_permutations)
{
  std::set<namespace_index> indices = {'a', 'b', 'c', 'd'};
  std::vector<std::vector<namespace_index>> interactions = {{':', ':'}};

  auto result = INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_permutations_with_repetition, true>(
      interactions, indices);

  std::vector<std::vector<namespace_index>> compare_set = {{'a', 'a'}, {'a', 'b'}, {'a', 'c'}, {'a', 'd'}, {'b', 'a'},
      {'b', 'b'}, {'b', 'c'}, {'b', 'd'}, {'c', 'a'}, {'c', 'b'}, {'c', 'c'}, {'c', 'd'}, {'d', 'a'}, {'d', 'b'},
      {'d', 'c'}, {'d', 'd'}};

  sort_all(compare_set);
  sort_all(result);
  check_vector_of_vectors_exact(result, compare_set);
}

BOOST_AUTO_TEST_CASE(compile_interactions_cubic_combinations)
{
  std::set<namespace_index> indices = {'a', 'b', 'c', 'd'};
  std::vector<std::vector<namespace_index>> interactions = {{':', ':', ':'}};

  auto result =
      INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_combinations_with_repetition, false>(
          interactions, indices);

  std::vector<std::vector<namespace_index>> compare_set = {
      {'a', 'a', 'a'},
      {'a', 'a', 'b'},
      {'a', 'a', 'c'},
      {'a', 'a', 'd'},
      {'a', 'b', 'b'},
      {'a', 'b', 'c'},
      {'a', 'b', 'd'},
      {'a', 'c', 'c'},
      {'a', 'c', 'd'},
      {'a', 'd', 'd'},
      {'b', 'b', 'b'},
      {'b', 'b', 'c'},
      {'b', 'b', 'd'},
      {'b', 'c', 'c'},
      {'b', 'c', 'd'},
      {'b', 'd', 'd'},
      {'c', 'c', 'c'},
      {'c', 'c', 'd'},
      {'c', 'd', 'd'},
      {'d', 'd', 'd'},
  };

  sort_all(compare_set);
  sort_all(result);
  check_vector_of_vectors_exact(result, compare_set);
}

BOOST_AUTO_TEST_CASE(compile_interactions_cubic_permutations)
{
  std::set<namespace_index> indices = {'a', 'b', 'c', 'd'};
  std::vector<std::vector<namespace_index>> interactions = {{':', ':', ':'}};

  auto result = INTERACTIONS::compile_interactions<INTERACTIONS::generate_namespace_permutations_with_repetition, true>(
      interactions, indices);

  std::vector<std::vector<namespace_index>> compare_set = {
      {'a', 'a', 'a'}, {'a', 'a', 'b'}, {'a', 'a', 'c'}, {'a', 'a', 'd'}, {'a', 'b', 'a'}, {'a', 'b', 'b'},
      {'a', 'b', 'c'}, {'a', 'b', 'd'}, {'a', 'c', 'a'}, {'a', 'c', 'b'}, {'a', 'c', 'c'}, {'a', 'c', 'd'},
      {'a', 'd', 'a'}, {'a', 'd', 'b'}, {'a', 'd', 'c'}, {'a', 'd', 'd'}, {'b', 'a', 'a'}, {'b', 'a', 'b'},
      {'b', 'a', 'c'}, {'b', 'a', 'd'}, {'b', 'b', 'a'}, {'b', 'b', 'b'}, {'b', 'b', 'c'}, {'b', 'b', 'd'},
      {'b', 'c', 'a'}, {'b', 'c', 'b'}, {'b', 'c', 'c'}, {'b', 'c', 'd'}, {'b', 'd', 'a'}, {'b', 'd', 'b'},
      {'b', 'd', 'c'}, {'b', 'd', 'd'}, {'c', 'a', 'a'}, {'c', 'a', 'b'}, {'c', 'a', 'c'}, {'c', 'a', 'd'},
      {'c', 'b', 'a'}, {'c', 'b', 'b'}, {'c', 'b', 'c'}, {'c', 'b', 'd'}, {'c', 'c', 'a'}, {'c', 'c', 'b'},
      {'c', 'c', 'c'}, {'c', 'c', 'd'}, {'c', 'd', 'a'}, {'c', 'd', 'b'}, {'c', 'd', 'c'}, {'c', 'd', 'd'},
      {'d', 'a', 'a'}, {'d', 'a', 'b'}, {'d', 'a', 'c'}, {'d', 'a', 'd'}, {'d', 'b', 'a'}, {'d', 'b', 'b'},
      {'d', 'b', 'c'}, {'d', 'b', 'd'}, {'d', 'c', 'a'}, {'d', 'c', 'b'}, {'d', 'c', 'c'}, {'d', 'c', 'd'},
      {'d', 'd', 'a'}, {'d', 'd', 'b'}, {'d', 'd', 'c'}, {'d', 'd', 'd'}

  };

  sort_all(compare_set);
  sort_all(result);
  check_vector_of_vectors_exact(result, compare_set);
}
