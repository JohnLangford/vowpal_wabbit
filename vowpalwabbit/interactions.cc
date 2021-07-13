// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "interactions.h"

#include "vw_math.h"
#include "v_array.h"

#include <algorithm>
#include <vector>
#include <sstream>
#include <cfloat>
#include <algorithm>
#include <iterator>

using namespace VW::config;

namespace INTERACTIONS
{
/*
 *  Estimation of generated features properties
 */

// returns number of new features that will be generated for example and sum of their squared values
void eval_count_of_generated_ft(bool permutations, const std::vector<std::vector<namespace_index>>& interactions,
    const VW::namespaced_feature_store& feature_spaces, size_t& new_features_cnt, float& new_features_value)
{
  new_features_cnt = 0;
  new_features_value = 0.;

  v_array<float> results;

  if (permutations)
  {
    // just multiply precomputed values for all namespacess
    for (const auto& inter : interactions)
    {
      size_t num_features_in_inter = 1;
      float sum_feat_sq_in_inter = 1.;

      for (namespace_index ns : inter)
      {
        for (const auto& ns_fs : feature_spaces.get_list(ns))
        {
          num_features_in_inter *= ns_fs.feats.size();
          sum_feat_sq_in_inter *= ns_fs.feats.sum_feat_sq;
          // If there are no features, then we don't want to accumulate the default value of 1.0, so we zero out here.
          if (num_features_in_inter == 0) { sum_feat_sq_in_inter = 0; }
        }
      }

      new_features_cnt += num_features_in_inter;
      new_features_value += sum_feat_sq_in_inter;
    }
  }
  else  // case of simple combinations
  {
    for (const auto& inter : interactions)
    {
      size_t num_features_in_inter = 1;
      float sum_feat_sq_in_inter = 1.;

      for (auto ns = inter.begin(); ns != inter.end(); ++ns)
      {
        for (const auto& ns_fs : feature_spaces.get_list(*ns))
        {
          if ((ns == inter.end() - 1) || (*ns != *(ns + 1)))  // neighbour namespaces are different
          {
            // just multiply precomputed values
            num_features_in_inter *= ns_fs.feats.size();
            sum_feat_sq_in_inter *= ns_fs.feats.sum_feat_sq;
            if (num_features_in_inter == 0) break;  // one of namespaces has no features - go to next interaction
          }
          else  // we are at beginning of a block made of same namespace (interaction is preliminary sorted)
          {
            // let's find out real length of this block

            // already compared ns == ns+1
            size_t order_of_inter = 2;

            for (auto ns_end = ns + 2; ns_end < inter.end(); ++ns_end)
              if (*ns == *ns_end) ++order_of_inter;

            // count number of features with value != 1.;
            size_t cnt_ft_value_non_1 = 0;

            // in this block we shall calculate number of generated features and sum of their values
            // keeping in mind rules applicable for simple combinations instead of permutations

            // let's calculate sum of their squared value for whole block

            // ensure results as big as order_of_inter and empty.
            results.resize_but_with_stl_behavior(order_of_inter);
            std::fill(results.begin(), results.end(), 0.f);

            // recurrent value calculations
            for (size_t i = 0; i < ns_fs.feats.size(); ++i)
            {
              const float x = ns_fs.feats.values[i] * ns_fs.feats.values[i];

              if (!PROCESS_SELF_INTERACTIONS(fs.values[i]))
              {
                for (size_t j = order_of_inter - 1; j > 0; --j) results[j] += results[j - 1] * x;

                results[0] += x;
              }
              else
              {
                results[0] += x;

                for (size_t j = 1; j < order_of_inter; ++j) results[j] += results[j - 1] * x;

                ++cnt_ft_value_non_1;
              }
            }

            sum_feat_sq_in_inter *= results[order_of_inter - 1];  // will be explained in http://bit.ly/1Hk9JX1

            // let's calculate  the number of a new features

            // if number of features is less than  order of interaction then go to the next interaction
            // as you can't make simple combination of interaction 'aaa' if a contains < 3 features.
            // unless one of them has value != 1. and we are counting them.
            const size_t ft_size = ns_fs.feats.size();
            if (cnt_ft_value_non_1 == 0 && ft_size < order_of_inter)
            {
              num_features_in_inter = 0;
              break;
            }

            size_t n;
            if (cnt_ft_value_non_1 == 0)  // number of generated simple combinations is C(n,k)
            {
              n = static_cast<size_t>(
                  VW::math::choose(static_cast<int64_t>(ft_size), static_cast<int64_t>(order_of_inter)));
            }
            else
            {
              n = 0;
              for (size_t l = 0; l <= order_of_inter; ++l)
              {
                // C(l+m-1, l) * C(n-m, k-l)
                size_t num = (l == 0) ? 1 : static_cast<size_t>(VW::math::choose(l + cnt_ft_value_non_1 - 1, l));

                if (ft_size - cnt_ft_value_non_1 >= order_of_inter - l)
                  num *= static_cast<size_t>(VW::math::choose(ft_size - cnt_ft_value_non_1, order_of_inter - l));
                else
                  num = 0;

                n += num;
              }

            }  // details on http://bit.ly/1Hk9JX1

            num_features_in_inter *= n;

            ns += order_of_inter - 1;  // jump over whole block
          }
        }
      }

      if (num_features_in_inter == 0) continue;  // signal that values should be ignored (as default value is 1)

      new_features_cnt += num_features_in_inter;
      new_features_value += sum_feat_sq_in_inter;
    }
  }
}

bool sort_interactions_comparator(const std::vector<namespace_index>& a, const std::vector<namespace_index>& b)
{
  if (a.size() != b.size()) { return a.size() > b.size(); }
  for (size_t i = 0; i < a.size(); i++)
  {
    if (a[i] < b[i])
      return true;
    else if (a[i] == b[i])
      continue;
    else
      return false;
  }
  return false;
}

/*
 *   Sorting and filtering duplicate interactions
 */

// returns true if iteraction contains one or more duplicated namespaces
// with one exeption - returns false if interaction made of one namespace
// like 'aaa' as it has no sense to sort such things.

inline bool must_be_left_sorted(const std::vector<namespace_index>& oi)
{
  if (oi.size() <= 1) return true;  // one letter in std::string - no need to sort

  bool diff_ns_found = false;
  bool pair_found = false;

  for (auto i = std::begin(oi); i != std::end(oi) - 1; ++i)
    if (*i == *(i + 1))  // pair found
    {
      if (diff_ns_found) return true;  // case 'abb'
      pair_found = true;
    }
    else
    {
      if (pair_found) return true;  // case 'aab'
      diff_ns_found = true;
    }

  return false;  // 'aaa' or 'abc'
}

std::vector<std::vector<namespace_index>> expand_quadratics_wildcard_interactions(
    bool leave_duplicate_interactions, const std::set<namespace_index>& new_example_indices)
{
  std::set<std::vector<namespace_index>> interactions;

  for (auto it = new_example_indices.begin(); it != new_example_indices.end(); ++it)
  {
    auto idx1 = *it;
    interactions.insert({idx1, idx1});

    for (auto jt = it; jt != new_example_indices.end(); ++jt)
    {
      auto idx2 = *jt;
      interactions.insert({idx1, idx2});
      interactions.insert({idx2, idx2});
      if (leave_duplicate_interactions) { interactions.insert({idx2, idx1}); }
    }
  }
  return std::vector<std::vector<namespace_index>>(interactions.begin(), interactions.end());
}

// used from parse_args.cc
// filter duplicate namespaces treating them as unordered sets of namespaces.
// also sort namespaces in interactions containing duplicate namespaces to make sure they are grouped together.

void sort_and_filter_duplicate_interactions(
    std::vector<std::vector<namespace_index>>& vec, bool filter_duplicates, size_t& removed_cnt, size_t& sorted_cnt)
{
  // 2 out parameters
  removed_cnt = 0;
  sorted_cnt = 0;

  // interaction value sort + original position
  std::vector<std::pair<std::vector<namespace_index>, size_t>> vec_sorted;
  for (size_t i = 0; i < vec.size(); ++i)
  {
    std::vector<namespace_index> sorted_i(vec[i]);
    std::stable_sort(std::begin(sorted_i), std::end(sorted_i));
    vec_sorted.push_back(std::make_pair(sorted_i, i));
  }

  if (filter_duplicates)
  {
    // remove duplicates
    std::stable_sort(vec_sorted.begin(), vec_sorted.end(),
        [](std::pair<std::vector<namespace_index>, size_t> const& a,
            std::pair<std::vector<namespace_index>, size_t> const& b) { return a.first < b.first; });
    auto last = unique(vec_sorted.begin(), vec_sorted.end(),
        [](std::pair<std::vector<namespace_index>, size_t> const& a,
            std::pair<std::vector<namespace_index>, size_t> const& b) { return a.first == b.first; });
    vec_sorted.erase(last, vec_sorted.end());

    // report number of removed interactions
    removed_cnt = vec.size() - vec_sorted.size();

    // restore original order
    std::stable_sort(vec_sorted.begin(), vec_sorted.end(),
        [](std::pair<std::vector<namespace_index>, size_t> const& a,
            std::pair<std::vector<namespace_index>, size_t> const& b) { return a.second < b.second; });
  }

  // we have original vector and vector with duplicates removed + corresponding indexes in original vector
  // plus second vector's data is sorted. We can reuse it if we need interaction to be left sorted.
  // let's make a new vector from these two sources - without dulicates and with sorted data whenever it's needed.
  std::vector<std::vector<namespace_index>> res;
  for (auto& i : vec_sorted)
  {
    if (must_be_left_sorted(i.first))
    {
      // if so - copy sorted data to result
      res.push_back(i.first);
      ++sorted_cnt;
    }
    else  // else - move unsorted data to result
      res.push_back(vec[i.second]);
  }

  vec = res;
}

std::vector<namespace_index> indices_to_values_one_based(
    const std::vector<size_t>& indices, const std::set<namespace_index>& values)
{
  std::vector<namespace_index> result;
  result.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); i++)
  {
    auto it = values.begin();
    std::advance(it, indices[i] - 1);
    result.push_back(*it);
  }
  return result;
}

std::vector<namespace_index> indices_to_values_ignore_last_index(
    const std::vector<size_t>& indices, const std::set<namespace_index>& values)
{
  std::vector<namespace_index> result;
  result.reserve(indices.size() - 1);
  for (size_t i = 0; i < indices.size() - 1; i++)
  {
    auto it = values.begin();
    std::advance(it, indices[i]);
    result.push_back(*it);
  }
  return result;
}

std::vector<std::vector<namespace_index>> generate_namespace_combinations_with_repetition(
    const std::set<namespace_index>& namespaces, size_t num_to_pick)
{
  std::vector<std::vector<namespace_index>> result;
  // This computation involves factorials and so can only be done with relatively small inputs.
  // Factorial 22 would result in 64 bit overflow.
  if ((namespaces.size() + num_to_pick) <= 21)
  { result.reserve(VW::math::number_of_combinations_with_repetition(namespaces.size(), num_to_pick)); }

  auto last_index = namespaces.size() - 1;
  // last index is used to signal when done
  std::vector<size_t> indices(num_to_pick + 1, 0);
  while (true)
  {
    for (size_t i = 0; i < num_to_pick; ++i)
    {
      if (indices[i] > last_index)
      {
        // Increment the next index
        indices[i + 1] += 1;
        // Decrement all past indices
        for (int k = static_cast<int>(i); k >= 0; --k) { indices[static_cast<size_t>(k)] = indices[i + 1]; }
      }
    }

    if (indices[num_to_pick] > 0) break;
    result.emplace_back(indices_to_values_ignore_last_index(indices, namespaces));

    indices[0] += 1;
  }

  return result;
}

std::vector<std::vector<namespace_index>> generate_namespace_permutations_with_repetition(
    const std::set<namespace_index>& namespaces, size_t num_to_pick)
{
  std::vector<std::vector<namespace_index>> result;
  result.reserve(VW::math::number_of_permutations_with_repetition(namespaces.size(), num_to_pick));

  std::vector<size_t> one_based_chosen_indices(num_to_pick, 0);
  for (size_t i = 0; i < num_to_pick - 1; i++) { one_based_chosen_indices[i] = 1; }
  one_based_chosen_indices[num_to_pick - 1] = 0;

  size_t number_of_namespaces = namespaces.size();
  size_t next_index = num_to_pick;

  while (true)
  {
    if (one_based_chosen_indices[next_index - 1] == number_of_namespaces)
    {
      next_index--;
      if (next_index == 0) { break; }
    }
    else
    {
      one_based_chosen_indices[next_index - 1]++;
      while (next_index < num_to_pick)
      {
        next_index++;
        one_based_chosen_indices[next_index - 1] = 1;
      }

      result.emplace_back(indices_to_values_one_based(one_based_chosen_indices, namespaces));
    }
  }

  return result;
}

}  // namespace INTERACTIONS
