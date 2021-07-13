// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include <cstdint>
#include "constant.h"
#include "feature_group.h"
#include "interactions.h"
#include "example_predict.h"
#include <vector>
#include <string>

const static std::pair<std::string, std::string> EMPTY_AUDIT_STRINGS = std::make_pair("", "");

namespace INTERACTIONS
{
/*
 * By default include interactions of feature with itself.
 * This approach produces slightly more interactions but it's safer
 * for some cases, as discussed in issues/698
 * Previous behaviour was: include interactions of feature with itself only if its value != value^2.
 *
 */

// 3 template functions to pass FuncT() proper argument (feature idx in regressor, or its coefficient)

template <class DataT, void (*FuncT)(DataT&, const float, float&), class WeightsT>
inline void call_FuncT(DataT& dat, WeightsT& weights, const float ft_value, const uint64_t ft_idx)
{
  FuncT(dat, ft_value, weights[ft_idx]);
}

template <class DataT, void (*FuncT)(DataT&, const float, float), class WeightsT>
inline void call_FuncT(DataT& dat, const WeightsT& weights, const float ft_value, const uint64_t ft_idx)
{
  FuncT(dat, ft_value, weights[ft_idx]);
}

template <class DataT, void (*FuncT)(DataT&, float, uint64_t), class WeightsT>
inline void call_FuncT(DataT& dat, WeightsT& /*weights*/, const float ft_value, const uint64_t ft_idx)
{
  FuncT(dat, ft_value, ft_idx);
}

// state data used in non-recursive feature generation algorithm
// contains N feature_gen_data records (where N is length of interaction)
struct feature_gen_data
{
  uint64_t hash = 0;              // hash of feature interactions of previous namespaces in the list
  float x = 1.f;                  // value of feature interactions of previous namespaces in the list
                                  // more than once calculated at preprocessing together with same_ns
  bool self_interaction = false;  // namespace interacting with itself

  namespace_index ns_idx;
  VW::chained_proxy_iterator<VW::namespaced_feature_store::list_iterator, features::audit_iterator> current_it;
  VW::chained_proxy_iterator<VW::namespaced_feature_store::list_iterator, features::audit_iterator> end_it;

  feature_gen_data(namespace_index ns_idx,
      VW::chained_proxy_iterator<VW::namespaced_feature_store::list_iterator, features::audit_iterator> begin,
      VW::chained_proxy_iterator<VW::namespaced_feature_store::list_iterator, features::audit_iterator> end)
      : ns_idx(ns_idx), current_it(begin), end_it(end)
  {
  }
};

// The inline function below may be adjusted to change the way
// synthetic (interaction) features' values are calculated, e.g.,
// fabs(value1-value2) or even value1>value2?1.0:-1.0
// Beware - its result must be non-zero.
inline float INTERACTION_VALUE(float value1, float value2) { return value1 * value2; }

// uncomment line below to disable usage of inner 'for' loops for pair and triple interactions
// end switch to usage of non-recursive feature generation algorithm for interactions of any length

// #define GEN_INTER_LOOP

template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT), bool audit,
    void (*audit_func)(DataT&, const audit_strings*), class WeightsT, typename FeatureIteratorT>
inline void inner_kernel(DataT& dat, FeatureIteratorT& begin, FeatureIteratorT& end, const uint64_t offset,
    WeightsT& weights, feature_value ft_value, feature_index halfhash)
{
  if (audit)
  {
    for (; begin != end; ++begin)
    {
      audit_func(dat, (*begin).audit() == nullptr ? &EMPTY_AUDIT_STRINGS : (*begin).audit());
      call_FuncT<DataT, FuncT>(
          dat, weights, INTERACTION_VALUE(ft_value, (*begin).value()), ((*begin).index() ^ halfhash) + offset);
      audit_func(dat, nullptr);
    }
  }
  else
  {
    for (; begin != end; ++begin)
      call_FuncT<DataT, FuncT>(
          dat, weights, INTERACTION_VALUE(ft_value, (*begin).value()), ((*begin).index() ^ halfhash) + offset);
  }
}

// this templated function generates new features for given example and set of interactions
// and passes each of them to given function FuncT()
// it must be in header file to avoid compilation problems
template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT), bool audit,
    void (*audit_func)(DataT&, const audit_strings*),
    class WeightsT>  // nullptr func can't be used as template param in old compilers
inline void generate_interactions(const std::vector<std::vector<namespace_index>>& interactions, bool permutations,
    example_predict& ec, DataT& dat, WeightsT& weights,
    size_t& num_features)  // default value removed to eliminate ambiguity in old complers
{
  num_features = 0;

  // often used values
  const uint64_t offset = ec.ft_offset;
  //    const uint64_t stride_shift = all.stride_shift; // it seems we don't need stride shift in FTRL-like hash

  // statedata for generic non-recursive iteration
  std::vector<feature_gen_data> state_data;

  for (const auto& ns : interactions)
  {  // current list of namespaces to interact.

#ifndef GEN_INTER_LOOP

    // unless GEN_INTER_LOOP is defined we use nested 'for' loops for interactions length 2 (pairs) and 3 (triples)
    // and generic non-recursive algorythm for all other cases.
    // nested 'for' loops approach is faster, but can't be used for interation of any length.
    const size_t len = ns.size();

    if (len == 2)  // special case of pairs
    {
      size_t ns0_i = 0;
      for (auto& first : ec.feature_space.get_list(ns[0]))
      {
        ns0_i++;
        if (first.feats.nonempty())
        {
          size_t ns1_i = 0;
          for (auto& second : ec.feature_space.get_list(ns[1]))
          {
            ns1_i++;
            if (second.feats.nonempty())
            {
              const bool same_namespace_index = (!permutations && (ns[0] == ns[1]));
              const bool same_namespace_hash = (!permutations && (first.hash == second.hash));

              // When there is more than one feature group of the same index, we should not process permutations. For
              // example we skip x2*x1 but do process x1*x2.
              if (same_namespace_index && (ns1_i < ns0_i)) { continue; }

              for (size_t i = 0; i < first.feats.indicies.size(); ++i)
              {
                feature_index halfhash = FNV_prime * static_cast<uint64_t>(first.feats.indicies[i]);
                if (audit)
                {
                  audit_func(
                      dat, i < first.feats.space_names.size() ? &first.feats.space_names[i] : &EMPTY_AUDIT_STRINGS);
                }
                // next index differs for permutations and simple combinations
                feature_value ft_value = first.feats.values[i];
                auto begin = second.feats.audit_cbegin();
                if (same_namespace_hash) { begin += (PROCESS_SELF_INTERACTIONS(ft_value)) ? i : i + 1; }
                auto end = second.feats.audit_cend();
                num_features += std::distance(begin, end);
                inner_kernel<DataT, WeightOrIndexT, FuncT, audit, audit_func>(
                    dat, begin, end, offset, weights, ft_value, halfhash);

                if (audit) audit_func(dat, nullptr);
              }  // end for(fst)
            }    // end if (data[snd] size > 0)
          }
        }  // end if (data[fst] size > 0)
      }
    }
    else if (len == 3)  // special case for triples
    {
      size_t ns0_i = 0;
      for (auto& first : ec.feature_space.get_list(ns[0]))
      {
        ns0_i++;
        if (first.feats.nonempty())
        {
          size_t ns1_i = 0;
          for (auto& second : ec.feature_space.get_list(ns[1]))
          {
            ns1_i++;
            if (second.feats.nonempty())
            {
              size_t ns2_i = 0;
              for (auto& third : ec.feature_space.get_list(ns[2]))
              {
                ns2_i++;
                if (third.feats.nonempty())
                {  // don't compare 1 and 3 as interaction is sorted
                  const bool same_namespace1 = (!permutations && (ns[0] == ns[1]));
                  const bool same_namespace1_hash = (!permutations && (first.hash == second.hash));

                  const bool same_namespace2 = (!permutations && (ns[1] == ns[2]));
                  const bool same_namespace2_hash = (!permutations && (second.hash == third.hash));

                  // We check it for skipping self interaction.
                  const bool same_namespace3 = (!permutations && (ns[0] == ns[2]));
                  const bool same_namespace3_hash = (!permutations && (first.hash == third.hash));

                  // When there is more than one feature group of the same index, we should not process permutations.
                  // For example we skip x2*x1 but do process x1*x2.
                  // TODO - ensure this cubic version works
                  if (same_namespace1 && (ns1_i < ns0_i)) { continue; }
                  if (same_namespace2 && (ns2_i < ns1_i)) { continue; }
                  if (same_namespace3 && (ns2_i < ns0_i)) { continue; }

                  for (size_t i = 0; i < first.feats.indicies.size(); ++i)
                  {
                    if (audit)
                    {
                      audit_func(
                          dat, i < first.feats.space_names.size() ? &first.feats.space_names[i] : &EMPTY_AUDIT_STRINGS);
                    }
                    const uint64_t halfhash1 = FNV_prime * static_cast<uint64_t>(first.feats.indicies[i]);
                    float first_ft_value = first.feats.values[i];
                    size_t j = 0;
                    if (same_namespace1_hash)  // next index differs for permutations and simple combinations
                      j = (PROCESS_SELF_INTERACTIONS(first_ft_value)) ? i : i + 1;

                    for (; j < second.feats.indicies.size(); ++j)
                    {  // f3 x k*(f2 x k*f1)
                      if (audit)
                      {
                        audit_func(dat,
                            j < second.feats.space_names.size() ? &second.feats.space_names[j] : &EMPTY_AUDIT_STRINGS);
                      }
                      feature_index halfhash =
                          FNV_prime * (halfhash1 ^ static_cast<uint64_t>(second.feats.indicies[j]));
                      feature_value ft_value = INTERACTION_VALUE(first_ft_value, second.feats.values[j]);

                      auto begin = third.feats.audit_cbegin();
                      // next index differs for permutations and simple combinations
                      if (same_namespace2_hash) { begin += (PROCESS_SELF_INTERACTIONS(ft_value)) ? j : j + 1; }
                      auto end = third.feats.audit_cend();
                      num_features += std::distance(begin, end);
                      inner_kernel<DataT, WeightOrIndexT, FuncT, audit, audit_func>(
                          dat, begin, end, offset, weights, ft_value, halfhash);
                      if (audit) audit_func(dat, nullptr);
                    }  // end for (snd)
                    if (audit) audit_func(dat, nullptr);
                  }  // end for (fst)

                }  // end if (data[thr] size > 0)
              }
            }  // end if (data[snd] size > 0)
          }
        }  // end if (data[fst] size > 0)
      }
    }
    else
#endif
    {
      state_data.clear();
      bool must_skip_interaction = false;
      // preparing state data
      for (auto n : ns)
      {
        auto begin = ec.feature_space.index_flat_begin(n);
        auto end = ec.feature_space.index_flat_end(n);

        if (begin == end)
        {
          must_skip_interaction = true;
          break;
        }
        state_data.push_back(
            feature_gen_data{n, ec.feature_space.index_flat_begin(n), ec.feature_space.index_flat_end(n)});
      }

      // if any of interacting namespace has 0 features - whole interaction is skipped
      if (must_skip_interaction) continue;  // no_data_to_interact

      auto fgd = state_data.data();                             // always equal to first ns
      auto fgd2 = state_data.data() + (state_data.size() - 1);  // always equal to last ns
      if (!permutations)                                        // adjust state_data for simple combinations
      {  // if permutations mode is disabeled then namespaces in ns are already sorted and thus grouped
        // (in fact, currently they are sorted even for enabled permutations mode)
        // let's go throw the list and calculate number of features to skip in namespaces which
        // repeated more than once to generate only simple combinations of features

        // size_t margin = 0;  // number of features to ignore if namespace has been seen before

        // iterate list backward as margin grows in this order

        for (fgd = state_data.data() + (state_data.size() - 1); fgd > state_data.data(); --fgd)
        {
          fgd2 = fgd - 1;
          fgd->self_interaction =
              (fgd->current_it == fgd2->current_it);  // state_data.begin().self_interaction is always false

          // Removing this section for now as it greatly simplifies things and is never actually used.
          // This is because PROCESS_SELF_INTERACTIONS is true and renders this a noop
          // if (fgd->self_interaction)
          // {
          //   size_t& loop_end = fgd2->loop_end;
          //   if (!PROCESS_SELF_INTERACTIONS(1.f /*unused_value*/)
          //   {
          //     ++margin;  // otherwise margin can't be increased
          //     must_skip_interaction = loop_end < margin;
          //     if (must_skip_interaction) break;
          //   }

          //   if (margin != 0) loop_end -= margin;  // skip some features and increase margin
          // }
          // else if (margin != 0)
          //   margin = 0;
        }

        // if impossible_without_permutations == true then we faced with case like interaction 'aaaa'
        // where namespace 'a' contains less than 4 unique features. It's impossible to make simple
        // combination of length 4 without repetitions from 3 or less elements.
        if (must_skip_interaction) continue;  // impossible_without_permutations
      }                                       // end of state_data adjustment

      fgd = state_data.data();                             // always equal to first ns
      fgd2 = state_data.data() + (state_data.size() - 1);  // always equal to last ns

      // beware: micro-optimization.
      /* start & end are always point to features in last namespace of interaction.
      for 'all.permutations == true' they are constant.*/
      // size_t start_i = 0;

      feature_gen_data* cur_data = fgd;
      // end of micro-optimization block

      // generic feature generation cycle for interactions of any length
      bool do_it = true;
      while (do_it)
      {
        if (cur_data < fgd2)  // can go further through the list of namespaces in interaction
        {
          feature_gen_data* next_data = cur_data + 1;

          if (next_data->self_interaction)
          {  // if next namespace is same, we should start with loop_idx + 1 to avoid feature interaction with itself
            // unless feature has value x and x != x*x. E.g. x != 0 and x != 1. Features with x == 0 are already
            // filtered out in parce_args.cc::maybeFeature().
            auto current_offset = cur_data->current_it - ec.feature_space.index_flat_begin(cur_data->ns_idx);
            current_offset += PROCESS_SELF_INTERACTIONS((*cur_data->current_it).value()) ? 0 : 1;
            next_data->current_it = ec.feature_space.index_flat_begin(next_data->ns_idx);
            next_data->current_it += current_offset;
          }
          else
          {
            next_data->current_it = ec.feature_space.index_flat_begin(next_data->ns_idx);
          }

          if (audit) audit_func(dat, (*cur_data->current_it).audit());
          if (cur_data == fgd)  // first namespace
          {
            next_data->hash = FNV_prime * (*cur_data->current_it).index();
            next_data->x = (*cur_data->current_it).value();  // data->x == 1.
          }
          else
          {  // feature2 xor (16777619*feature1)
            next_data->hash = FNV_prime * (cur_data->hash ^ (*cur_data->current_it).index());
            next_data->x = INTERACTION_VALUE((*cur_data->current_it).value(), cur_data->x);
          }

          ++cur_data;
        }
        else
        {
          // last namespace - iterate its features and go back
          // start value is not a constant in this case
          size_t start_i = 0;
          if (!permutations) { start_i = fgd2->current_it - ec.feature_space.index_flat_begin(fgd2->ns_idx); }

          feature_value ft_value = fgd2->x;
          feature_index halfhash = fgd2->hash;

          auto begin = ec.feature_space.index_flat_begin(cur_data->ns_idx);
          begin += start_i;

          auto end = ec.feature_space.index_flat_end(cur_data->ns_idx);

          num_features += (end - begin);
          inner_kernel<DataT, WeightOrIndexT, FuncT, audit, audit_func, WeightsT>(
              dat, begin, end, offset, weights, ft_value, halfhash);

          // trying to go back increasing loop_idx of each namespace by the way

          bool go_further = true;

          do
          {
            --cur_data;
            ++cur_data->current_it;
            go_further = cur_data->current_it == cur_data->end_it;
            if (audit) audit_func(dat, nullptr);
          } while (go_further && cur_data != fgd);

          do_it = !(cur_data == fgd && go_further);
          // if do_it==false - we've reached 0 namespace but its 'cur_data.loop_idx > cur_data.loop_end' -> exit the
          // while loop
        }  // if last namespace
      }    // while do_it
    }
  }  // foreach interaction in all.interactions
}  // namespace INTERACTIONS

}  // namespace INTERACTIONS
