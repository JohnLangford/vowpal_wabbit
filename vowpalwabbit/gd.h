// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "parse_regressor.h"
#include "constant.h"
#include "interactions.h"
#include "array_parameters.h"
#include "gd_predict.h"
#include "vw_math.h"

namespace GD
{
VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

struct gd;

float finalize_prediction(shared_data* sd, vw_logger& logger, float ret);
void print_features(vw& all, example& ec);
void print_audit_features(vw&, example& ec);
void save_load_regressor(vw& all, io_buf& model_file, bool read, bool text);
void save_load_online_state(vw& all, io_buf& model_file, bool read, bool text, double& total_weight,
    GD::gd* g = nullptr, uint32_t ftrl_size = 0);

template <class T>
struct multipredict_info
{
  size_t count;
  size_t step;
  polyprediction* pred;
  const T& weights; /* & for l1: */
  float gravity;
};

template <class T>
inline void vec_add_multipredict(multipredict_info<T>& mp, const float fx, uint64_t fi)
{
  if ((-1e-10 < fx) && (fx < 1e-10)) return;
  uint64_t mask = mp.weights.mask();
  polyprediction* p = mp.pred;
  fi &= mask;
  uint64_t top = fi + (uint64_t)((mp.count - 1) * mp.step);
  uint64_t i = 0;
  if (top <= mask)
  {
    i += fi;
    for (; i <= top; i += mp.step, ++p)
    {
      p->scalar += fx * mp.weights[i];  // TODO: figure out how to use
                                        // weight_parameters::iterator (not using
                                        // change_begin())
    }
  }
  else  // TODO: this could be faster by unrolling into two loops
    for (size_t c = 0; c < mp.count; ++c, fi += (uint64_t)mp.step, ++p)
    {
      fi &= mask;
      p->scalar += fx * mp.weights[fi];
    }
}

// iterate through one namespace (or its part), callback function FuncT(some_data_R, feature_value_x, feature_weight)
template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT)>
inline void foreach_feature(vw& all, example& ec, DataT& dat)
{
  return all.weights.sparse
      ? foreach_feature<DataT, WeightOrIndexT, FuncT, sparse_parameters>(all.weights.sparse_weights,
            all.ignore_some_linear, all.ignore_linear, *ec.interactions, all.permutations, ec, dat)
      : foreach_feature<DataT, WeightOrIndexT, FuncT, dense_parameters>(all.weights.dense_weights,
            all.ignore_some_linear, all.ignore_linear, *ec.interactions, all.permutations, ec, dat);
}

// iterate through one namespace (or its part), callback function FuncT(some_data_R, feature_value_x, feature_weight)
template <class DataT, class WeightOrIndexT, void (*FuncT)(DataT&, float, WeightOrIndexT)>
inline void foreach_feature(vw& all, example& ec, DataT& dat, size_t& num_interacted_features)
{
  return all.weights.sparse ? foreach_feature<DataT, WeightOrIndexT, FuncT, sparse_parameters>(
                                  all.weights.sparse_weights, all.ignore_some_linear, all.ignore_linear,
                                  *ec.interactions, all.permutations, ec, dat, num_interacted_features)
                            : foreach_feature<DataT, WeightOrIndexT, FuncT, dense_parameters>(all.weights.dense_weights,
                                  all.ignore_some_linear, all.ignore_linear, *ec.interactions, all.permutations, ec,
                                  dat, num_interacted_features);
}

// iterate through all namespaces and quadratic&cubic features, callback function T(some_data_R, feature_value_x,
// feature_weight)
template <class DataT, void (*FuncT)(DataT&, float, float&)>
inline void foreach_feature(vw& all, example& ec, DataT& dat)
{
  foreach_feature<DataT, float&, FuncT>(all, ec, dat);
}

template <class DataT, void (*FuncT)(DataT&, float, float)>
inline void foreach_feature(vw& all, example& ec, DataT& dat)
{
  foreach_feature<DataT, float, FuncT>(all, ec, dat);
}

template <class DataT, void (*FuncT)(DataT&, float, float&)>
inline void foreach_feature(vw& all, example& ec, DataT& dat, size_t& num_interacted_features)
{
  foreach_feature<DataT, float&, FuncT>(all, ec, dat, num_interacted_features);
}

template <class DataT, void (*FuncT)(DataT&, float, const float&)>
inline void foreach_feature(vw& all, example& ec, DataT& dat, size_t& num_interacted_features)
{
  foreach_feature<DataT, const float&, FuncT>(all, ec, dat, num_interacted_features);
}

inline float inline_predict(vw& all, example& ec)
{
  const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  return all.weights.sparse
      ? inline_predict<sparse_parameters>(all.weights.sparse_weights, all.ignore_some_linear, all.ignore_linear,
            *ec.interactions, all.permutations, ec, simple_red_features.initial)
      : inline_predict<dense_parameters>(all.weights.dense_weights, all.ignore_some_linear, all.ignore_linear,
            *ec.interactions, all.permutations, ec, simple_red_features.initial);
}

inline float inline_predict(vw& all, example& ec, size_t& num_generated_features)
{
  const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  return all.weights.sparse
      ? inline_predict<sparse_parameters>(all.weights.sparse_weights, all.ignore_some_linear, all.ignore_linear,
            *ec.interactions, all.permutations, ec, num_generated_features, simple_red_features.initial)
      : inline_predict<dense_parameters>(all.weights.dense_weights, all.ignore_some_linear, all.ignore_linear,
            *ec.interactions, all.permutations, ec, num_generated_features, simple_red_features.initial);
}

inline float trunc_weight(const float w, const float gravity)
{
  return (gravity < fabsf(w)) ? w - VW::math::sign(w) * gravity : 0.f;
}

}  // namespace GD

namespace INTERACTIONS
{
template <class R, class S, void (*T)(R&, float, S), bool audit, void (*audit_func)(R&, const audit_strings*)>
inline void generate_interactions(vw& all, example_predict& ec, R& dat, size_t& num_interacted_features)
{
  if (all.weights.sparse)
    generate_interactions<R, S, T, audit, audit_func, sparse_parameters>(
        *ec.interactions, all.permutations, ec, dat, all.weights.sparse_weights, num_interacted_features);
  else
    generate_interactions<R, S, T, audit, audit_func, dense_parameters>(
        *ec.interactions, all.permutations, ec, dat, all.weights.dense_weights, num_interacted_features);
}

// this code is for C++98/03 complience as I unable to pass null function-pointer as template argument in g++-4.6
template <class R, class S, void (*T)(R&, float, S)>
inline void generate_interactions(vw& all, example_predict& ec, R& dat, size_t& num_interacted_features)
{
  if (all.weights.sparse)
    generate_interactions<R, S, T, sparse_parameters>(
        all.interactions, all.permutations, ec, dat, all.weights.sparse_weights, num_interacted_features);
  else
    generate_interactions<R, S, T, dense_parameters>(
        all.interactions, all.permutations, ec, dat, all.weights.dense_weights, num_interacted_features);
}

}  // namespace INTERACTIONS
