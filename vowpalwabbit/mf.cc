// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#ifdef _WIN32
#  define NOMINMAX
#  include <winsock2.h>
#else
#  include <netdb.h>
#endif

#include "reductions.h"
#include "gd.h"
#include "scope_exit.h"

using namespace VW::LEARNER;
using namespace VW::config;

struct mf
{
  size_t rank;

  uint32_t increment;

  // array to cache w*x, (l^k * x_l) and (r^k * x_r)
  // [ w*(1,x_l,x_r) , l^1*x_l, r^1*x_r, l^2*x_l, r^2*x_2, ... ]
  v_array<float> sub_predictions;

  // array for temp storage of indices during prediction
  v_array<unsigned char> predict_indices;

  // array for temp storage of indices
  v_array<unsigned char> indices;

  // array for temp storage of features
  features temp_features;

  vw* all;  // for pairs? and finalize
};

template <bool cache_sub_predictions>
void predict(mf& data, single_learner& base, example& ec)
{
  float prediction = 0;
  if (cache_sub_predictions) { data.sub_predictions.resize_but_with_stl_behavior(2 * data.rank + 1); }

  // predict from linear terms
  base.predict(ec);

  // store linear prediction
  if (cache_sub_predictions) data.sub_predictions[0] = ec.partial_prediction;
  prediction += ec.partial_prediction;

  // store namespace indices
  data.predict_indices = ec.indices;

  // erase indices
  ec.indices.clear();
  ec.indices.push_back(0);

  auto* saved_interactions = ec.interactions;
  auto restore_guard = VW::scope_exit([saved_interactions, &ec] { ec.interactions = saved_interactions; });

  std::vector<std::vector<namespace_index>> empty_interactions;
  ec.interactions = &empty_interactions;

  // add interaction terms to prediction
  for (auto& i : *saved_interactions)
  {
    auto left_ns = static_cast<int>(i[0]);
    auto right_ns = static_cast<int>(i[1]);

    if (ec.feature_space[left_ns].size() > 0 && ec.feature_space[right_ns].size() > 0)
    {
      for (size_t k = 1; k <= data.rank; k++)
      {
        ec.indices[0] = static_cast<namespace_index>(left_ns);

        // compute l^k * x_l using base learner
        base.predict(ec, k);
        float x_dot_l = ec.partial_prediction;
        if (cache_sub_predictions) data.sub_predictions[2 * k - 1] = x_dot_l;

        // set example to right namespace only
        ec.indices[0] = static_cast<namespace_index>(right_ns);

        // compute r^k * x_r using base learner
        base.predict(ec, k + data.rank);
        float x_dot_r = ec.partial_prediction;
        if (cache_sub_predictions) data.sub_predictions[2 * k] = x_dot_r;

        // accumulate prediction
        prediction += (x_dot_l * x_dot_r);
      }
    }
  }
  // restore namespace indices and label
  ec.indices = data.predict_indices;

  // finalize prediction
  ec.partial_prediction = prediction;
  ec.pred.scalar = GD::finalize_prediction(data.all->sd, data.all->logger, ec.partial_prediction);
}

void learn(mf& data, single_learner& base, example& ec)
{
  // predict with current weights
  predict<true>(data, base, ec);
  float predicted = ec.pred.scalar;

  // update linear weights
  base.update(ec);
  ec.pred.scalar = ec.updated_prediction;

  // store namespace indices
  data.indices = ec.indices;

  // erase indices
  ec.indices.clear();
  ec.indices.push_back(0);

  auto* saved_interactions = ec.interactions;
  std::vector<std::vector<namespace_index>> empty_interactions;
  ec.interactions = &empty_interactions;

  // update interaction terms
  // looping over all pairs of non-empty namespaces
  for (auto& i : *saved_interactions)
  {
    int left_ns = static_cast<int>(i[0]);
    int right_ns = static_cast<int>(i[1]);

    if (ec.feature_space[left_ns].size() > 0 && ec.feature_space[right_ns].size() > 0)
    {
      // set example to left namespace only
      ec.indices[0] = static_cast<namespace_index>(left_ns);

      // store feature values in left namespace
      data.temp_features = ec.feature_space[left_ns];

      for (size_t k = 1; k <= data.rank; k++)
      {
        features& fs = ec.feature_space[left_ns];
        // multiply features in left namespace by r^k * x_r
        for (size_t j = 0; j < fs.size(); ++j) fs.values[j] *= data.sub_predictions[2 * k];

        // update l^k using base learner
        base.update(ec, k);

        // restore left namespace features (undoing multiply)
        fs = data.temp_features;

        // compute new l_k * x_l scaling factors
        // base.predict(ec, k);
        // data.sub_predictions[2*k-1] = ec.partial_prediction;
        // ec.pred.scalar = ec.updated_prediction;
      }

      // set example to right namespace only
      ec.indices[0] = static_cast<namespace_index>(right_ns);

      // store feature values for right namespace
      data.temp_features = ec.feature_space[right_ns];

      for (size_t k = 1; k <= data.rank; k++)
      {
        features& fs = ec.feature_space[right_ns];
        // multiply features in right namespace by l^k * x_l
        for (size_t j = 0; j < fs.size(); ++j) fs.values[j] *= data.sub_predictions[2 * k - 1];

        // update r^k using base learner
        base.update(ec, k + data.rank);
        ec.pred.scalar = ec.updated_prediction;

        // restore right namespace features
        fs = data.temp_features;
      }
    }
  }
  // restore namespace indices
  ec.indices = data.indices;

  // restore original prediction
  ec.pred.scalar = predicted;
  ec.interactions = saved_interactions;
}

base_learner* mf_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<mf>();
  option_group_definition new_options("Matrix Factorization Reduction");
  new_options.add(
      make_option("new_mf", data->rank).keep().necessary().help("rank for reduction-based matrix factorization"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  data->all = &all;
  // store global pairs in local data structure and clear global pairs
  // for eventual calls to base learner
  auto non_pair_count = std::count_if(all.interactions.begin(), all.interactions.end(),
      [](const std::vector<unsigned char>& interaction) { return interaction.size() != 2; });
  if (non_pair_count > 0) { THROW("can only use pairs with new_mf"); }

  all.random_positive_weights = true;

  learner<mf, example>& l = init_learner(data, as_singleline(setup_base(options, all)), learn, predict<false>,
      2 * data->rank + 1, all.get_setupfn_name(mf_setup));
  return make_base(l);
}
