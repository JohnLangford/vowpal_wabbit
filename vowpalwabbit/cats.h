// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner_no_throw.h"
#include "api_status.h"
#include "error_constants.h"

// forward declaration
namespace VW
{
namespace config
{
struct options_i;
}
}  // namespace VW

namespace VW
{
namespace continuous_action
{
namespace cats
{
LEARNER::base_learner* setup(config::options_i& options, vw& all);
struct cats
{
  uint32_t num_actions;
  float bandwidth;
  float min_value;
  float max_value;

  cats(VW::LEARNER::single_learner* p_base, uint32_t num_actions_, float bandwidth_, float min_value_, float max_value_,
      bool bandwidth_was_supplied)
      : num_actions(num_actions_), bandwidth(bandwidth_), min_value(min_value_), max_value(max_value_), _base(p_base)
  {
    if (!bandwidth_was_supplied)
    {
      float leaf_width = (max_value - min_value) / (num_actions);  // aka unit range
      float half_leaf_width = leaf_width / 2.f;
      bandwidth = half_leaf_width;
    }
  }

  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status*)
  {
    VW_DBG(ec) << "cats::predict(), " << features_to_string(ec) << std::endl;
    _base->predict(ec);
    return VW::experimental::error_code::success;
  }
  float get_loss(const VW::cb_continuous::continuous_label& cb_cont_costs, float predicted_action) const;

private:
  VW::LEARNER::single_learner* _base = nullptr;
};

inline void predict(cats& reduction, VW::LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

inline void learn(cats& reduction, VW::LEARNER::single_learner&, example& ec);

}  // namespace cats
}  // namespace continuous_action
}  // namespace VW
