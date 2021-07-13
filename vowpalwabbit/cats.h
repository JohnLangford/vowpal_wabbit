// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner.h"
#include "options.h"
#include "api_status.h"

namespace VW
{
namespace continuous_action
{
namespace cats
{
LEARNER::base_learner* setup(setup_base_fn& setup_base_fn, config::options_i& options, vw& all);
struct cats
{
  uint32_t num_actions;
  float bandwidth;
  float min_value;
  float max_value;

  cats(LEARNER::single_learner* p_base);

  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status* status);
  float get_loss(const VW::cb_continuous::continuous_label& cb_cont_costs, float predicted_action) const;

private:
  LEARNER::single_learner* _base = nullptr;
};
}  // namespace cats
}  // namespace continuous_action
}  // namespace VW
