// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "baseline.h"
#include "guard.h"

#include "cb.h"
#include "example.h"
#include "learner.h"

#include <cfloat>

// TODO: extend to handle CSOAA_LDF and WAP_LDF
VW::LEARNER::base_learner* cb_algs_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

#define CB_TYPE_DR 0
#define CB_TYPE_DM 1
#define CB_TYPE_IPS 2
#define CB_TYPE_MTR 3
#define CB_TYPE_SM 4

namespace CB_ALGS
{
template <bool is_learn>
float get_cost_pred(
    VW::LEARNER::single_learner* scorer, const CB::cb_class& known_cost, example& ec, uint32_t index, uint32_t base)
{
  VW_DBG(ec) << "get_cost_pred:" << is_learn << std::endl;
  CB::label ld = ec.l.cb;

  label_data simple_temp;
  if (index == known_cost.action)
    simple_temp.label = known_cost.cost;
  else
    simple_temp.label = FLT_MAX;

  const bool baseline_enabled_old = BASELINE::baseline_enabled(&ec);
  BASELINE::set_baseline_enabled(&ec);
  ec.l.simple = simple_temp;
  bool learn = is_learn && index == known_cost.action;

  if (learn)
  {
    float old_weight = ec.weight;
    ec.weight /= known_cost.probability;
    scorer->learn(ec, index - 1 + base);
    ec.weight = old_weight;
  }
  else
    scorer->predict(ec, index - 1 + base);

  if (!baseline_enabled_old) BASELINE::reset_baseline_disabled(&ec);
  float pred = ec.pred.scalar;
  return pred;
}

// IPS estimate
inline float get_cost_estimate(const CB::cb_class& observation, uint32_t action, float offset = 0.)
{
  if (action == observation.action) return (observation.cost - offset) / observation.probability;
  return 0.;
}

// doubly robust estimate
inline float get_cost_estimate(const CB::cb_class& observation, const COST_SENSITIVE::label& scores, uint32_t action)
{
  for (auto& cl : scores.costs)
    if (cl.class_index == action) return get_cost_estimate(observation, action, cl.x) + cl.x;
  // defaults to IPS when there are no scores
  return get_cost_estimate(observation, action);
}

// IPS
inline float get_cost_estimate(const CB::label& ld, uint32_t action)
{
  for (auto& cl : ld.costs)
    if (cl.action == action) return get_cost_estimate(cl, action);
  return 0.0f;
}

// doubly robust estimate
inline float get_cost_estimate(const ACTION_SCORE::action_score& a_s, float cost, uint32_t action, float offset = 0.)
{
  if (action == a_s.action) return (cost - offset) / a_s.score;
  return 0.;
}

inline bool example_is_newline_not_header(example const& ec)
{
  return (example_is_newline(ec) && !CB::ec_is_example_header(ec));
}

void generic_output_example(vw& all, float loss, example& ec, const CB::label& ld, CB::cb_class* known_cost);

}  // namespace CB_ALGS
