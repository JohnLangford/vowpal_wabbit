// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "learner_no_throw.h"
#include "error_constants.h"
#include "api_status.h"

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
////////////////////////////////////////////////////
// BEGIN sample_pdf reduction and reduction methods
struct cb_explore_pdf
{
  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status*)
  {
    const auto& reduction_features = ec._reduction_features.template get<VW::continuous_actions::reduction_features>();
    if (first_only && !reduction_features.is_pdf_set() && !reduction_features.is_chosen_action_set())
    {
      // uniform random
      ec.pred.pdf.push_back(
          VW::continuous_actions::pdf_segment{min_value, max_value, static_cast<float>(1. / (max_value - min_value))});
      return VW::experimental::error_code::success;
    }
    else if (first_only && reduction_features.is_pdf_set())
    {
      // pdf provided
      ec.pred.pdf = reduction_features.pdf;
      return VW::experimental::error_code::success;
    }

    _base->predict(ec);

    continuous_actions::probability_density_function& _pred_pdf = ec.pred.pdf;
    for (uint32_t i = 0; i < _pred_pdf.size(); i++)
    { _pred_pdf[i].pdf_value = _pred_pdf[i].pdf_value * (1 - epsilon) + epsilon / (max_value - min_value); }
    return VW::experimental::error_code::success;
  }

  void init(VW::LEARNER::single_learner* p_base) { _base = p_base; }

  float epsilon;
  float min_value;
  float max_value;
  bool first_only;

private:
  VW::LEARNER::single_learner* _base = nullptr;
};  // namespace continuous_action

// Free function to tie function pointers to reduction class methods
inline void predict(cb_explore_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

inline void learn(cb_explore_pdf& reduction, VW::LEARNER::single_learner&, example& ec);

// Setup reduction in stack
LEARNER::base_learner* cb_explore_pdf_setup(config::options_i& options, vw& all);
}  // namespace continuous_action
}  // namespace VW
