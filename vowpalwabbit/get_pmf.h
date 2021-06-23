// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner_no_throw.h"
#include "example.h"
#include "error_constants.h"
#include "api_status.h"
#include "guard.h"

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
struct get_pmf
{
  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status*)
  {
    uint32_t base_prediction;

    {  // predict & restore prediction
      auto restore = VW::stash_guard(ec.pred);
      _base->predict(ec);
      base_prediction = ec.pred.multiclass - 1;
    }

    // Assume ec.pred.a_s allocated by the caller (probably pmf_to_pdf);
    ec.pred.a_s.clear();
    ec.pred.a_s.push_back({base_prediction, 1.0f});

    return VW::experimental::error_code::success;
  }

  void init(LEARNER::single_learner* p_base, float epsilon)
  {
    _base = p_base;
    _epsilon = epsilon;
  }

private:
  LEARNER::single_learner* _base = nullptr;
  float _epsilon;
};

// Free function to tie function pointers to reduction class methods
inline void learn(get_pmf& reduction, LEARNER::single_learner&, example& ec);
// Free function to tie function pointers to reduction class methods
inline void predict(get_pmf& reduction, LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

// Setup reduction in stack
LEARNER::base_learner* get_pmf_setup(config::options_i& options, vw& all);
}  // namespace continuous_action
}  // namespace VW
