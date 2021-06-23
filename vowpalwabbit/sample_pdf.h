// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner_no_throw.h"
#include "error_constants.h"
#include "api_status.h"
#include "explore.h"
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
struct sample_pdf
{
  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status*)
  {
    _pred_pdf.clear();

    {  // scope to predict & restore prediction
      auto restore = VW::swap_guard(ec.pred.pdf, _pred_pdf);
      _base->predict(ec);
    }

    const int ret_code = exploration::sample_pdf(_p_random_state, std::begin(_pred_pdf), std::end(_pred_pdf),
        ec.pred.pdf_value.action, ec.pred.pdf_value.pdf_value);

    if (ret_code != S_EXPLORATION_OK) return VW::experimental::error_code::sample_pdf_failed;

    return VW::experimental::error_code::success;
  }

  void init(VW::LEARNER::single_learner* p_base, uint64_t* p_random_seed)
  {
    _base = p_base;
    _p_random_state = p_random_seed;
    _pred_pdf.clear();
  }

private:
  uint64_t* _p_random_state;
  continuous_actions::probability_density_function _pred_pdf;
  VW::LEARNER::single_learner* _base = nullptr;
};

// Free function to tie function pointers to reduction class methods
inline void predict(sample_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

// Free function to tie function pointers to reduction class methods
inline void learn(sample_pdf& reduction, VW::LEARNER::single_learner&, example& ec);
// END sample_pdf reduction and reduction methods
////////////////////////////////////////////////////

// Setup reduction in stack
LEARNER::base_learner* sample_pdf_setup(config::options_i& options, vw& all);
}  // namespace continuous_action
}  // namespace VW
