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
namespace cats_pdf
{
////////////////////////////////////////////////////
// BEGIN cats_pdf reduction and reduction methods
struct cats_pdf
{
  cats_pdf(VW::LEARNER::single_learner* p_base, bool always_predict) : _base(p_base), _always_predict(always_predict) {}

  int learn(example& ec, experimental::api_status* status);
  // Pass through
  int predict(example& ec, experimental::api_status*)
  {
    VW_DBG(ec) << "cats_pdf::predict(), " << features_to_string(ec) << std::endl;
    _base->predict(ec);
    return VW::experimental::error_code::success;
  }

private:
  VW::LEARNER::single_learner* _base = nullptr;
  bool _always_predict = false;
};

// Free function to tie function pointers to reduction class methods
inline void predict(cats_pdf& reduction, VW::LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

// Free function to tie function pointers to reduction class methods
inline void learn(cats_pdf& reduction, VW::LEARNER::single_learner&, example& ec);

// Setup reduction in stack
LEARNER::base_learner* setup(config::options_i& options, vw& all);

}  // namespace cats_pdf
}  // namespace continuous_action
}  // namespace VW
