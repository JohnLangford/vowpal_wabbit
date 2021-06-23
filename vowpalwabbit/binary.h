#pragma once
// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions_fwd.h"
#include "example.h"
#include "learner_no_throw.h"

#ifndef VW_NOEXCEPT
#  include "io/logger.h"

namespace logger = VW::io::logger;
#endif

namespace VW
{
namespace binary
{
VW::LEARNER::base_learner* binary_setup(VW::config::options_i& options, vw& all);
template <bool is_learn>
void predict_or_learn(char&, VW::LEARNER::single_learner& base, example& ec)
{
  if (is_learn) { base.learn(ec); }
  else
  {
    base.predict(ec);
  }

  if (ec.pred.scalar > 0)
    ec.pred.scalar = 1;
  else
    ec.pred.scalar = -1;

  if (ec.l.simple.label != FLT_MAX)
  {
    if (std::fabs(ec.l.simple.label) != 1.f)
    {
        
#ifndef VW_NOEXCEPT
      logger::log_error("You are using label {} not -1 or 1 as loss function expects!", ec.l.simple.label);
#endif
    }
    else if (ec.l.simple.label == ec.pred.scalar)
      ec.loss = 0.;
    else
      ec.loss = ec.weight;
  }
}
}  // namespace binary
}  // namespace VW
