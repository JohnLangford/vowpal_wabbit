// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include <cstdint>
#include <vector>

#include "cb_explore_adf_common.h"
#include "cb.h"
#include "cost_sensitive.h"
#include "v_array.h"
#include "action_score.h"
#include "gen_cs_example.h"
#include "reductions_fwd.h"

namespace VW
{
namespace cb_explore_adf
{
namespace cover
{
VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);
}  // namespace cover
}  // namespace cb_explore_adf
}  // namespace VW
