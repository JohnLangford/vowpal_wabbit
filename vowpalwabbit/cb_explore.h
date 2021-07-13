// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include "reductions_fwd.h"
#include "cb.h"

VW::LEARNER::base_learner* cb_explore_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

namespace CB_EXPLORE
{
void generic_output_example(vw& all, float loss, example& ec, CB::label& ld);
}