// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "reductions_fwd.h"
#include "example_predict.h"
#include "interactions.h"

#include <vector>
#include <set>

VW::LEARNER::base_learner* generate_interactions_setup(
    VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);
