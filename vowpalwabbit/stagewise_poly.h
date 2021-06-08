// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "reductions_fwd.h"

VW::LEARNER::base_learner* stagewise_poly_setup(
    VW::setup_base_fn setup_base_fn, VW::config::options_i& options, vw& all);
