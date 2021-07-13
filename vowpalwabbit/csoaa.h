// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include "reductions_fwd.h"

namespace CSOAA
{
VW::LEARNER::base_learner* csoaa_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

VW::LEARNER::base_learner* csldf_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);
struct csoaa;
void finish_example(vw& all, csoaa&, example& ec);
}  // namespace CSOAA
