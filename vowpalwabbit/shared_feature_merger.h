// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#pragma once
#include "reductions_fwd.h"

struct vw;

namespace VW
{
namespace shared_feature_merger
{
VW::LEARNER::base_learner* shared_feature_merger_setup(
    VW::setup_base_fn& setup_base, config::options_i& options, vw& all);

}  // namespace shared_feature_merger
}  // namespace VW
