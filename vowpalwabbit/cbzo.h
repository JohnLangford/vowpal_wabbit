// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "reductions.h"

namespace VW
{
namespace cbzo
{
VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

}  // namespace cbzo
}  // namespace VW
