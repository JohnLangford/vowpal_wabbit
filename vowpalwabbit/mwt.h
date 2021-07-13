// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include "reductions_fwd.h"

#include "io/io_adapter.h"

VW::LEARNER::base_learner* mwt_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all);

namespace MWT
{
void print_scalars(VW::io::writer* f, v_array<float>& scalars, v_array<char>& tag);
}  // namespace MWT
