// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "reductions_fwd.h"

struct feature;

VW::LEARNER::base_learner* lda_setup(VW::setup_base_fn& setup_base, VW::config::options_i&, vw&);

void get_top_weights(vw* all, int top_words_count, int topic, std::vector<feature>& output);
