// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

// This is a learner which does nothing with examples.  Used when VW is used as a compressor.

#include "reductions.h"

using namespace VW::config;

void learn(char&, VW::LEARNER::base_learner&, example&) {}

VW::LEARNER::base_learner* noop_setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  bool noop = false;
  option_group_definition new_options("Noop Learner");
  new_options.add(make_option("noop", noop).keep().necessary().help("do no learning"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  return make_base(VW::LEARNER::init_learner(learn, 1, all.get_setupfn_name(noop_setup)));
}
