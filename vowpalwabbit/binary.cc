// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "debug_log.h"
#include "reductions.h"
#include "binary.h"

#include <cfloat>
#include <cmath>

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::binary

using namespace VW::config;
using std::endl;

namespace VW
{
namespace binary
{

VW::LEARNER::base_learner* binary_setup(options_i& options, vw& all)
{
  bool binary = false;
  option_group_definition new_options("Binary loss");
  new_options.add(
      make_option("binary", binary).keep().necessary().help("report loss as binary classification on -1,1"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  auto ret = VW::LEARNER::make_no_data_reduction_learner(as_singleline(setup_base(options, all)),
      predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(binary_setup))
                 .set_learn_returns_prediction(true)
                 .build();
  // auto ret = VW::LEARNER::init_learner(as_singleline(setup_base(options, all)),
  //     predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(binary_setup), true);
  return make_base(*ret);
}

}  // namespace binary
}  // namespace VW
