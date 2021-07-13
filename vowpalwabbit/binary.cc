// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "debug_log.h"
#include "reductions.h"
#include <cfloat>
#include <cmath>

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::binary

#include "io/logger.h"

using namespace VW::config;
using std::endl;

namespace logger = VW::io::logger;

namespace VW
{
namespace binary
{
template <bool is_learn>
void predict_or_learn(char&, VW::LEARNER::single_learner& base, example& ec)
{
  if (is_learn) { base.learn(ec); }
  else
  {
    base.predict(ec);
  }

  if (ec.pred.scalar > 0)
    ec.pred.scalar = 1;
  else
    ec.pred.scalar = -1;

  VW_DBG(ec) << "binary: final-pred " << scalar_pred_to_string(ec) << features_to_string(ec) << endl;

  if (ec.l.simple.label != FLT_MAX)
  {
    if (std::fabs(ec.l.simple.label) != 1.f)
      logger::log_error("You are using label {} not -1 or 1 as loss function expects!", ec.l.simple.label);
    else if (ec.l.simple.label == ec.pred.scalar)
      ec.loss = 0.;
    else
      ec.loss = ec.weight;
  }
}

VW::LEARNER::base_learner* binary_setup(setup_base_fn& setup_base, options_i& options, vw& all)
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

  return make_base(*ret);
}

}  // namespace binary
}  // namespace VW
