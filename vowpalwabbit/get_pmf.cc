// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "get_pmf.h"
#include "debug_log.h"
#include "parse_args.h"
#include "options.h"

// Aliases
using std::endl;
using VW::cb_continuous::continuous_label;
using VW::cb_continuous::continuous_label_elm;
using VW::config::make_option;
using VW::config::option_group_definition;
using VW::config::options_i;
using VW::LEARNER::single_learner;

// Enable/Disable indented debug statements
#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cb_explore_get_pmf

namespace VW
{
namespace continuous_action
{
int get_pmf::learn(example& ec, experimental::api_status*)
{
  _base->learn(ec);
  return VW::experimental::error_code::success;
}

// END sample_pdf reduction and reduction methods
////////////////////////////////////////////////////

// Free function to tie function pointers to reduction class methods
void learn(get_pmf& reduction, LEARNER::single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.learn(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << std::endl; }
}

// Setup reduction in stack
LEARNER::base_learner* get_pmf_setup(config::options_i& options, vw& all)
{
  option_group_definition new_options("Continuous actions - convert to pmf");
  bool invoked = false;
  float epsilon = 0.0f;
  new_options.add(
      make_option("get_pmf", invoked).keep().necessary().help("Convert a single multiclass prediction to a pmf"));

  // If reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  LEARNER::base_learner* p_base = setup_base(options, all);
  auto p_reduction = VW::make_unique<get_pmf>();
  p_reduction->init(as_singleline(p_base), epsilon);

  auto* l = VW::LEARNER::make_reduction_learner(
      std::move(p_reduction), as_singleline(p_base), learn, predict, all.get_setupfn_name(get_pmf_setup))
                .set_params_per_weight(1)
                .set_prediction_type(prediction_type_t::pdf)
                .build();

  return make_base(*l);
}
}  // namespace continuous_action
}  // namespace VW
