// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_pdf.h"
#include "debug_log.h"
#include "parse_args.h"
#include "learner.h"

// Aliases
using std::endl;
using VW::config::make_option;
using VW::config::option_group_definition;
using VW::config::options_i;
using VW::LEARNER::single_learner;

// Enable/Disable indented debug statements
#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cb_explore_pdf

namespace VW
{
namespace continuous_action
{
int cb_explore_pdf::learn(example& ec, experimental::api_status*)
{
  _base->learn(ec);
  return VW::experimental::error_code::success;
}

// Free function to tie function pointers to reduction class methods
void learn(cb_explore_pdf& reduction, single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.learn(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << endl; }
}

// Setup reduction in stack
LEARNER::base_learner* cb_explore_pdf_setup(config::options_i& options, vw& all)
{
  option_group_definition new_options("Continuous actions - cb_explore_pdf");
  bool invoked = false;
  float epsilon;
  float min;
  float max;
  bool first_only = false;
  new_options
      .add(make_option("cb_explore_pdf", invoked)
               .keep()
               .necessary()
               .help("Sample a pdf and pick a continuous valued action"))
      .add(make_option("epsilon", epsilon)
               .keep()
               .allow_override()
               .default_value(0.05f)
               .help("epsilon-greedy exploration"))
      .add(make_option("min_value", min).keep().default_value(0.0f).help("min value for continuous range"))
      .add(make_option("max_value", max).keep().default_value(1.0f).help("max value for continuous range"))
      .add(make_option("first_only", first_only)
               .keep()
               .help("Use user provided first action or user provided pdf or uniform random"));

  // If reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (!options.was_supplied("min_value") || !options.was_supplied("max_value"))
    THROW("error: min and max values must be supplied with cb_explore_pdf");

  LEARNER::base_learner* p_base = setup_base(options, all);
  auto p_reduction = VW::make_unique<cb_explore_pdf>();
  p_reduction->init(as_singleline(p_base));
  p_reduction->epsilon = epsilon;
  p_reduction->min_value = min;
  p_reduction->max_value = max;
  p_reduction->first_only = first_only;

  auto* l = make_reduction_learner(
      std::move(p_reduction), as_singleline(p_base), learn, predict, all.get_setupfn_name(cb_explore_pdf_setup))
                .set_prediction_type(prediction_type_t::pdf)
                .set_label_type(label_type_t::cb)
                .build();
  return make_base(*l);
}
}  // namespace continuous_action
}  // namespace VW
