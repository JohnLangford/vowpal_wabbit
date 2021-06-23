// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "sample_pdf.h"
#include "error_constants.h"
#include "api_status.h"
#include "debug_log.h"
#include "parse_args.h"
#include "guard.h"

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
#define VW_DEBUG_LOG vw_dbg::cb_sample_pdf

namespace VW
{
namespace continuous_action
{
int sample_pdf::learn(example& ec, experimental::api_status*)
{
  // one of the base reductions will call predict so we need a valid
  // predict buffer
  _pred_pdf.clear();
  {  // scope to predict & restore prediction
    auto restore = VW::swap_guard(ec.pred.pdf, _pred_pdf);
    _base->learn(ec);
  }
  return VW::experimental::error_code::success;
}

// Free function to tie function pointers to reduction class methods
void learn(sample_pdf& reduction, single_learner&, example& ec)
{
  experimental::api_status status;
  reduction.learn(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << endl; }
}

// END sample_pdf reduction and reduction methods
////////////////////////////////////////////////////

LEARNER::base_learner* sample_pdf_setup(options_i& options, vw& all)
{
  option_group_definition new_options("Continuous actions - sample pdf");
  bool invoked = false;
  new_options.add(
      make_option("sample_pdf", invoked).keep().necessary().help("Sample a pdf and pick a continuous valued action"));

  // If sample_pdf reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  LEARNER::base_learner* p_base = setup_base(options, all);
  auto p_reduction = scoped_calloc_or_throw<sample_pdf>();
  p_reduction->init(as_singleline(p_base), &all.random_seed);

  LEARNER::learner<sample_pdf, example>& l = init_learner(p_reduction, as_singleline(p_base), learn, predict, 1,
      prediction_type_t::action_pdf_value, all.get_setupfn_name(sample_pdf_setup));

  return make_base(l);
}
}  // namespace continuous_action
}  // namespace VW
