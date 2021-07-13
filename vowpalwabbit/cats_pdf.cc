// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

// Notes:
// This reduction exists to be invoked as a top level reduction that
// can ouput pdf related to cats_tree
// It can also parse a continuous labeled example.

#include "cats_pdf.h"
#include "parse_args.h"
#include "error_constants.h"
#include "api_status.h"
#include "cb_continuous_label.h"
#include "debug_log.h"
#include "shared_data.h"

#include <cfloat>
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
#define VW_DEBUG_LOG vw_dbg::cats_pdf

// Forward declarations
namespace VW
{
void finish_example(vw& all, example& ec);
}

namespace VW
{
namespace continuous_action
{
namespace cats_pdf
{
////////////////////////////////////////////////////
// BEGIN cats_pdf reduction and reduction methods
struct cats_pdf
{
  cats_pdf(single_learner* p_base, bool always_predict = false);

  int learn(example& ec, experimental::api_status* status);
  int predict(example& ec, experimental::api_status* status);

private:
  single_learner* _base = nullptr;
  bool _always_predict = false;
};

// Pass through
int cats_pdf::predict(example& ec, experimental::api_status*)
{
  VW_DBG(ec) << "cats_pdf::predict(), " << features_to_string(ec) << endl;
  _base->predict(ec);
  return VW::experimental::error_code::success;
}

// Pass through
int cats_pdf::learn(example& ec, experimental::api_status*)
{
  assert(!ec.test_only);
  VW_DBG(ec) << "cats_pdf::learn(), " << to_string(ec.l.cb_cont) << features_to_string(ec) << endl;

  if (_always_predict) { _base->predict(ec); }

  _base->learn(ec);
  return VW::experimental::error_code::success;
}

cats_pdf::cats_pdf(single_learner* p_base, bool always_predict) : _base(p_base), _always_predict(always_predict) {}

// Free function to tie function pointers to reduction class methods
template <bool is_learn>
void predict_or_learn(cats_pdf& reduction, single_learner&, example& ec)
{
  experimental::api_status status;
  if (is_learn)
    reduction.learn(ec, &status);
  else
    reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << endl; }
}
// END cats_pdf reduction and reduction methods
////////////////////////////////////////////////////

///////////////////////////////////////////////////
// BEGIN: functions to output progress
class reduction_output
{
public:
  static void report_progress(vw& all, const cats_pdf&, const example& ec);
  static void output_predictions(std::vector<std::unique_ptr<VW::io::writer>>& predict_file_descriptors,
      const continuous_actions::probability_density_function& prediction);

private:
  static inline bool does_example_have_label(const example& ec);
  static void print_update_cb_cont(vw& all, const example& ec);
};

// Free function to tie function pointers to output class methods
void finish_example(vw& all, cats_pdf& data, example& ec)
{
  // add output example
  reduction_output::report_progress(all, data, ec);
  reduction_output::output_predictions(all.final_prediction_sink, ec.pred.pdf);
  VW::finish_example(all, ec);
}

void reduction_output::output_predictions(std::vector<std::unique_ptr<VW::io::writer>>& predict_file_descriptors,
    const continuous_actions::probability_density_function& prediction)
{
  // output to the prediction to all files
  const std::string str = to_string(prediction, true);
  for (auto& f : predict_file_descriptors)
  {
    f->write(str.c_str(), str.size());
    f->write("\n", 1);
  }
}

// "average loss" "since last" "example counter" "example weight"
// "current label" "current predict" "current features"
void reduction_output::report_progress(vw& all, const cats_pdf&, const example& ec)
{
  const auto& cb_cont_costs = ec.l.cb_cont.costs;
  all.sd->update(ec.test_only, does_example_have_label(ec), cb_cont_costs.empty() ? 0.f : cb_cont_costs[0].cost,
      ec.weight, ec.get_num_features());
  all.sd->weighted_labels += ec.weight;
  print_update_cb_cont(all, ec);
}

inline bool reduction_output::does_example_have_label(const example& ec)
{
  return (!ec.l.cb_cont.costs.empty() && ec.l.cb_cont.costs[0].action != FLT_MAX);
}

void reduction_output::print_update_cb_cont(vw& all, const example& ec)
{
  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.logger.quiet && !all.bfgs)
  {
    all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass,
        ec.test_only ? "unknown" : to_string(ec.l.cb_cont.costs[0]),  // Label
        to_string(ec.pred.pdf),                                       // Prediction
        ec.get_num_features(), all.progress_add, all.progress_arg);
  }
}

// END: functions to output progress
////////////////////////////////////////////////////

// Setup reduction in stack
LEARNER::base_learner* setup(setup_base_fn& setup_base, config::options_i& options, vw& all)
{
  option_group_definition new_options("Continuous action tree with smoothing with full pdf");
  int num_actions = 0;
  new_options.add(
      make_option("cats_pdf", num_actions).keep().necessary().help("number of tree labels <k> for cats_pdf"));

  // If cats reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (num_actions <= 0) THROW(VW::experimental::error_code::num_actions_gt_zero_s);

  // cats stack = [cats_pdf -> cb_explore_pdf -> pmf_to_pdf -> get_pmf -> cats_tree]
  if (!options.was_supplied("cb_explore_pdf")) options.insert("cb_explore_pdf", "");
  options.insert("pmf_to_pdf", std::to_string(num_actions));

  if (!options.was_supplied("get_pmf")) options.insert("get_pmf", "");
  options.insert("cats_tree", std::to_string(num_actions));

  LEARNER::base_learner* p_base = setup_base(options, all);
  bool always_predict = all.final_prediction_sink.size() > 0;
  auto p_reduction = scoped_calloc_or_throw<cats_pdf>(as_singleline(p_base), always_predict);

  LEARNER::learner<cats_pdf, example>& l = init_learner(p_reduction, as_singleline(p_base), predict_or_learn<true>,
      predict_or_learn<false>, 1, prediction_type_t::pdf, all.get_setupfn_name(setup), true);

  l.set_finish_example(finish_example);
  all.example_parser->lbl_parser = cb_continuous::the_label_parser;

  return make_base(l);
}
}  // namespace cats_pdf
}  // namespace continuous_action
}  // namespace VW
