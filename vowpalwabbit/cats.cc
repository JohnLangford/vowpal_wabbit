// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cats.h"
#include "parse_args.h"
#include "error_constants.h"
#include "debug_log.h"
#include "shared_data.h"

#include <cfloat>
#include <cmath>
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
#define VW_DEBUG_LOG vw_dbg::cats

// Forward declarations
namespace VW
{
void finish_example(vw& all, example& ec);
}

namespace VW
{
namespace continuous_action
{
namespace cats
{
////////////////////////////////////////////////////
// BEGIN cats reduction and reduction methods
// Pass through
int cats::predict(example& ec, experimental::api_status*)
{
  VW_DBG(ec) << "cats::predict(), " << features_to_string(ec) << endl;
  _base->predict(ec);
  return VW::experimental::error_code::success;
}

// Pass through
int cats::learn(example& ec, experimental::api_status* status = nullptr)
{
  assert(!ec.test_only);
  predict(ec, status);
  VW_DBG(ec) << "cats::learn(), " << to_string(ec.l.cb_cont) << features_to_string(ec) << endl;
  _base->learn(ec);
  return VW::experimental::error_code::success;
}

float cats::get_loss(const VW::cb_continuous::continuous_label& cb_cont_costs, float predicted_action) const
{
  float loss = 0.f;
  if (!cb_cont_costs.costs.empty())
  {
    const float continuous_range = max_value - min_value;
    const float unit_range = continuous_range / num_actions;

    const float ac = (predicted_action - min_value) / unit_range;
    int discretized_action = std::min(static_cast<int>(num_actions - 1), static_cast<int>(std::floor(ac)));
    // centre of predicted action
    const float centre = min_value + discretized_action * unit_range + unit_range / 2.0f;

    // is centre close to action from label
    auto logged_action = cb_cont_costs.costs[0].action;
    if ((logged_action - centre <= bandwidth) && (centre - logged_action <= bandwidth))
    {
      float actual_b = std::min(max_value, centre + bandwidth) - std::max(min_value, centre - bandwidth);

      loss = cb_cont_costs.costs[0].cost / float(cb_cont_costs.costs[0].pdf_value * actual_b);
    }
  }

  return loss;
}

cats::cats(single_learner* p_base) : _base(p_base) {}

// Free function to tie function pointers to reduction class methods
template <bool is_learn>
void predict_or_learn(cats& reduction, single_learner&, example& ec)
{
  experimental::api_status status;
  if (is_learn)
    reduction.learn(ec, &status);
  else
    reduction.predict(ec, &status);

  if (status.get_error_code() != VW::experimental::error_code::success)
  { VW_DBG(ec) << status.get_error_msg() << endl; }
}

// END cats reduction and reduction methods
/////////////////////////////////////////////////

///////////////////////////////////////////////////
// BEGIN: functions to output progress
class reduction_output
{
public:
  static void report_progress(vw& all, const cats&, const example& ec);
  static void output_predictions(std::vector<std::unique_ptr<VW::io::writer>>& predict_file_descriptors,
      const continuous_actions::probability_density_function_value& prediction);

private:
  static inline bool does_example_have_label(const example& ec);
  static void print_update_cb_cont(vw& all, const example& ec);
};

// Free function to tie function pointers to output class methods
void finish_example(vw& all, cats& data, example& ec)
{
  // add output example
  reduction_output::report_progress(all, data, ec);
  reduction_output::output_predictions(all.final_prediction_sink, ec.pred.pdf_value);
  VW::finish_example(all, ec);
}

void reduction_output::output_predictions(std::vector<std::unique_ptr<VW::io::writer>>& predict_file_descriptors,
    const continuous_actions::probability_density_function_value& prediction)
{
  // output to the prediction to all files
  const std::string str = to_string(prediction, true);
  for (auto& f : predict_file_descriptors) f->write(str.c_str(), str.size());
}

void reduction_output::report_progress(vw& all, const cats& data, const example& ec)
{
  auto loss = data.get_loss(ec.l.cb_cont, ec.pred.pdf_value.action);

  all.sd->update(ec.test_only, does_example_have_label(ec), loss, ec.weight, ec.get_num_features());
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
        to_string(ec.pred.pdf_value),                                 // Prediction
        ec.get_num_features(), all.progress_add, all.progress_arg);
  }
}

// END: functions to output progress
////////////////////////////////////////////////////

// Setup reduction in stack
LEARNER::base_learner* setup(setup_base_fn& setup_base, options_i& options, vw& all)
{
  option_group_definition new_options("Continuous actions tree with smoothing");
  uint32_t num_actions = 0;
  float bandwidth = 0;
  float min_value = 0;
  float max_value = 0;
  new_options.add(make_option("cats", num_actions).keep().necessary().help("number of discrete actions <k> for cats"))
      .add(make_option("min_value", min_value).keep().help("Minimum continuous value"))
      .add(make_option("max_value", max_value).keep().help("Maximum continuous value"))
      .add(make_option("bandwidth", bandwidth)
               .keep()
               .help("Bandwidth (radius) of randomization around discrete actions in terms of continuous range. By "
                     "default will be set to half of the continuous action unit-range resulting in smoothing that "
                     "stays inside the action space unit-range:\nunit_range = (max_value - "
                     "min_value)/num-of-actions\ndefault bandwidth = unit_range / 2.0"));

  // If cats reduction was not invoked, don't add anything
  // to the reduction stack;
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (num_actions <= 0) THROW(VW::experimental::error_code::num_actions_gt_zero_s);

  // cats stack = [cats -> sample_pdf -> cats_pdf ... rest specified by cats_pdf]
  if (!options.was_supplied("sample_pdf")) options.insert("sample_pdf", "");
  options.insert("cats_pdf", std::to_string(num_actions));

  if (!options.was_supplied("bandwidth"))
  {
    float leaf_width = (max_value - min_value) / (num_actions);  // aka unit range
    float half_leaf_width = leaf_width / 2.f;
    bandwidth = half_leaf_width;
    *(all.trace_message) << "Bandwidth was not supplied, setting default to half the continuous action unit range: "
                         << bandwidth << std::endl;
  }

  LEARNER::base_learner* p_base = setup_base(options, all);
  auto p_reduction = scoped_calloc_or_throw<cats>(as_singleline(p_base));
  p_reduction->num_actions = num_actions;
  p_reduction->bandwidth = bandwidth;
  p_reduction->max_value = max_value;
  p_reduction->min_value = min_value;

  LEARNER::learner<cats, example>& l = init_learner(p_reduction, as_singleline(p_base), predict_or_learn<true>,
      predict_or_learn<false>, 1, prediction_type_t::action_pdf_value, all.get_setupfn_name(setup), true);

  l.set_finish_example(finish_example);

  return make_base(l);
}

}  // namespace cats
}  // namespace continuous_action
}  // namespace VW
