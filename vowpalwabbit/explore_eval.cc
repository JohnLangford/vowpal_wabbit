// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "cb_algs.h"
#include "vw.h"
#include "cb_adf.h"
#include "rand48.h"
#include "gen_cs_example.h"
#include <memory>
#include <cfloat>
#include "shared_data.h"

// Do evaluation of nonstationary policies.
// input = contextual bandit label
// output = chosen ranking

using namespace VW::LEARNER;
using namespace CB_ALGS;
using namespace VW::config;

namespace EXPLORE_EVAL
{
struct explore_eval
{
  CB::cb_class known_cost;
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  uint64_t offset;
  CB::label action_label;
  CB::label empty_label;
  size_t example_counter;

  size_t update_count;
  size_t violations;
  float multiplier;

  bool fixed_multiplier;
};

void finish(explore_eval& data)
{
  if (!data.all->logger.quiet)
  {
    *(data.all->trace_message) << "update count = " << data.update_count << std::endl;
    if (data.violations > 0) *(data.all->trace_message) << "violation count = " << data.violations << std::endl;
    if (!data.fixed_multiplier) *(data.all->trace_message) << "final multiplier = " << data.multiplier << std::endl;
  }
}

// Semantics: Currently we compute the IPS loss no matter what flags
// are specified. We print the first action and probability, based on
// ordering by scores in the final output.

void output_example(vw& all, explore_eval& c, example& ec, multi_ex* ec_seq)
{
  if (example_is_newline_not_header(ec)) return;

  size_t num_features = 0;

  float loss = 0.;
  ACTION_SCORE::action_scores preds = (*ec_seq)[0]->pred.a_s;
  label_type_t label_type = all.example_parser->lbl_parser.label_type;

  for (size_t i = 0; i < (*ec_seq).size(); i++)
    if (!VW::LEARNER::ec_is_example_header(*(*ec_seq)[i], label_type)) num_features += (*ec_seq)[i]->get_num_features();

  bool labeled_example = true;
  if (c.known_cost.probability > 0)
  {
    for (uint32_t i = 0; i < preds.size(); i++)
    {
      float l = get_cost_estimate(c.known_cost, preds[i].action);
      loss += l * preds[i].score;
    }
  }
  else
    labeled_example = false;

  bool holdout_example = labeled_example;
  for (size_t i = 0; i < ec_seq->size(); i++) holdout_example &= (*ec_seq)[i]->test_only;

  all.sd->update(holdout_example, labeled_example, loss, ec.weight, num_features);

  for (auto& sink : all.final_prediction_sink) print_action_score(sink.get(), ec.pred.a_s, ec.tag);

  if (all.raw_prediction != nullptr)
  {
    std::string outputString;
    std::stringstream outputStringStream(outputString);
    const auto& costs = ec.l.cb.costs;

    for (size_t i = 0; i < costs.size(); i++)
    {
      if (i > 0) outputStringStream << ' ';
      outputStringStream << costs[i].action << ':' << costs[i].partial_prediction;
    }
    all.print_text_by_ref(all.raw_prediction.get(), outputStringStream.str(), ec.tag);
  }

  CB::print_update(all, !labeled_example, ec, ec_seq, true, nullptr);
}

void output_example_seq(vw& all, explore_eval& data, multi_ex& ec_seq)
{
  if (ec_seq.size() > 0)
  {
    output_example(all, data, **(ec_seq.begin()), &(ec_seq));
    if (all.raw_prediction != nullptr) all.print_text_by_ref(all.raw_prediction.get(), "", ec_seq[0]->tag);
  }
}

void finish_multiline_example(vw& all, explore_eval& data, multi_ex& ec_seq)
{
  if (ec_seq.size() > 0)
  {
    output_example_seq(all, data, ec_seq);
    CB_ADF::global_print_newline(all.final_prediction_sink);
  }
  VW::finish_example(all, ec_seq);
}

template <bool is_learn>
void do_actual_learning(explore_eval& data, multi_learner& base, multi_ex& ec_seq)
{
  example* label_example = CB_ADF::test_adf_sequence(ec_seq);

  if (label_example != nullptr)  // extract label
  {
    data.action_label = std::move(label_example->l.cb);
    label_example->l.cb = std::move(data.empty_label);
  }
  multiline_learn_or_predict<false>(base, ec_seq, data.offset);

  if (label_example != nullptr)  // restore label
  {
    label_example->l.cb = std::move(data.action_label);
    data.empty_label.costs.clear();
    data.empty_label.weight = 1.f;
  }

  data.known_cost = CB_ADF::get_observed_cost_or_default_cb_adf(ec_seq);
  if (label_example != nullptr && is_learn)
  {
    ACTION_SCORE::action_scores& a_s = ec_seq[0]->pred.a_s;

    float action_probability = 0;
    for (size_t i = 0; i < a_s.size(); i++)
      if (data.known_cost.action == a_s[i].action) action_probability = a_s[i].score;

    float threshold = action_probability / data.known_cost.probability;

    if (!data.fixed_multiplier)
      data.multiplier = std::min(data.multiplier, 1 / threshold);
    else
      threshold *= data.multiplier;

    if (threshold > 1. + 1e-6) data.violations++;

    if (data._random_state->get_and_update_random() < threshold)
    {
      example* ec_found = nullptr;
      for (example*& ec : ec_seq)
      {
        if (ec->l.cb.costs.size() == 1 && ec->l.cb.costs[0].cost != FLT_MAX && ec->l.cb.costs[0].probability > 0)
          ec_found = ec;
        if (threshold > 1) ec->weight *= threshold;
      }
      ec_found->l.cb.costs[0].probability = action_probability;

      multiline_learn_or_predict<true>(base, ec_seq, data.offset);

      if (threshold > 1)
      {
        float inv_threshold = 1.f / threshold;
        for (auto& ec : ec_seq) ec->weight *= inv_threshold;
      }
      ec_found->l.cb.costs[0].probability = data.known_cost.probability;
      data.update_count++;
    }
  }
}
}  // namespace EXPLORE_EVAL

using namespace EXPLORE_EVAL;

base_learner* explore_eval_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<explore_eval>();
  bool explore_eval_option = false;
  option_group_definition new_options("Explore evaluation");
  new_options
      .add(make_option("explore_eval", explore_eval_option)
               .keep()
               .necessary()
               .help("Evaluate explore_eval adf policies"))
      .add(make_option("multiplier", data->multiplier)
               .help("Multiplier used to make all rejection sample probabilities <= 1"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  data->all = &all;
  data->_random_state = all.get_random_state();

  if (options.was_supplied("multiplier"))
    data->fixed_multiplier = true;
  else
    data->multiplier = 1;

  if (!options.was_supplied("cb_explore_adf")) options.insert("cb_explore_adf", "");

  multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  learner<explore_eval, multi_ex>& l = init_learner(data, base, do_actual_learning<true>, do_actual_learning<false>, 1,
      prediction_type_t::action_probs, all.get_setupfn_name(explore_eval_setup), true);

  l.set_finish_example(finish_multiline_example);
  l.set_finish(finish);
  return make_base(l);
}
