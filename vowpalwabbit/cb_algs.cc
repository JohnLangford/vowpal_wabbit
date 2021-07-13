// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cfloat>

#include "vw.h"
#include "reductions.h"
#include "cb_algs.h"
#include "vw_exception.h"
#include "gen_cs_example.h"
#include "cb_label_parser.h"
#include "shared_data.h"

#include "io/logger.h"

using namespace VW::LEARNER;
using namespace VW::config;

using namespace CB;
using namespace GEN_CS;

namespace logger = VW::io::logger;

namespace CB_ALGS
{
struct cb
{
  cb_to_cs cbcs;
};

bool know_all_cost_example(CB::label& ld)
{
  if (ld.costs.size() <= 1)  // this means we specified an example where all actions are possible but only specified the
                             // cost for the observed action
    return false;

  // if we specified more than 1 action for this example, i.e. either we have a limited set of possible actions, or all
  // actions are specified than check if all actions have a specified cost
  for (auto& cl : ld.costs)
    if (cl.cost == FLT_MAX) return false;

  return true;
}

template <bool is_learn>
void predict_or_learn(cb& data, single_learner& base, example& ec)
{
  cb_to_cs& c = data.cbcs;
  auto optional_cost = get_observed_cost_cb(ec.l.cb);
  // cost observed, not default
  if (optional_cost.first) { c.known_cost = optional_cost.second; }
  else
  {
    c.known_cost = CB::cb_class{};
  }

  // cost observed, not default
  if (optional_cost.first && (c.known_cost.action < 1 || c.known_cost.action > c.num_actions))
    logger::errlog_error("invalid action: {}", c.known_cost.action);

  // generate a cost-sensitive example to update classifiers
  gen_cs_example<is_learn>(c, ec, ec.l.cb, ec.l.cs);

  if (c.cb_type != CB_TYPE_DM)
  {
    if (is_learn)
      base.learn(ec);
    else
      base.predict(ec);

    for (size_t i = 0; i < ec.l.cb.costs.size(); i++)
      ec.l.cb.costs[i].partial_prediction = ec.l.cs.costs[i].partial_prediction;
  }
}

void predict_eval(cb&, single_learner&, example&) { THROW("can not use a test label for evaluation"); }

void learn_eval(cb& data, single_learner&, example& ec)
{
  cb_to_cs& c = data.cbcs;
  auto optional_cost = get_observed_cost_cb(ec.l.cb_eval.event);
  // cost observed, not default
  if (optional_cost.first) { c.known_cost = optional_cost.second; }
  else
  {
    c.known_cost = CB::cb_class{};
  }
  gen_cs_example<true>(c, ec, ec.l.cb_eval.event, ec.l.cs);

  for (size_t i = 0; i < ec.l.cb_eval.event.costs.size(); i++)
    ec.l.cb_eval.event.costs[i].partial_prediction = ec.l.cs.costs[i].partial_prediction;

  ec.pred.multiclass = ec.l.cb_eval.action;
}

void output_example(vw& all, cb& data, example& ec, CB::label& ld)
{
  float loss = 0.;

  cb_to_cs& c = data.cbcs;
  if (!CB::is_test_label(ld)) loss = get_cost_estimate(c.known_cost, c.pred_scores, ec.pred.multiclass);

  generic_output_example(all, loss, ec, ld, &c.known_cost);
}

void generic_output_example(vw& all, float loss, example& ec, const CB::label& ld, CB::cb_class* known_cost)
{
  all.sd->update(ec.test_only, !CB::is_test_label(ld), loss, 1.f, ec.get_num_features());

  for (auto& sink : all.final_prediction_sink)
    all.print_by_ref(sink.get(), static_cast<float>(ec.pred.multiclass), 0, ec.tag);

  if (all.raw_prediction != nullptr)
  {
    std::stringstream outputStringStream;
    for (unsigned int i = 0; i < ld.costs.size(); i++)
    {
      cb_class cl = ld.costs[i];
      if (i > 0) outputStringStream << ' ';
      outputStringStream << cl.action << ':' << cl.partial_prediction;
    }
    all.print_text_by_ref(all.raw_prediction.get(), outputStringStream.str(), ec.tag);
  }

  bool is_ld_test_label = CB::is_test_label(ld);
  if (!is_ld_test_label) { print_update(all, is_ld_test_label, ec, nullptr, false, known_cost); }
  else
  {
    print_update(all, is_ld_test_label, ec, nullptr, false, nullptr);
  }
}

void finish_example(vw& all, cb& c, example& ec)
{
  output_example(all, c, ec, ec.l.cb);
  VW::finish_example(all, ec);
}

void eval_finish_example(vw& all, cb& c, example& ec)
{
  output_example(all, c, ec, ec.l.cb_eval.event);
  VW::finish_example(all, ec);
}
}  // namespace CB_ALGS
using namespace CB_ALGS;
base_learner* cb_algs_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<cb>();
  std::string type_string = "dr";
  bool eval = false;
  bool force_legacy = true;

  option_group_definition new_options("Contextual Bandit Options");
  new_options
      .add(make_option("cb", data->cbcs.num_actions)
               .keep()
               .necessary()
               .help("Use contextual bandit learning with <k> costs"))
      .add(make_option("cb_type", type_string).keep().help("contextual bandit method to use in {ips,dm,dr}"))
      .add(make_option("eval", eval).help("Evaluate a policy rather than optimizing."))
      .add(make_option("cb_force_legacy", force_legacy)
               .keep()
               .help("Default to non-adf cb implementation (cb_to_cb_adf)"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (!eval && !force_legacy) return nullptr;

  // Ensure serialization of this option in all cases.
  if (!options.was_supplied("cb_type"))
  {
    options.insert("cb_type", type_string);
    options.add_and_parse(new_options);
  }

  cb_to_cs& c = data->cbcs;

  size_t problem_multiplier = 2;  // default for DR
  if (type_string.compare("dr") == 0)
    c.cb_type = CB_TYPE_DR;
  else if (type_string.compare("dm") == 0)
  {
    if (eval) THROW("direct method can not be used for evaluation --- it is biased.");
    c.cb_type = CB_TYPE_DM;
    problem_multiplier = 1;
  }
  else if (type_string.compare("ips") == 0)
  {
    c.cb_type = CB_TYPE_IPS;
    problem_multiplier = 1;
  }
  else
  {
    logger::errlog_warn("warning: cb_type must be in {'ips','dm','dr'}; resetting to dr.");
    c.cb_type = CB_TYPE_DR;
  }

  if (!options.was_supplied("csoaa"))
  {
    std::stringstream ss;
    ss << data->cbcs.num_actions;
    options.insert("csoaa", ss.str());
  }

  auto base = as_singleline(setup_base(options, all));
  if (eval) { all.example_parser->lbl_parser = CB_EVAL::cb_eval; }
  else
  {
    all.example_parser->lbl_parser = CB::cb_label;
  }

  learner<cb, example>* l;
  if (eval)
  {
    l = &init_learner(data, base, learn_eval, predict_eval, problem_multiplier, prediction_type_t::multiclass,
        all.get_setupfn_name(cb_algs_setup) + "-eval", true);
    l->set_finish_example(eval_finish_example);
  }
  else
  {
    l = &init_learner(data, base, predict_or_learn<true>, predict_or_learn<false>, problem_multiplier,
        prediction_type_t::multiclass, all.get_setupfn_name(cb_algs_setup));
    l->set_finish_example(finish_example);
  }
  c.scorer = all.scorer;

  return make_base(*l);
}
