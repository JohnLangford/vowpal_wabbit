// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_to_cb_adf.h"

#include "vw_versions.h"
#include "reductions.h"
#include "learner.h"
#include "vw.h"
#include "cbify.h"
#include "cb_label_parser.h"

using namespace VW::LEARNER;
using namespace VW::config;

struct cb_to_cb_adf
{
  parameters* weights;
  cbify_adf_data adf_data;
  bool explore_mode;
  multi_learner* adf_learner;
};

template <bool is_learn>
void predict_or_learn(cb_to_cb_adf& data, multi_learner& base, example& ec)
{
  data.adf_data.copy_example_to_adf(*data.weights, ec);

  CB::label backup_ld;
  CB::label new_ld;
  bool is_test_label = CB::is_test_label(ec.l.cb);

  uint32_t chosen_action = 0;
  uint32_t index_with_cost = 0;

  if (!is_test_label)
  {
    // cb.costs can be more than one in --cb
    // we need to keep the observed cost one only
    if (ec.l.cb.costs.size() > 1)
    {
      for (auto c : ec.l.cb.costs)
      {
        if (c.has_observed_cost())
        {
          chosen_action = c.action - 1;
          break;
        }
        index_with_cost++;
      }
    }
    else
    {
      chosen_action = ec.l.cb.costs[0].action - 1;
      index_with_cost = 0;
    }

    new_ld.weight = ec.l.cb.weight;
    new_ld.costs.emplace_back(ec.l.cb.costs[index_with_cost].cost, ec.l.cb.costs[index_with_cost].action - 1,
        ec.l.cb.costs[index_with_cost].probability);
  }

  if (!is_test_label && chosen_action < data.adf_data.num_actions)
  {
    backup_ld = std::move(data.adf_data.ecs[chosen_action]->l.cb);
    data.adf_data.ecs[chosen_action]->l.cb = std::move(new_ld);
  }

  auto restore_guard = VW::scope_exit([&backup_ld, &data, &ec, &chosen_action, &is_test_label, &new_ld] {
    if (!is_test_label && chosen_action < data.adf_data.num_actions)
    {
      new_ld = std::move(data.adf_data.ecs[chosen_action]->l.cb);
      data.adf_data.ecs[chosen_action]->l.cb = std::move(backup_ld);
    }
  });

  if (!base.learn_returns_prediction || !is_learn) { base.predict(data.adf_data.ecs); }

  if (is_learn) { base.learn(data.adf_data.ecs); }

  ec.pred.a_s = std::move(data.adf_data.ecs[0]->pred.a_s);
  if (!data.explore_mode) ec.pred.multiclass = ec.pred.a_s[0].action + 1;
}

void finish_example(vw& all, cb_to_cb_adf& c, example& ec)
{
  c.adf_data.ecs[0]->pred.a_s = std::move(ec.pred.a_s);

  c.adf_learner->print_example(all, c.adf_data.ecs);

  VW::finish_example(all, ec);
}

/*
    Purpose: run before cb, cb_explore, cbify and cb_adf related reductions
    This will 'translate' cb (non adf) input commands into their cb_adf counterparts
    Except when:
        - the model file loaded is from a version older than and not including 8.11.0
        - user bypasses this translation step using '--cb_force_legacy'

    Related files: cb_algs.cc, cb_explore.cc, cbify.cc
*/
VW::LEARNER::base_learner* cb_to_cb_adf_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  bool compat_old_cb = false;
  bool force_legacy = false;
  std::string type_string = "mtr";
  uint32_t num_actions;
  uint32_t cbx_num_actions;
  uint32_t cbi_num_actions;

  option_group_definition new_options("Contextual Bandit Options: cb -> cb_adf");
  new_options
      .add(make_option("cb_to_cbadf", num_actions).necessary().help("Maps cb_adf to cb. Disable with cb_force_legacy."))
      .add(make_option("cb", num_actions).keep().help("Maps cb_adf to cb. Disable with cb_force_legacy."))
      .add(make_option("cb_explore", cbx_num_actions)
               .keep()
               .help("Translate cb explore to cb_explore_adf. Disable with cb_force_legacy."))
      .add(
          make_option("cbify", cbi_num_actions).keep().help("Translate cbify to cb_adf. Disable with cb_force_legacy."))
      .add(make_option("cb_type", type_string).keep().help("contextual bandit method to use in {}"))
      .add(make_option("cb_force_legacy", force_legacy).keep().help("Default to non-adf cb implementation (cb_algs)"));

  options.add_parse_and_check_necessary(new_options);

  if (options.was_supplied("eval")) return nullptr;

  // ANY model created with older version should default to --cb_force_legacy
  if (all.model_file_ver != EMPTY_VERSION_FILE) compat_old_cb = !(all.model_file_ver >= VERSION_FILE_WITH_CB_TO_CBADF);

  // not compatible with adf
  if (options.was_supplied("cbify_reg")) compat_old_cb = true;

  if (force_legacy) compat_old_cb = true;

  bool override_cb = options.was_supplied("cb");
  bool override_cb_explore = options.was_supplied("cb_explore");
  bool override_cbify = options.was_supplied("cbify");

  if (!override_cb && !override_cb_explore && !override_cbify)
  {
    // do nothing
    return nullptr;
  }
  else
  {
    if (compat_old_cb)
    {
      options.insert("cb_force_legacy", "");
      // make sure cb_force_legacy gets serialized to the model on write
      options.add_and_parse(new_options);
      return nullptr;
    }
  }

  // if cb_explore_adf is being specified this is a noop
  if (override_cbify && options.was_supplied("cb_explore_adf")) return nullptr;

  if (override_cbify)
  {
    options.insert("cb_explore_adf", "");
    // no need to register custom predict/learn, cbify will take care of that
    return setup_base(options, all);
  }

  // user specified "cb_explore" but we're not using an old model file
  if (override_cb_explore)
  {
    num_actions = cbx_num_actions;
    options.insert("cb_explore_adf", "");
  }
  else  // if (override_cb) case
  {
    // force cb_adf; cb_adf will pick up cb_type
    options.insert("cb_adf", "");
  }

  auto data = scoped_calloc_or_throw<cb_to_cb_adf>();
  data->explore_mode = override_cb_explore;
  data->weights = &(all.weights);

  multi_learner* base = as_multiline(setup_base(options, all));

  learner<cb_to_cb_adf, example>* l;

  if (num_actions <= 0) THROW("cb num actions must be positive");

  data->adf_data.init_adf_data(num_actions, base->increment, all.interactions);

  // see csoaa.cc ~ line 894 / setup for csldf_setup
  all.example_parser->emptylines_separate_examples = false;

  if (data->explore_mode)
  {
    data->adf_learner = as_multiline(base->get_learner_by_name_prefix("cb_explore_adf_"));

    l = &init_learner(data, base, predict_or_learn<true>, predict_or_learn<false>, 1, prediction_type_t::action_probs,
        "cb_to_cb_adf", true);
  }
  else
  {
    data->adf_learner = as_multiline(base->get_learner_by_name_prefix("cb_adf"));

    l = &init_learner(data, base, predict_or_learn<true>, predict_or_learn<false>, 1, prediction_type_t::multiclass,
        "cb_to_cb_adf", true);
  }

  l->set_finish_example(finish_example);

  return make_base(*l);
}
