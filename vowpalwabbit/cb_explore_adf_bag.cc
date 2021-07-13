// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_bag.h"

#include "cb_explore_adf_common.h"
#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"
#include "explore.h"
#include "label_parser.h"
#include <vector>
#include <algorithm>
#include <cmath>

// All exploration algorithms return a vector of id, probability tuples, sorted in order of scores. The probabilities
// are the probability with which each action should be replaced to the top of the list.

namespace VW
{
namespace cb_explore_adf
{
namespace bag
{
struct cb_explore_adf_bag
{
private:
  float _epsilon;
  size_t _bag_size;
  bool _greedify;
  bool _first_only;
  std::shared_ptr<rand_state> _random_state;

  v_array<ACTION_SCORE::action_score> _action_probs;
  std::vector<float> _scores;
  std::vector<float> _top_actions;

public:
  using PredictionT = v_array<ACTION_SCORE::action_score>;

  cb_explore_adf_bag(
      float epsilon, size_t bag_size, bool greedify, bool first_only, std::shared_ptr<rand_state> random_state);

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(VW::LEARNER::multi_learner &base, multi_ex &examples);
  void learn(VW::LEARNER::multi_learner &base, multi_ex &examples);

  const PredictionT &get_cached_prediction() { return _action_probs; };

private:
  uint32_t get_bag_learner_update_count(uint32_t learner_index);
};

cb_explore_adf_bag::cb_explore_adf_bag(
    float epsilon, size_t bag_size, bool greedify, bool first_only, std::shared_ptr<rand_state> random_state)
    : _epsilon(epsilon), _bag_size(bag_size), _greedify(greedify), _first_only(first_only), _random_state(random_state)
{
}

uint32_t cb_explore_adf_bag::get_bag_learner_update_count(uint32_t learner_index)
{
  // If _greedify then always update the first policy once
  // for others the update count depends on drawing from a poisson
  if (_greedify && learner_index == 0)
    return 1;
  else
    return BS::weight_gen(_random_state);
}

void cb_explore_adf_bag::predict(VW::LEARNER::multi_learner &base, multi_ex &examples)
{
  // Randomize over predictions from a base set of predictors
  v_array<ACTION_SCORE::action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = static_cast<uint32_t>(examples.size());
  if (num_actions == 0)
  {
    preds.clear();
    return;
  }

  _scores.assign(num_actions, 0.f);
  _top_actions.assign(num_actions, 0);

  for (uint32_t i = 0; i < _bag_size; i++)
  {
    VW::LEARNER::multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset, i);

    assert(preds.size() == num_actions);
    for (auto e : preds) _scores[e.action] += e.score;

    if (!_first_only)
    {
      size_t tied_actions = fill_tied(preds);
      for (size_t j = 0; j < tied_actions; ++j) _top_actions[preds[j].action] += 1.f / tied_actions;
    }
    else
      _top_actions[preds[0].action] += 1.f;
  }

  _action_probs.clear();
  for (uint32_t i = 0; i < _scores.size(); i++) _action_probs.push_back({i, 0.});

  // generate distribution over actions
  exploration::generate_bag(
      begin(_top_actions), end(_top_actions), begin_scores(_action_probs), end_scores(_action_probs));

  exploration::enforce_minimum_probability(_epsilon, true, begin_scores(_action_probs), end_scores(_action_probs));
  sort_action_probs(_action_probs, _scores);
  std::copy(std::begin(_action_probs), std::end(_action_probs), std::begin(preds));
}

void cb_explore_adf_bag::learn(VW::LEARNER::multi_learner &base, multi_ex &examples)
{
  for (uint32_t i = 0; i < _bag_size; i++)
  {
    // learn_count determines how many times learner (i) will learn from this
    // example.
    uint32_t learn_count = get_bag_learner_update_count(i);

    VW_DBG(examples) << "cb_explore_adf_bag::learn, bag_learner_idx = " << i << ", learn_count = " << learn_count
                     << std::endl;

    for (uint32_t j = 0; j < learn_count; j++)
      VW::LEARNER::multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset, i);
  }
}

void finish_bag_example(vw &all, cb_explore_adf_base<cb_explore_adf_bag> &data, multi_ex &ec_seq)
{
  assert(ec_seq.size() > 0);
  ec_seq[0]->pred.a_s = data.explore.get_cached_prediction();
  cb_explore_adf_base<cb_explore_adf_bag>::finish_multiline_example(all, data, ec_seq);
}

void print_bag_example(vw& all, cb_explore_adf_base<cb_explore_adf_bag>& data, multi_ex& ec_seq)
{
  assert(ec_seq.size() > 0);
  ec_seq[0]->pred.a_s = data.explore.get_cached_prediction();
  cb_explore_adf_base<cb_explore_adf_bag>::print_multiline_example(all, data, ec_seq);
}

VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  float epsilon = 0.;
  size_t bag_size = 0;
  bool greedify = false;
  bool first_only = false;
  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (bagging)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("epsilon", epsilon).keep().allow_override().help("epsilon-greedy exploration"))
      .add(make_option("bag", bag_size).keep().necessary().help("bagging-based exploration"))
      .add(make_option("greedify", greedify).keep().help("always update first policy once in bagging"))
      .add(make_option("first_only", first_only).keep().help("Only explore the first action in a tie-breaking event"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }

  // Signal cb_adf MTR to not predict when training.  The framework already
  // handles calling
  // predict before training is called.
  if (!options.was_supplied("no_predict")) { options.insert("no_predict", ""); }

  size_t problem_multiplier = bag_size;
  VW::LEARNER::multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_bag>;
  auto data =
      VW::make_unique<explore_type>(with_metrics, epsilon, bag_size, greedify, first_only, all.get_random_state());
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-bag")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(finish_bag_example)
                .set_print_example(print_bag_example)
                .set_persist_metrics(explore_type::persist_metrics)
                .build();
  return make_base(*l);
}

}  // namespace bag
}  // namespace cb_explore_adf
}  // namespace VW
