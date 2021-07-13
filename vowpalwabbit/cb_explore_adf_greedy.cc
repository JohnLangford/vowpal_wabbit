// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_greedy.h"
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
#include <functional>

namespace VW
{
namespace cb_explore_adf
{
namespace greedy
{
struct cb_explore_adf_greedy
{
private:
  float _epsilon;
  bool _first_only;

public:
  cb_explore_adf_greedy(float epsilon, bool first_only);
  ~cb_explore_adf_greedy() = default;

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }

private:
  template <bool is_learn>
  void predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples);
  void update_example_prediction(multi_ex& examples);
};

cb_explore_adf_greedy::cb_explore_adf_greedy(float epsilon, bool first_only)
    : _epsilon(epsilon), _first_only(first_only)
{
}

void cb_explore_adf_greedy::update_example_prediction(multi_ex& examples)
{
  ACTION_SCORE::action_scores& preds = examples[0]->pred.a_s;

  uint32_t num_actions = static_cast<uint32_t>(preds.size());

  size_t tied_actions = fill_tied(preds);

  const float prob = _epsilon / num_actions;
  for (size_t i = 0; i < num_actions; i++) preds[i].score = prob;
  if (!_first_only)
  {
    for (size_t i = 0; i < tied_actions; ++i) preds[i].score += (1.f - _epsilon) / tied_actions;
  }
  else
    preds[0].score += 1.f - _epsilon;
}

template <bool is_learn>
void cb_explore_adf_greedy::predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples)
{
  // Explore uniform random an epsilon fraction of the time.
  if (is_learn)
    base.learn(examples);
  else
    base.predict(examples);

  update_example_prediction(examples);
}

VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  float epsilon = 0.;
  bool first_only = false;

  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (greedy)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("epsilon", epsilon).keep().allow_override().help("epsilon-greedy exploration"))
      .add(make_option("first_only", first_only).keep().help("Only explore the first action in a tie-breaking event"));

  options.add_and_parse(new_options);

  // NOTE: epsilon-greedy is the default explore type.
  // This basically runs if none of the other explore strategies are used
  bool use_greedy = !(options.was_supplied("first") || options.was_supplied("bag") || options.was_supplied("cover") ||
      options.was_supplied("regcb") || options.was_supplied("regcbopt") || options.was_supplied("squarecb") ||
      options.was_supplied("rnd") || options.was_supplied("softmax") || options.was_supplied("synthcover"));

  if (!cb_explore_adf_option || !use_greedy) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf"))
  {
    options.insert("cb_adf", "");
    options.insert("no_predict", "");
  }

  size_t problem_multiplier = 1;

  if (!options.was_supplied("epsilon")) epsilon = 0.05f;

  VW::LEARNER::multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_greedy>;
  auto data = VW::make_unique<explore_type>(with_metrics, epsilon, first_only);

  if (epsilon < 0.0 || epsilon > 1.0) { THROW("The value of epsilon must be in [0,1]"); }
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-greedy")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(explore_type::finish_multiline_example)
                .set_print_example(explore_type::print_multiline_example)
                .set_persist_metrics(explore_type::persist_metrics)
                .build();
  return make_base(*l);
}

}  // namespace greedy
}  // namespace cb_explore_adf
}  // namespace VW
