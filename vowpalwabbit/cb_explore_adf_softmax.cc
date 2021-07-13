// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_softmax.h"
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

namespace VW
{
namespace cb_explore_adf
{
namespace softmax
{
struct cb_explore_adf_softmax
{
private:
  float _epsilon;
  float _lambda;

public:
  cb_explore_adf_softmax(float epsilon, float lambda);
  ~cb_explore_adf_softmax() = default;

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }

private:
  template <bool is_learn>
  void predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples);
};

cb_explore_adf_softmax::cb_explore_adf_softmax(float epsilon, float lambda) : _epsilon(epsilon), _lambda(lambda) {}

template <bool is_learn>
void cb_explore_adf_softmax::predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples)
{
  VW::LEARNER::multiline_learn_or_predict<is_learn>(base, examples, examples[0]->ft_offset);

  v_array<ACTION_SCORE::action_score>& preds = examples[0]->pred.a_s;
  exploration::generate_softmax(
      -_lambda, begin_scores(preds), end_scores(preds), begin_scores(preds), end_scores(preds));

  exploration::enforce_minimum_probability(_epsilon, true, begin_scores(preds), end_scores(preds));
}

VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  bool softmax = false;
  float epsilon = 0.;
  float lambda = 0.;
  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (softmax)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("epsilon", epsilon).keep().allow_override().help("epsilon-greedy exploration"))
      .add(make_option("softmax", softmax).keep().necessary().help("softmax exploration"))
      .add(make_option("lambda", lambda).keep().allow_override().default_value(1.f).help("parameter for softmax"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (lambda < 0)  // Lambda should always be positive because we are using a cost basis.
    lambda = -lambda;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }

  // Set explore_type
  size_t problem_multiplier = 1;

  VW::LEARNER::multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_softmax>;
  auto data = VW::make_unique<explore_type>(with_metrics, epsilon, lambda);

  if (epsilon < 0.0 || epsilon > 1.0) { THROW("The value of epsilon must be in [0,1]"); }
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-softmax")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(explore_type::finish_multiline_example)
                .set_print_example(explore_type::print_multiline_example)
                .set_persist_metrics(explore_type::persist_metrics)
                .build();
  return make_base(*l);
}
}  // namespace softmax
}  // namespace cb_explore_adf
}  // namespace VW
