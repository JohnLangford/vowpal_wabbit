// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_synthcover.h"

#include "cb_explore_adf_common.h"
#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"
#include "explore.h"
#include "vw_versions.h"
#include "version.h"
#include "label_parser.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include "io/logger.h"

namespace logger = VW::io::logger;

// All exploration algorithms return a vector of id, probability tuples, sorted in order of scores. The probabilities
// are the probability with which each action should be replaced to the top of the list.

namespace VW
{
namespace cb_explore_adf
{
namespace synthcover
{
struct cb_explore_adf_synthcover
{
private:
  float _epsilon;
  float _psi;
  size_t _synthcoversize;
  std::shared_ptr<rand_state> _random_state;

  VW::version_struct _model_file_version;

  v_array<ACTION_SCORE::action_score> _action_probs;
  float _min_cost;
  float _max_cost;

public:
  cb_explore_adf_synthcover(float epsilon, float psi, size_t synthcoversize, std::shared_ptr<rand_state> random_state,
      VW::version_struct model_file_version);

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }
  void save_load(io_buf& model_file, bool read, bool text);

private:
  template <bool is_learn>
  void predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples);
};

cb_explore_adf_synthcover::cb_explore_adf_synthcover(float epsilon, float psi, size_t synthcoversize,
    std::shared_ptr<rand_state> random_state, VW::version_struct model_file_version)
    : _epsilon(epsilon)
    , _psi(psi)
    , _synthcoversize(synthcoversize)
    , _random_state(random_state)
    , _model_file_version(model_file_version)
    , _min_cost(0.0)
    , _max_cost(0.0)
{
}

template <bool is_learn>
void cb_explore_adf_synthcover::predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples)
{
  VW::LEARNER::multiline_learn_or_predict<is_learn>(base, examples, examples[0]->ft_offset);

  const auto it =
      std::find_if(examples.begin(), examples.end(), [](example* item) { return !item->l.cb.costs.empty(); });

  if (it != examples.end())
  {
    const CB::cb_class logged = (*it)->l.cb.costs[0];

    _min_cost = std::min(logged.cost, _min_cost);
    _max_cost = std::max(logged.cost, _max_cost);
  }

  ACTION_SCORE::action_scores& preds = examples[0]->pred.a_s;
  uint32_t num_actions = static_cast<uint32_t>(examples.size());
  if (num_actions == 0)
  {
    preds.clear();
    return;
  }
  else if (num_actions == 1)
  {
    preds[0].score = 1;
    return;
  }

  for (size_t i = 0; i < num_actions; i++)
  { preds[i].score = std::min(_max_cost, std::max(_min_cost, preds[i].score)); }
  std::make_heap(
      preds.begin(), preds.end(), [](const ACTION_SCORE::action_score& a, const ACTION_SCORE::action_score& b) {
        return ACTION_SCORE::score_comp(&a, &b) > 0;
      });

  _action_probs.clear();
  for (uint32_t i = 0; i < num_actions; i++) _action_probs.push_back({i, 0.});

  for (uint32_t i = 0; i < _synthcoversize;)
  {
    std::pop_heap(
        preds.begin(), preds.end(), [](const ACTION_SCORE::action_score& a, const ACTION_SCORE::action_score& b) {
          return ACTION_SCORE::score_comp(&a, &b) > 0;
        });
    auto minpred = preds.back();
    preds.pop_back();

    auto secondminpred = preds[0];
    for (; secondminpred.score >= minpred.score && i < _synthcoversize; i++)
    {
      _action_probs[minpred.action].score += 1.0f / _synthcoversize;
      minpred.score += (1.0f / _synthcoversize) * _psi;
    }

    preds.push_back(minpred);
    std::push_heap(
        preds.begin(), preds.end(), [](const ACTION_SCORE::action_score& a, const ACTION_SCORE::action_score& b) {
          return ACTION_SCORE::score_comp(&a, &b) > 0;
        });
  }

  exploration::enforce_minimum_probability(_epsilon, true, begin_scores(_action_probs), end_scores(_action_probs));

  std::sort(_action_probs.begin(), _action_probs.end(),
      [](const ACTION_SCORE::action_score& a, const ACTION_SCORE::action_score& b) {
        return ACTION_SCORE::score_comp(&a, &b) > 0;
      });

  for (size_t i = 0; i < num_actions; i++) preds[i] = _action_probs[i];
}

void cb_explore_adf_synthcover::save_load(io_buf& model_file, bool read, bool text)
{
  if (model_file.num_files() == 0) { return; }
  if (!read || _model_file_version >= VERSION_FILE_WITH_CCB_MULTI_SLOTS_SEEN_FLAG)
  {
    std::stringstream msg;
    if (!read) msg << "_min_cost " << _min_cost << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&_min_cost), sizeof(_min_cost), "", read, msg, text);

    if (!read) msg << "_max_cost " << _max_cost << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&_max_cost), sizeof(_max_cost), "", read, msg, text);
  }
}

VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  float epsilon = 0.;
  size_t synthcoversize;
  bool use_synthcover = false;
  float psi;
  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (synthetic cover)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("epsilon", epsilon).keep().allow_override().help("epsilon-greedy exploration"))
      .add(make_option("synthcover", use_synthcover).keep().necessary().help("use synthetic cover exploration"))
      .add(make_option("synthcoverpsi", psi)
               .keep()
               .default_value(0.1f)
               .allow_override()
               .help("exploration reward bonus"))
      .add(make_option("synthcoversize", synthcoversize)
               .keep()
               .default_value(100)
               .allow_override()
               .help("number of policies in cover"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }

  if (synthcoversize <= 0) { THROW("synthcoversize must be >= 1"); }
  if (epsilon < 0) { THROW("epsilon must be non-negative"); }
  if (psi <= 0) { THROW("synthcoverpsi must be positive"); }

  if (!all.logger.quiet)
  {
    *(all.trace_message) << "Using synthcover for CB exploration" << std::endl;
    *(all.trace_message) << "synthcoversize = " << synthcoversize << std::endl;
    if (epsilon > 0) *(all.trace_message) << "epsilon = " << epsilon << std::endl;
    *(all.trace_message) << "synthcoverpsi = " << psi << std::endl;
  }

  size_t problem_multiplier = 1;
  VW::LEARNER::multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_synthcover>;
  auto data = VW::make_unique<explore_type>(
      with_metrics, epsilon, psi, synthcoversize, all.get_random_state(), all.model_file_ver);
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-synthcover")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(explore_type::finish_multiline_example)
                .set_print_example(explore_type::print_multiline_example)
                .set_save_load(explore_type::save_load)
                .set_persist_metrics(explore_type::persist_metrics)
                .build();
  return make_base(*l);
}

}  // namespace synthcover
}  // namespace cb_explore_adf
}  // namespace VW
