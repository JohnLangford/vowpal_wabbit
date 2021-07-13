// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_cover.h"
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

// All exploration algorithms return a vector of id, probability tuples, sorted in order of scores. The probabilities
// are the probability with which each action should be replaced to the top of the list.

namespace VW
{
namespace cb_explore_adf
{
namespace cover
{
struct cb_explore_adf_cover
{
private:
  size_t _cover_size;
  float _psi;
  bool _nounif;
  float _epsilon;
  bool _epsilon_decay;
  bool _first_only;
  size_t _counter;

  VW::LEARNER::multi_learner* _cs_ldf_learner;
  GEN_CS::cb_to_cs_adf _gen_cs;

  VW::version_struct _model_file_version;

  v_array<ACTION_SCORE::action_score> _action_probs;
  std::vector<float> _scores;
  COST_SENSITIVE::label _cs_labels;
  COST_SENSITIVE::label _cs_labels_2;
  std::vector<COST_SENSITIVE::label> _prepped_cs_labels;
  std::vector<CB::label> _cb_labels;

public:
  cb_explore_adf_cover(size_t cover_size, float psi, bool nounif, float epsilon, bool epsilon_decay, bool first_only,
      VW::LEARNER::multi_learner* cs_ldf_learner, VW::LEARNER::single_learner* scorer, size_t cb_type,
      VW::version_struct model_file_version);

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }
  void save_load(io_buf& io, bool read, bool text);

private:
  template <bool is_learn>
  void predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples);
};

cb_explore_adf_cover::cb_explore_adf_cover(size_t cover_size, float psi, bool nounif, float epsilon, bool epsilon_decay,
    bool first_only, VW::LEARNER::multi_learner* cs_ldf_learner, VW::LEARNER::single_learner* scorer, size_t cb_type,
    VW::version_struct model_file_version)
    : _cover_size(cover_size)
    , _psi(psi)
    , _nounif(nounif)
    , _epsilon(epsilon)
    , _epsilon_decay(epsilon_decay)
    , _first_only(first_only)
    , _counter(0)
    , _cs_ldf_learner(cs_ldf_learner)
    , _model_file_version(std::move(model_file_version))
{
  _gen_cs.cb_type = cb_type;
  _gen_cs.scorer = scorer;
}

template <bool is_learn>
void cb_explore_adf_cover::predict_or_learn_impl(VW::LEARNER::multi_learner& base, multi_ex& examples)
{
  // Redundant with the call in cb_explore_adf_base, but encapsulation means we need to do this again here
  _gen_cs.known_cost = CB_ADF::get_observed_cost_or_default_cb_adf(examples);

  // Randomize over predictions from a base set of predictors
  // Use cost sensitive oracle to cover actions to form distribution.
  const bool is_mtr = _gen_cs.cb_type == CB_TYPE_MTR;
  if (is_learn)
  {
    if (is_mtr)  // use DR estimates for non-ERM policies in MTR
      GEN_CS::gen_cs_example_dr<true>(_gen_cs, examples, _cs_labels);
    else
      GEN_CS::gen_cs_example<false>(_gen_cs, examples, _cs_labels);

    if (base.learn_returns_prediction)
    {
      // First predict() since the result of the predictions are used to learn
      // later in the reduction
      VW_DBG(examples) << "cb_explore_adf_cover: "
                          "LEARNER::multiline_learn_or_predict<false>()"
                       << std::endl;
      VW::LEARNER::multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);
    }

    VW_DBG(examples) << "cb_explore_adf_cover: LEARNER::multiline_learn_or_predict<true>()" << std::endl;
    VW::LEARNER::multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset);
  }
  else
  {
    GEN_CS::gen_cs_example_ips(examples, _cs_labels);
    VW::LEARNER::multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);
  }
  v_array<ACTION_SCORE::action_score>& preds = examples[0]->pred.a_s;
  const uint32_t num_actions = static_cast<uint32_t>(preds.size());

  float additive_probability = 1.f / static_cast<float>(_cover_size);

  float min_prob = _epsilon_decay
      ? std::min(_epsilon / num_actions, _epsilon / static_cast<float>(std::sqrt(_counter * num_actions)))
      : _epsilon / num_actions;

  _action_probs.clear();
  for (uint32_t i = 0; i < num_actions; i++) _action_probs.push_back({i, 0.});
  _scores.clear();
  for (uint32_t i = 0; i < num_actions; i++) _scores.push_back(preds[i].score);

  if (!_first_only)
  {
    size_t tied_actions = fill_tied(preds);
    for (size_t i = 0; i < tied_actions; ++i)
      _action_probs[preds[i].action].score += additive_probability / tied_actions;
  }
  else
    _action_probs[preds[0].action].score += additive_probability;

  float norm = min_prob * num_actions + (additive_probability - min_prob);
  for (size_t i = 1; i < _cover_size; i++)
  {
    // Create costs of each action based on online cover
    if (is_learn)
    {
      _cs_labels_2.costs.clear();
      /* Cover's learn policy is similar to bag in that we have multiple learners
       * The main difference here is that Cover's learners interact with each other.
       * The following code generates cost = cost + penalty where a penalty is applied
       * to any action a previous policy has already selected
       */
      for (uint32_t j = 0; j < num_actions; j++)
      {
        float pseudo_cost =
            _cs_labels.costs[j].x - _psi * min_prob / ((std::max)(_action_probs[j].score, min_prob) / norm);
        _cs_labels_2.costs.push_back({pseudo_cost, j, 0., 0.});
      }

      GEN_CS::cs_ldf_learn_or_predict<true>(*(_cs_ldf_learner), examples, _cb_labels, _cs_labels_2, _prepped_cs_labels,
          true, examples[0]->ft_offset, i + 1);
    }
    else
      GEN_CS::cs_ldf_learn_or_predict<false>(*(_cs_ldf_learner), examples, _cb_labels, _cs_labels, _prepped_cs_labels,
          false, examples[0]->ft_offset, i + 1);

    for (uint32_t j = 0; j < num_actions; j++) _scores[j] += preds[j].score;
    if (!_first_only)
    {
      size_t tied_actions = fill_tied(preds);
      const float add_prob = additive_probability / tied_actions;
      for (size_t j = 0; j < tied_actions; ++j)
      {
        if (_action_probs[preds[j].action].score < min_prob)
          norm += (std::max)(0.f, add_prob - (min_prob - _action_probs[preds[j].action].score));
        else
          norm += add_prob;
        _action_probs[preds[j].action].score += add_prob;
      }
    }
    else
    {
      uint32_t action = preds[0].action;
      if (_action_probs[action].score < min_prob)
        norm += (std::max)(0.f, additive_probability - (min_prob - _action_probs[action].score));
      else
        norm += additive_probability;
      _action_probs[action].score += additive_probability;
    }
  }

  exploration::enforce_minimum_probability(
      min_prob * num_actions, !_nounif, begin_scores(_action_probs), end_scores(_action_probs));

  sort_action_probs(_action_probs, _scores);
  for (size_t i = 0; i < num_actions; i++) preds[i] = _action_probs[i];

  if (VW_DEBUG_LOG)
  {
    VW_DBG(examples) << "a_p[]=";
    for (auto const& ap : _action_probs) VW_DBG_0 << ap.action << "::" << ap.score << ",";
    VW_DBG_0 << std::endl;

    VW_DBG(examples) << "scores[]=";
    for (auto const& s : _scores) VW_DBG_0 << s << ",";
    VW_DBG_0 << std::endl;
  }

  if (is_learn) ++_counter;
}

void cb_explore_adf_cover::save_load(io_buf& io, bool read, bool text)
{
  if (io.num_files() == 0) { return; }
  if (!read || _model_file_version >= VERSION_FILE_WITH_CCB_MULTI_SLOTS_SEEN_FLAG)
  {
    std::stringstream msg;
    if (!read) { msg << "cb cover adf storing example counter:  = " << _counter << "\n"; }
    bin_text_read_write_fixed_validated(io, reinterpret_cast<char*>(&_counter), sizeof(_counter), "", read, msg, text);
  }
}

VW::LEARNER::base_learner* setup(VW::setup_base_fn& setup_base, config::options_i& options, vw& all)
{
  using config::make_option;

  bool cb_explore_adf_option = false;
  std::string type_string = "mtr";
  size_t cover_size = 0;
  float psi = 0.;
  bool nounif = false;
  bool first_only = false;
  float epsilon = 0.;

  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (online cover)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("cover", cover_size).keep().necessary().help("Online cover based exploration"))
      .add(make_option("psi", psi).keep().default_value(1.0f).help("disagreement parameter for cover"))
      .add(make_option("nounif", nounif).keep().help("do not explore uniformly on zero-probability actions in cover"))
      .add(make_option("first_only", first_only).keep().help("Only explore the first action in a tie-breaking event"))
      .add(make_option("cb_type", type_string)
               .keep()
               .help("contextual bandit method to use in {ips,dr,mtr}. Default: mtr"))
      .add(make_option("epsilon", epsilon)
               .keep()
               .allow_override()
               .default_value(0.05f)
               .help("epsilon-greedy exploration"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_type in all cases.
  if (!options.was_supplied("cb_type"))
  {
    options.insert("cb_type", type_string);
    options.add_and_parse(new_options);
  }

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }

  // Set cb_type
  size_t cb_type_enum;
  if (type_string.compare("dr") == 0)
    cb_type_enum = CB_TYPE_DR;
  else if (type_string.compare("ips") == 0)
    cb_type_enum = CB_TYPE_IPS;
  else if (type_string.compare("mtr") == 0)
  {
    *(all.trace_message) << "warning: currently, mtr is only used for the first policy in cover, other policies use dr"
                         << std::endl;
    cb_type_enum = CB_TYPE_MTR;
  }
  else
  {
    *(all.trace_message) << "warning: cb_type must be in {'ips','dr','mtr'}; resetting to mtr." << std::endl;
    options.replace("cb_type", "mtr");
    cb_type_enum = CB_TYPE_MTR;
  }

  // Set explore_type
  size_t problem_multiplier = cover_size + 1;

  // Cover is using doubly robust without the cooperation of the base reduction
  if (cb_type_enum == CB_TYPE_MTR) { problem_multiplier *= 2; }

  VW::LEARNER::multi_learner* base = VW::LEARNER::as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool epsilon_decay;
  if (options.was_supplied("epsilon"))
  {
    // fixed epsilon during learning
    epsilon_decay = false;
  }
  else
  {
    epsilon = 1.f;
    epsilon_decay = true;
  }

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_cover>;
  auto data = VW::make_unique<explore_type>(with_metrics, cover_size, psi, nounif, epsilon, epsilon_decay, first_only,
      as_multiline(all.cost_sensitive), all.scorer, cb_type_enum, all.model_file_ver);
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-cover")
                .set_learn_returns_prediction(true)
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
}  // namespace cover
}  // namespace cb_explore_adf
}  // namespace VW
