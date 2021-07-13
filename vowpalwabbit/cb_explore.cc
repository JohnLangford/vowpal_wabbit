// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore.h"
#include "reductions.h"
#include "cb_algs.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "explore.h"
#include "debug_log.h"
#include "scope_exit.h"
#include "vw_versions.h"
#include "version.h"
#include "cb_label_parser.h"
#include <cfloat>
#include "shared_data.h"

#include <memory>

using namespace VW::LEARNER;
using namespace ACTION_SCORE;
using namespace GEN_CS;
using namespace CB_ALGS;
using namespace exploration;
using namespace VW::config;
using std::endl;
// All exploration algorithms return a vector of probabilities, to be used by GenericExplorer downstream

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cb_explore

namespace CB_EXPLORE
{
struct cb_explore
{
  std::shared_ptr<rand_state> _random_state;
  cb_to_cs cbcs;
  v_array<uint32_t> preds;
  v_array<float> cover_probs;

  CB::label cb_label;
  COST_SENSITIVE::label cs_label;
  COST_SENSITIVE::label second_cs_label;

  learner<cb_explore, example>* cs;

  uint64_t tau;
  float epsilon;
  uint64_t bag_size;
  uint64_t cover_size;
  float psi;
  bool nounif;
  bool epsilon_decay;
  VW::version_struct model_file_version;

  size_t counter;
};

template <bool is_learn>
void predict_or_learn_first(cb_explore& data, single_learner& base, example& ec)
{
  // Explore tau times, then act according to optimal.
  action_scores probs = ec.pred.a_s;
  bool learn = is_learn && ec.l.cb.costs[0].probability < 1;
  if (learn)
    base.learn(ec);
  else
    base.predict(ec);

  probs.clear();
  if (data.tau > 0)
  {
    float prob = 1.f / static_cast<float>(data.cbcs.num_actions);
    for (uint32_t i = 0; i < data.cbcs.num_actions; i++) probs.push_back({i, prob});
    data.tau--;
  }
  else
  {
    uint32_t chosen = ec.pred.multiclass - 1;
    for (uint32_t i = 0; i < data.cbcs.num_actions; i++) probs.push_back({i, 0.});
    probs[chosen].score = 1.0;
  }

  ec.pred.a_s = probs;
}

template <bool is_learn>
void predict_or_learn_greedy(cb_explore& data, single_learner& base, example& ec)
{
  // Explore uniform random an epsilon fraction of the time.
  // TODO: pointers are copied here. What happens if base.learn/base.predict re-allocs?
  // ec.pred.a_s = probs; will restore the than free'd memory
  action_scores probs = ec.pred.a_s;
  probs.clear();

  if (is_learn)
    base.learn(ec);
  else
    base.predict(ec);

  // pre-allocate pdf

  VW_DBG(ec) << "cb_explore: " << (is_learn ? "learn() " : "predict() ") << multiclass_pred_to_string(ec) << endl;

  probs.reserve(data.cbcs.num_actions);
  for (uint32_t i = 0; i < data.cbcs.num_actions; i++) probs.push_back({i, 0});
  generate_epsilon_greedy(data.epsilon, ec.pred.multiclass - 1, begin_scores(probs), end_scores(probs));

  ec.pred.a_s = probs;
}

template <bool is_learn>
void predict_or_learn_bag(cb_explore& data, single_learner& base, example& ec)
{
  // Randomize over predictions from a base set of predictors
  action_scores probs = ec.pred.a_s;
  probs.clear();

  for (uint32_t i = 0; i < data.cbcs.num_actions; i++) probs.push_back({i, 0.});
  float prob = 1.f / static_cast<float>(data.bag_size);
  for (size_t i = 0; i < data.bag_size; i++)
  {
    uint32_t count = BS::weight_gen(data._random_state);
    bool learn = is_learn && count > 0;
    if (learn)
      base.learn(ec, i);
    else
      base.predict(ec, i);
    uint32_t chosen = ec.pred.multiclass - 1;
    probs[chosen].score += prob;
    if (is_learn)
      for (uint32_t j = 1; j < count; j++) base.learn(ec, i);
  }

  ec.pred.a_s = probs;
}

void get_cover_probabilities(
    cb_explore& data, single_learner& /* base */, example& ec, v_array<action_score>& probs, float min_prob)
{
  float additive_probability = 1.f / static_cast<float>(data.cover_size);
  data.preds.clear();

  for (uint32_t i = 0; i < data.cbcs.num_actions; i++) probs.push_back({i, 0.});

  for (size_t i = 0; i < data.cover_size; i++)
  {
    // get predicted cost-sensitive predictions
    if (i == 0)
      data.cs->predict(ec, i);
    else
      data.cs->predict(ec, i + 1);
    uint32_t pred = ec.pred.multiclass;
    probs[pred - 1].score += additive_probability;
    data.preds.push_back(pred);
  }
  uint32_t num_actions = data.cbcs.num_actions;

  enforce_minimum_probability(min_prob * num_actions, !data.nounif, begin_scores(probs), end_scores(probs));
}

template <bool is_learn>
void predict_or_learn_cover(cb_explore& data, single_learner& base, example& ec)
{
  VW_DBG(ec) << "predict_or_learn_cover:" << is_learn << " start" << endl;
  // Randomize over predictions from a base set of predictors
  // Use cost sensitive oracle to cover actions to form distribution.

  uint32_t num_actions = data.cbcs.num_actions;

  action_scores probs = ec.pred.a_s;
  probs.clear();
  data.cs_label.costs.clear();

  for (uint32_t j = 0; j < num_actions; j++) data.cs_label.costs.push_back({FLT_MAX, j + 1, 0., 0.});

  size_t cover_size = data.cover_size;
  v_array<float>& probabilities = data.cover_probs;
  v_array<uint32_t>& predictions = data.preds;

  float additive_probability = 1.f / static_cast<float>(cover_size);

  data.cb_label = ec.l.cb;

  // Guard example state restore against throws
  auto restore_guard = VW::scope_exit([&data, &ec] { ec.l.cb = data.cb_label; });

  ec.l.cs = data.cs_label;

  float min_prob = data.epsilon_decay
      ? std::min(data.epsilon / num_actions, data.epsilon / static_cast<float>(std::sqrt(data.counter * num_actions)))
      : data.epsilon / num_actions;

  get_cover_probabilities(data, base, ec, probs, min_prob);

  if (is_learn)
  {
    data.counter++;
    ec.l.cb = data.cb_label;
    base.learn(ec);

    // Now update oracles

    // 1. Compute loss vector
    data.cs_label.costs.clear();
    float norm = min_prob * num_actions;
    ec.l.cb = data.cb_label;
    auto optional_cost = get_observed_cost_cb(data.cb_label);
    // cost observed, not default
    if (optional_cost.first) { data.cbcs.known_cost = optional_cost.second; }
    else
    {
      data.cbcs.known_cost = CB::cb_class{};
    }
    gen_cs_example<false>(data.cbcs, ec, data.cb_label, data.cs_label);
    for (uint32_t i = 0; i < num_actions; i++) probabilities[i] = 0.f;

    ec.l.cs = std::move(data.second_cs_label);
    // 2. Update functions
    for (size_t i = 0; i < cover_size; i++)
    {
      // Create costs of each action based on online cover
      for (uint32_t j = 0; j < num_actions; j++)
      {
        float pseudo_cost =
            data.cs_label.costs[j].x - data.psi * min_prob / (std::max(probabilities[j], min_prob) / norm) + 1;
        ec.l.cs.costs[j].class_index = j + 1;
        ec.l.cs.costs[j].x = pseudo_cost;
      }
      if (i != 0) data.cs->learn(ec, i + 1);
      if (probabilities[predictions[i] - 1] < min_prob)
        norm += std::max(0.f, additive_probability - (min_prob - probabilities[predictions[i] - 1]));
      else
        norm += additive_probability;
      probabilities[predictions[i] - 1] += additive_probability;
    }
    data.second_cs_label = std::move(ec.l.cs);
  }

  ec.l.cs = COST_SENSITIVE::label{};
  ec.pred.a_s = probs;
}

void print_update_cb_explore(vw& all, bool is_test, example& ec, std::stringstream& pred_string)
{
  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.logger.quiet && !all.bfgs)
  {
    std::stringstream label_string;
    if (is_test)
      label_string << " unknown";
    else
    {
      const auto& cost = ec.l.cb.costs[0];
      label_string << cost.action << ":" << cost.cost << ":" << cost.probability;
    }
    all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass, label_string.str(),
        pred_string.str(), ec.get_num_features(), all.progress_add, all.progress_arg);
  }
}

float calc_loss(cb_explore& data, example& ec, const CB::label& ld)
{
  float loss = 0.;

  cb_to_cs& c = data.cbcs;

  auto optional_cost = CB::get_observed_cost_cb(ld);
  // cost observed, not default
  if (optional_cost.first == true)
  {
    for (uint32_t i = 0; i < ec.pred.a_s.size(); i++)
      loss += get_cost_estimate(optional_cost.second, c.pred_scores, i + 1) * ec.pred.a_s[i].score;
  }

  return loss;
}

void generic_output_example(vw& all, float loss, example& ec, CB::label& ld)
{
  all.sd->update(ec.test_only, !CB::is_test_label(ld), loss, 1.f, ec.get_num_features());

  std::stringstream ss;
  float maxprob = 0.;
  uint32_t maxid = 0;
  for (uint32_t i = 0; i < ec.pred.a_s.size(); i++)
  {
    ss << std::fixed << ec.pred.a_s[i].score << " ";
    if (ec.pred.a_s[i].score > maxprob)
    {
      maxprob = ec.pred.a_s[i].score;
      maxid = ec.pred.a_s[i].action + 1;
    }
  }
  for (auto& sink : all.final_prediction_sink) all.print_text_by_ref(sink.get(), ss.str(), ec.tag);

  std::stringstream sso;
  sso << maxid << ":" << std::fixed << maxprob;
  print_update_cb_explore(all, CB::is_test_label(ld), ec, sso);
}

void finish_example(vw& all, cb_explore& data, example& ec)
{
  float loss = calc_loss(data, ec, ec.l.cb);

  CB_EXPLORE::generic_output_example(all, loss, ec, ec.l.cb);
  VW::finish_example(all, ec);
}

void save_load(cb_explore& cb, io_buf& io, bool read, bool text)
{
  if (io.num_files() == 0) { return; }

  if (!read || cb.model_file_version >= VERSION_FILE_WITH_CCB_MULTI_SLOTS_SEEN_FLAG)
  {
    std::stringstream msg;
    if (!read) { msg << "cb cover storing example counter:  = " << cb.counter << "\n"; }
    bin_text_read_write_fixed_validated(
        io, reinterpret_cast<char*>(&cb.counter), sizeof(cb.counter), "", read, msg, text);
  }
}
}  // namespace CB_EXPLORE
using namespace CB_EXPLORE;

base_learner* cb_explore_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<cb_explore>();
  option_group_definition new_options("Contextual Bandit Exploration");
  new_options
      .add(make_option("cb_explore", data->cbcs.num_actions)
               .keep()
               .necessary()
               .help("Online explore-exploit for a <k> action contextual bandit problem"))
      .add(make_option("first", data->tau).keep().help("tau-first exploration"))
      .add(make_option("epsilon", data->epsilon)
               .keep()
               .allow_override()
               .default_value(0.05f)
               .help("epsilon-greedy exploration"))
      .add(make_option("bag", data->bag_size).keep().help("bagging-based exploration"))
      .add(make_option("cover", data->cover_size).keep().help("Online cover based exploration"))
      .add(make_option("nounif", data->nounif)
               .keep()
               .help("do not explore uniformly on zero-probability actions in cover"))
      .add(make_option("psi", data->psi).keep().default_value(1.0f).help("disagreement parameter for cover"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (!options.was_supplied("cb_force_legacy")) return nullptr;

  data->_random_state = all.get_random_state();
  uint32_t num_actions = data->cbcs.num_actions;

  // If neither cb nor cats_tree are present on the reduction stack then
  // add cb to the reduction stack as the default reduction for cb_explore.
  if (!options.was_supplied("cats_tree") && !options.was_supplied("cb"))
  {
    // none of the relevant options are set, default to cb
    std::stringstream ss;
    ss << data->cbcs.num_actions;
    options.insert("cb", ss.str());
  }

  if (data->epsilon < 0.0 || data->epsilon > 1.0) { THROW("The value of epsilon must be in [0,1]"); }

  data->cbcs.cb_type = CB_TYPE_DR;
  data->model_file_version = all.model_file_ver;

  single_learner* base = as_singleline(setup_base(options, all));
  data->cbcs.scorer = all.scorer;

  learner<cb_explore, example>* l;
  if (options.was_supplied("cover"))
  {
    if (options.was_supplied("epsilon"))
    {
      // fixed epsilon during learning
      data->epsilon_decay = false;
    }
    else
    {
      data->epsilon = 1.f;
      data->epsilon_decay = true;
    }
    data->cs = reinterpret_cast<learner<cb_explore, example>*>(as_singleline(all.cost_sensitive));
    for (uint32_t j = 0; j < num_actions; j++) { data->second_cs_label.costs.push_back(COST_SENSITIVE::wclass{}); }
    data->cover_probs.resize_but_with_stl_behavior(num_actions);
    data->preds.reserve(data->cover_size);
    l = &init_learner(data, base, predict_or_learn_cover<true>, predict_or_learn_cover<false>, data->cover_size + 1,
        prediction_type_t::action_probs, all.get_setupfn_name(cb_explore_setup) + "-cover");
  }
  else if (options.was_supplied("bag"))
    l = &init_learner(data, base, predict_or_learn_bag<true>, predict_or_learn_bag<false>, data->bag_size,
        prediction_type_t::action_probs, all.get_setupfn_name(cb_explore_setup) + "-bag");
  else if (options.was_supplied("first"))
    l = &init_learner(data, base, predict_or_learn_first<true>, predict_or_learn_first<false>, 1,
        prediction_type_t::action_probs, all.get_setupfn_name(cb_explore_setup) + "-first");
  else  // greedy
    l = &init_learner(data, base, predict_or_learn_greedy<true>, predict_or_learn_greedy<false>, 1,
        prediction_type_t::action_probs, all.get_setupfn_name(cb_explore_setup) + "-greedy");

  l->set_finish_example(finish_example);
  l->set_save_load(save_load);
  return make_base(*l);
}
