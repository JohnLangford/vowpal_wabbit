// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cfloat>
#include "reductions.h"
#include "cb_algs.h"
#include "rand48.h"
#include "bs.h"
#include "vw.h"
#include "hash.h"
#include "explore.h"
#include "vw_exception.h"
#include "scope_exit.h"
#include "cb_label_parser.h"

#include "io/logger.h"

#include <vector>
#include <memory>

using namespace VW::LEARNER;
using namespace exploration;
using namespace ACTION_SCORE;
using namespace VW::config;

namespace logger = VW::io::logger;

#define WARM_START 1
#define INTERACTION 2
#define SKIP 3

#define SUPERVISED_WS 1
#define BANDIT_WS 2

#define UAR 1
#define CIRCULAR 2
#define OVERWRITE 3

#define ABS_CENTRAL 1
#define ABS_CENTRAL_ZEROONE 2
#define MINIMAX_CENTRAL 3
#define MINIMAX_CENTRAL_ZEROONE 4

struct warm_cb
{
  CB::label cb_label;
  uint64_t app_seed;
  action_scores a_s;
  // used as the seed
  size_t example_counter;
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  multi_ex ecs;
  float loss0;
  float loss1;

  // warm start parameters
  uint32_t ws_period;
  uint32_t inter_period;
  uint32_t choices_lambda;
  bool upd_ws;
  bool upd_inter;
  int cor_type_ws;
  float cor_prob_ws;
  int vali_method;
  int wt_scheme;
  int lambda_scheme;
  uint32_t overwrite_label;
  int ws_type;
  bool sim_bandit;

  // auxiliary variables
  uint32_t num_actions;
  float epsilon;
  std::vector<float> lambdas;
  action_scores a_s_adf;
  std::vector<float> cumulative_costs;
  CB::cb_class cl_adf;
  uint32_t ws_train_size;
  uint32_t ws_vali_size;
  std::vector<example*> ws_vali;
  float cumu_var;
  uint32_t ws_iter;
  uint32_t inter_iter;
  MULTICLASS::label_t mc_label;
  COST_SENSITIVE::label cs_label;
  std::vector<COST_SENSITIVE::label> csls;
  std::vector<CB::label> cbls;
  bool use_cs;

  ~warm_cb()
  {
    for (size_t a = 0; a < num_actions; ++a) { VW::dealloc_examples(ecs[a], 1); }

    for (auto* ex : ws_vali) { VW::dealloc_examples(ex, 1); }
  }
};

float loss(warm_cb& data, uint32_t label, uint32_t final_prediction)
{
  if (label != final_prediction)
    return data.loss1;
  else
    return data.loss0;
}

float loss_cs(warm_cb& data, v_array<COST_SENSITIVE::wclass>& costs, uint32_t final_prediction)
{
  float cost = 0.;
  for (auto wc : costs)
  {
    if (wc.class_index == final_prediction)
    {
      cost = wc.x;
      break;
    }
  }
  return data.loss0 + (data.loss1 - data.loss0) * cost;
}

template <class T>
uint32_t find_min(std::vector<T> arr)
{
  T min_val = FLT_MAX;
  uint32_t argmin = 0;

  for (uint32_t i = 0; i < arr.size(); i++)
  {
    if (arr[i] < min_val)
    {
      min_val = arr[i];
      argmin = i;
    }
  }
  return argmin;
}

void finish(warm_cb& data)
{
  uint32_t argmin = find_min(data.cumulative_costs);

  if (!data.all->logger.quiet)
  {
    *(data.all->trace_message) << "average variance estimate = " << data.cumu_var / data.inter_iter << std::endl;
    *(data.all->trace_message) << "theoretical average variance = " << data.num_actions / data.epsilon << std::endl;
    *(data.all->trace_message) << "last lambda chosen = " << data.lambdas[argmin] << " among lambdas ranging from "
                               << data.lambdas[0] << " to " << data.lambdas[data.choices_lambda - 1] << std::endl;
  }
}

void copy_example_to_adf(warm_cb& data, example& ec)
{
  const uint64_t ss = data.all->weights.stride_shift();
  const uint64_t mask = data.all->weights.mask();

  for (size_t a = 0; a < data.num_actions; ++a)
  {
    auto& eca = *data.ecs[a];
    // clear label
    auto& lab = eca.l.cb;
    CB::default_label(lab);

    // copy data
    VW::copy_example_data(&eca, &ec);

    // offset indicies for given action
    for (features& fs : eca)
    {
      for (feature_index& idx : fs.indicies)
      { idx = ((((idx >> ss) * 28904713) + 4832917 * static_cast<uint64_t>(a)) << ss) & mask; }
    }

    // avoid empty example by adding a tag (hacky)
    if (CB_ALGS::example_is_newline_not_header(eca) && CB::cb_label.test_label(&eca.l)) { eca.tag.push_back('n'); }
  }
}

// Changing the minimax value from eps/(K+eps)
// to eps/(1+eps) to accomodate for
// weight scaling of bandit examples by factor 1/K in mtr reduction
float minimax_lambda(float epsilon) { return epsilon / (1.0f + epsilon); }

void setup_lambdas(warm_cb& data)
{
  // The lambdas are arranged in ascending order
  std::vector<float>& lambdas = data.lambdas;
  for (uint32_t i = 0; i < data.choices_lambda; i++) lambdas.push_back(0.f);

  // interaction only: set all lambda's to be identically 1
  if (!data.upd_ws && data.upd_inter)
  {
    for (uint32_t i = 0; i < data.choices_lambda; i++) lambdas[i] = 1.0;
    return;
  }

  // warm start only: set all lambda's to be identically 0
  if (!data.upd_inter && data.upd_ws)
  {
    for (uint32_t i = 0; i < data.choices_lambda; i++) lambdas[i] = 0.0;
    return;
  }

  uint32_t mid = data.choices_lambda / 2;

  if (data.lambda_scheme == ABS_CENTRAL || data.lambda_scheme == ABS_CENTRAL_ZEROONE)
    lambdas[mid] = 0.5;
  else
    lambdas[mid] = minimax_lambda(data.epsilon);

  for (uint32_t i = mid; i > 0; i--) lambdas[i - 1] = lambdas[i] / 2.0f;

  for (uint32_t i = mid + 1; i < data.choices_lambda; i++) lambdas[i] = 1.f - (1.f - lambdas[i - 1]) / 2.0f;

  if (data.lambda_scheme == MINIMAX_CENTRAL_ZEROONE || data.lambda_scheme == ABS_CENTRAL_ZEROONE)
  {
    lambdas[0] = 0.0;
    lambdas[data.choices_lambda - 1] = 1.0;
  }
}

uint32_t generate_uar_action(warm_cb& data)
{
  float randf = data._random_state->get_and_update_random();

  for (uint32_t i = 1; i <= data.num_actions; i++)
  {
    if (randf <= float(i) / data.num_actions) return i;
  }
  return data.num_actions;
}

uint32_t corrupt_action(warm_cb& data, uint32_t action, int ec_type)
{
  float cor_prob = 0.;
  uint32_t cor_type = UAR;
  uint32_t cor_action;

  if (ec_type == WARM_START)
  {
    cor_prob = data.cor_prob_ws;
    cor_type = data.cor_type_ws;
  }

  float randf = data._random_state->get_and_update_random();
  if (randf < cor_prob)
  {
    if (cor_type == UAR)
      cor_action = generate_uar_action(data);
    else if (cor_type == OVERWRITE)
      cor_action = data.overwrite_label;
    else
      cor_action = (action % data.num_actions) + 1;
  }
  else
    cor_action = action;
  return cor_action;
}

bool ind_update(warm_cb& data, int ec_type)
{
  if (ec_type == WARM_START)
    return data.upd_ws;
  else
    return data.upd_inter;
}

float compute_weight_multiplier(warm_cb& data, size_t i, int ec_type)
{
  float weight_multiplier;
  float ws_train_size = static_cast<float>(data.ws_train_size);
  float inter_train_size = static_cast<float>(data.inter_period);
  float total_train_size = ws_train_size + inter_train_size;
  float total_weight = (1 - data.lambdas[i]) * ws_train_size + data.lambdas[i] * inter_train_size;

  if (ec_type == WARM_START)
    weight_multiplier = (1 - data.lambdas[i]) * total_train_size / (total_weight + FLT_MIN);
  else
    weight_multiplier = data.lambdas[i] * total_train_size / (total_weight + FLT_MIN);

  return weight_multiplier;
}

uint32_t predict_sublearner_adf(warm_cb& data, multi_learner& base, example& ec, uint32_t i)
{
  copy_example_to_adf(data, ec);
  base.predict(data.ecs, i);
  return data.ecs[0]->pred.a_s[0].action + 1;
}

void accumu_costs_iv_adf(warm_cb& data, multi_learner& base, example& ec)
{
  CB::cb_class& cl = data.cl_adf;
  // IPS for approximating the cumulative costs for all lambdas
  for (uint32_t i = 0; i < data.choices_lambda; i++)
  {
    uint32_t action = predict_sublearner_adf(data, base, ec, i);

    if (action == cl.action) data.cumulative_costs[i] += cl.cost / cl.probability;
  }
}

template <bool use_cs>
void add_to_vali(warm_cb& data, example& ec)
{
  // TODO: set the first parameter properly
  example* ec_copy = VW::alloc_examples(1);
  VW::copy_example_data_with_label(ec_copy, &ec);
  data.ws_vali.push_back(ec_copy);
}

uint32_t predict_sup_adf(warm_cb& data, multi_learner& base, example& ec)
{
  uint32_t argmin = find_min(data.cumulative_costs);
  return predict_sublearner_adf(data, base, ec, argmin);
}

template <bool use_cs>
void learn_sup_adf(warm_cb& data, example& ec, int ec_type)
{
  copy_example_to_adf(data, ec);
  // generate cost-sensitive label (for cost-sensitive learner's temporary use)
  auto& csls = data.csls;
  auto& cbls = data.cbls;
  for (uint32_t a = 0; a < data.num_actions; ++a)
  {
    csls[a].costs[0].class_index = a + 1;
    if (use_cs)
      csls[a].costs[0].x = loss_cs(data, ec.l.cs.costs, a + 1);
    else
      csls[a].costs[0].x = loss(data, ec.l.multi.label, a + 1);
  }
  for (size_t a = 0; a < data.num_actions; ++a)
  {
    cbls[a] = data.ecs[a]->l.cb;
    data.ecs[a]->l.cs = csls[a];
  }

  std::vector<float> old_weights;
  for (size_t a = 0; a < data.num_actions; ++a) old_weights.push_back(data.ecs[a]->weight);

  for (uint32_t i = 0; i < data.choices_lambda; i++)
  {
    float weight_multiplier = compute_weight_multiplier(data, i, ec_type);
    for (size_t a = 0; a < data.num_actions; ++a) data.ecs[a]->weight = old_weights[a] * weight_multiplier;
    multi_learner* cs_learner = as_multiline(data.all->cost_sensitive);
    cs_learner->learn(data.ecs, i);
  }

  for (size_t a = 0; a < data.num_actions; ++a) data.ecs[a]->weight = old_weights[a];

  for (size_t a = 0; a < data.num_actions; ++a) data.ecs[a]->l.cb = cbls[a];
}

template <bool use_cs>
void predict_or_learn_sup_adf(warm_cb& data, multi_learner& base, example& ec, int ec_type)
{
  uint32_t action = predict_sup_adf(data, base, ec);

  if (ind_update(data, ec_type)) learn_sup_adf<use_cs>(data, ec, ec_type);

  ec.pred.multiclass = action;
}

uint32_t predict_bandit_adf(warm_cb& data, multi_learner& base, example& ec)
{
  uint32_t argmin = find_min(data.cumulative_costs);

  copy_example_to_adf(data, ec);
  base.predict(data.ecs, argmin);

  auto& out_ec = *data.ecs[0];
  uint32_t chosen_action;
  if (sample_after_normalizing(data.app_seed + data.example_counter++, begin_scores(out_ec.pred.a_s),
          end_scores(out_ec.pred.a_s), chosen_action))
    THROW("Failed to sample from pdf");

  auto& a_s = data.a_s_adf;
  a_s = out_ec.pred.a_s;

  return chosen_action;
}

void learn_bandit_adf(warm_cb& data, multi_learner& base, example& ec, int ec_type)
{
  copy_example_to_adf(data, ec);

  // add cb label to chosen action
  auto& cl = data.cl_adf;
  auto& lab = data.ecs[cl.action - 1]->l.cb;
  lab.costs.push_back(cl);

  std::vector<float> old_weights;
  for (size_t a = 0; a < data.num_actions; ++a) old_weights.push_back(data.ecs[a]->weight);

  // Guard example state restore against throws
  auto restore_guard = VW::scope_exit([&old_weights, &data] {
    for (size_t a = 0; a < data.num_actions; ++a) { data.ecs[a]->weight = old_weights[a]; }
  });

  for (uint32_t i = 0; i < data.choices_lambda; i++)
  {
    float weight_multiplier = compute_weight_multiplier(data, i, ec_type);
    for (size_t a = 0; a < data.num_actions; ++a) data.ecs[a]->weight = old_weights[a] * weight_multiplier;
    base.learn(data.ecs, i);
  }
}

template <bool use_cs>
void predict_or_learn_bandit_adf(warm_cb& data, multi_learner& base, example& ec, int ec_type)
{
  uint32_t chosen_action = predict_bandit_adf(data, base, ec);

  auto& cl = data.cl_adf;
  auto& a_s = data.a_s_adf;
  cl.action = a_s[chosen_action].action + 1;
  cl.probability = a_s[chosen_action].score;

  if (!cl.action) THROW("No action with non-zero probability found!");

  if (use_cs)
    cl.cost = loss_cs(data, ec.l.cs.costs, cl.action);
  else
    cl.cost = loss(data, ec.l.multi.label, cl.action);

  if (ec_type == INTERACTION) accumu_costs_iv_adf(data, base, ec);

  if (ind_update(data, ec_type)) learn_bandit_adf(data, base, ec, ec_type);

  ec.pred.multiclass = cl.action;
}

void accumu_var_adf(warm_cb& data, multi_learner& base, example& ec)
{
  size_t pred_best_approx = predict_sup_adf(data, base, ec);
  float temp_var = 0.f;

  for (size_t a = 0; a < data.num_actions; ++a)
    if (pred_best_approx == data.a_s_adf[a].action + 1) temp_var = 1.0f / data.a_s_adf[a].score;

  data.cumu_var += temp_var;
}

template <bool use_cs>
void predict_and_learn_adf(warm_cb& data, multi_learner& base, example& ec)
{
  // Corrupt labels (only corrupting multiclass labels as of now)
  if (use_cs)
    data.cs_label = ec.l.cs;
  else
  {
    data.mc_label = ec.l.multi;
    if (data.ws_iter < data.ws_period) ec.l.multi.label = corrupt_action(data, data.mc_label.label, WARM_START);
  }

  // Warm start phase
  if (data.ws_iter < data.ws_period)
  {
    if (data.ws_type == SUPERVISED_WS)
      predict_or_learn_sup_adf<use_cs>(data, base, ec, WARM_START);
    else if (data.ws_type == BANDIT_WS)
      predict_or_learn_bandit_adf<use_cs>(data, base, ec, WARM_START);

    ec.weight = 0;
    data.ws_iter++;
  }
  // Interaction phase
  else if (data.inter_iter < data.inter_period)
  {
    predict_or_learn_bandit_adf<use_cs>(data, base, ec, INTERACTION);
    accumu_var_adf(data, base, ec);
    data.a_s_adf.clear();
    data.inter_iter++;
  }
  // Skipping the rest of the examples
  else
  {
    ec.weight = 0;
    ec.pred.multiclass = 1;
  }

  // Restore the original labels
  if (use_cs)
    ec.l.cs = data.cs_label;
  else
    ec.l.multi = data.mc_label;
}

void init_adf_data(warm_cb& data, const uint32_t num_actions)
{
  data.num_actions = num_actions;
  if (data.sim_bandit)
    data.ws_type = BANDIT_WS;
  else
    data.ws_type = SUPERVISED_WS;
  data.ecs.resize(num_actions);
  for (size_t a = 0; a < num_actions; ++a)
  {
    data.ecs[a] = VW::alloc_examples(1);
    auto& lab = data.ecs[a]->l.cb;
    CB::default_label(lab);
  }

  // The rest of the initialization is for warm start CB
  data.csls.resize(num_actions);
  for (uint32_t a = 0; a < num_actions; ++a)
  {
    COST_SENSITIVE::default_label(data.csls[a]);
    data.csls[a].costs.push_back({0, a + 1, 0, 0});
  }
  data.cbls.resize(num_actions);

  data.ws_train_size = data.ws_period;
  data.ws_vali_size = 0;

  data.ws_iter = 0;
  data.inter_iter = 0;

  setup_lambdas(data);
  for (uint32_t i = 0; i < data.choices_lambda; i++) data.cumulative_costs.push_back(0.f);
  data.cumu_var = 0.f;
}

base_learner* warm_cb_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  uint32_t num_actions = 0;
  auto data = scoped_calloc_or_throw<warm_cb>();
  bool use_cs;

  option_group_definition new_options("Make Multiclass into Warm-starting Contextual Bandit");

  new_options
      .add(make_option("warm_cb", num_actions)
               .keep()
               .necessary()
               .help("Convert multiclass on <k> classes into a contextual bandit problem"))
      .add(make_option("warm_cb_cs", use_cs)
               .help("consume cost-sensitive classification examples instead of multiclass"))
      .add(make_option("loss0", data->loss0).default_value(0.f).help("loss for correct label"))
      .add(make_option("loss1", data->loss1).default_value(1.f).help("loss for incorrect label"))
      .add(make_option("warm_start", data->ws_period)
               .default_value(0U)
               .help("number of training examples for warm start phase"))
      .add(make_option("epsilon", data->epsilon).keep().allow_override().help("epsilon-greedy exploration"))
      .add(make_option("interaction", data->inter_period)
               .default_value(UINT32_MAX)
               .help("number of examples for the interactive contextual bandit learning phase"))
      .add(make_option("warm_start_update", data->upd_ws).help("indicator of warm start updates"))
      .add(make_option("interaction_update", data->upd_inter).help("indicator of interaction updates"))
      .add(make_option("corrupt_type_warm_start", data->cor_type_ws)
               .default_value(UAR)
               .help("type of label corruption in the warm start phase (1: uniformly at random, 2: circular, 3: "
                     "replacing with overwriting label)"))
      .add(make_option("corrupt_prob_warm_start", data->cor_prob_ws)
               .default_value(0.f)
               .help("probability of label corruption in the warm start phase"))
      .add(make_option("choices_lambda", data->choices_lambda)
               .default_value(1U)
               .help("the number of candidate lambdas to aggregate (lambda is the importance weight parameter between "
                     "the two sources)"))
      .add(make_option("lambda_scheme", data->lambda_scheme)
               .default_value(ABS_CENTRAL)
               .help("The scheme for generating candidate lambda set (1: center lambda=0.5, 2: center lambda=0.5, min "
                     "lambda=0, max lambda=1, 3: center lambda=epsilon/(1+epsilon), 4: center "
                     "lambda=epsilon/(1+epsilon), min lambda=0, max lambda=1); the rest of candidate lambda values are "
                     "generated using a doubling scheme"))
      .add(make_option("overwrite_label", data->overwrite_label)
               .default_value(1U)
               .help("the label used by type 3 corruptions (overwriting)"))
      .add(make_option("sim_bandit", data->sim_bandit)
               .help("simulate contextual bandit updates on warm start examples"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  if (use_cs && (options.was_supplied("corrupt_type_warm_start") || options.was_supplied("corrupt_prob_warm_start")))
  { THROW("label corruption on cost-sensitive examples not currently supported"); }

  data->app_seed = uniform_hash("vw", 2, 0);
  data->all = &all;
  data->_random_state = all.get_random_state();
  data->use_cs = use_cs;

  init_adf_data(*data.get(), num_actions);

  options.insert("cb_min_cost", std::to_string(data->loss0));
  options.insert("cb_max_cost", std::to_string(data->loss1));

  if (options.was_supplied("baseline"))
  {
    std::stringstream ss;
    ss << std::max(std::abs(data->loss0), std::abs(data->loss1)) / (data->loss1 - data->loss0);
    options.insert("lr_multiplier", ss.str());
  }

  learner<warm_cb, example>* l;

  multi_learner* base = as_multiline(setup_base(options, all));
  // Note: the current version of warm start CB can only support epsilon-greedy exploration
  // We need to wait for the epsilon value to be passed from the base
  // cb_explore learner, if there is one

  if (!options.was_supplied("epsilon"))
  {
    logger::errlog_warn("Warning: no epsilon (greedy parameter) specified; resetting to 0.05");
    data->epsilon = 0.05f;
  }

  if (use_cs)
  {
    l = &init_cost_sensitive_learner(data, base, predict_and_learn_adf<true>, predict_and_learn_adf<true>,
        all.example_parser, data->choices_lambda, all.get_setupfn_name(warm_cb_setup) + "-cs",
        prediction_type_t::multiclass, true);
    all.example_parser->lbl_parser.label_type = label_type_t::cs;
  }
  else
  {
    l = &init_multiclass_learner(data, base, predict_and_learn_adf<false>, predict_and_learn_adf<false>,
        all.example_parser, data->choices_lambda, all.get_setupfn_name(warm_cb_setup) + "-multi",
        prediction_type_t::multiclass, true);
    all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
  }

  l->set_finish(finish);

  return make_base(*l);
}
