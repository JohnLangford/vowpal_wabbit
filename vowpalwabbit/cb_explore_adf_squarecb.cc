// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_squarecb.h"
#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"
#include "explore.h"
#include "action_score.h"
#include "cb.h"
#include "vw_versions.h"
#include "version.h"
#include "label_parser.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

/*
This file implements the SquareCB algorithm/reduction (Foster and Rakhlin (2020), https://arxiv.org/abs/2002.04926),
with the VW learner as the base algorithm.
*/

// All exploration algorithms return a vector of id, probability tuples, sorted in order of scores. The probabilities
// are the probability with which each action should be replaced to the top of the list.

#define B_SEARCH_MAX_ITER 20

using namespace VW::LEARNER;

namespace VW
{
namespace cb_explore_adf
{
namespace squarecb
{
struct cb_explore_adf_squarecb
{
private:
  // size_t _counter;
  size_t _counter;
  float _gamma_scale;     // Scale factor for SquareCB reediness parameter $\gamma$.
  float _gamma_exponent;  // Exponent on $t$ for SquareCB reediness parameter $\gamma$.

  // Parameters and data structures for RegCB action set computation
  bool _elim;
  float _c0;
  float _min_cb_cost;
  float _max_cb_cost;

  std::vector<float> _min_costs;
  std::vector<float> _max_costs;

  VW::version_struct _model_file_version;

  // for backing up cb example data when computing sensitivities
  std::vector<ACTION_SCORE::action_scores> _ex_as;
  std::vector<v_array<CB::cb_class>> _ex_costs;

public:
  cb_explore_adf_squarecb(float gamma_scale, float gamma_exponent, bool elim, float c0, float min_cb_cost,
      float max_cb_cost, VW::version_struct model_file_version);
  ~cb_explore_adf_squarecb() = default;

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }
  void save_load(io_buf& io, bool read, bool text);

private:
  template <bool is_learn>
  void predict_or_learn_impl(multi_learner& base, multi_ex& examples);

  void get_cost_ranges(float delta, multi_learner& base, multi_ex& examples, bool min_only);
  float binary_search(float fhat, float delta, float sens, float tol = 1e-6);
};

cb_explore_adf_squarecb::cb_explore_adf_squarecb(float gamma_scale, float gamma_exponent, bool elim, float c0,
    float min_cb_cost, float max_cb_cost, VW::version_struct model_file_version)
    : _counter(0)
    , _gamma_scale(gamma_scale)
    , _gamma_exponent(gamma_exponent)
    , _elim(elim)
    , _c0(c0)
    , _min_cb_cost(min_cb_cost)
    , _max_cb_cost(max_cb_cost)
    , _model_file_version(model_file_version)
{
}

// TODO: same as cs_active.cc and cb_explore_adf_regcb.cc, move to shared place
float cb_explore_adf_squarecb::binary_search(float fhat, float delta, float sens, float tol)
{
  /*
     Binary search to find the largest weight w such that w*(fhat^2 - (fhat - w*sens)^2) \leq delta.
     Implements binary search procedure described at the end of Section 7.1 in https://arxiv.org/pdf/1703.01014.pdf.
  */

  // We are always guaranteed that the solution to the problem above lies in (0, maxw), as long as fhat \geq 0.
  const float maxw = (std::min)(fhat / sens, FLT_MAX);

  // If the objective value for maxw satisfies the delta constraint, we can just take this and skip the binary search.
  if (maxw * fhat * fhat <= delta) return maxw;

  // Upper and lower bounds on w for binary search.
  float l = 0;
  float u = maxw;
  // Binary search variable.
  float w;
  // Value for w.
  float v;

  // Standard binary search given the objective described above.
  for (int iter = 0; iter < B_SEARCH_MAX_ITER; iter++)
  {
    w = (u + l) / 2.f;
    v = w * (fhat * fhat - (fhat - sens * w) * (fhat - sens * w)) - delta;
    if (v > 0)
      u = w;
    else
      l = w;
    if (std::fabs(v) <= tol || u - l <= tol) break;
  }

  return l;
}

// TODO: Same as cb_explore_adf_regcb.cc
void cb_explore_adf_squarecb::get_cost_ranges(float delta, multi_learner& base, multi_ex& examples, bool min_only)
{
  const size_t num_actions = examples[0]->pred.a_s.size();
  _min_costs.resize(num_actions);
  _max_costs.resize(num_actions);

  _ex_as.clear();
  _ex_costs.clear();

  // backup cb example data
  for (const auto& ex : examples)
  {
    _ex_as.push_back(ex->pred.a_s);
    _ex_costs.push_back(ex->l.cb.costs);
  }

  // set regressor predictions
  for (const auto& as : _ex_as[0]) { examples[as.action]->pred.scalar = as.score; }

  const float cmin = _min_cb_cost;
  const float cmax = _max_cb_cost;

  for (size_t a = 0; a < num_actions; ++a)
  {
    example* ec = examples[a];
    ec->l.simple.label = cmin - 1;
    float sens = base.sensitivity(*ec);
    float w = 0;  // importance weight

    if (ec->pred.scalar < cmin || std::isnan(sens) || std::isinf(sens))
      _min_costs[a] = cmin;
    else
    {
      w = binary_search(ec->pred.scalar - cmin + 1, delta, sens);
      _min_costs[a] = (std::max)(ec->pred.scalar - sens * w, cmin);
      if (_min_costs[a] > cmax) _min_costs[a] = cmax;
    }

    if (!min_only)
    {
      ec->l.simple.label = cmax + 1;
      sens = base.sensitivity(*ec);
      if (ec->pred.scalar > cmax || std::isnan(sens) || std::isinf(sens)) { _max_costs[a] = cmax; }
      else
      {
        w = binary_search(cmax + 1 - ec->pred.scalar, delta, sens);
        _max_costs[a] = (std::min)(ec->pred.scalar + sens * w, cmax);
        if (_max_costs[a] < cmin) _max_costs[a] = cmin;
      }
    }
  }

  // reset cb example data
  for (size_t i = 0; i < examples.size(); ++i)
  {
    examples[i]->pred.a_s = _ex_as[i];
    examples[i]->l.cb.costs = _ex_costs[i];
  }
}

template <bool is_learn>
void cb_explore_adf_squarecb::predict_or_learn_impl(multi_learner& base, multi_ex& examples)
{
  if (is_learn)
  {
    for (size_t i = 0; i < examples.size() - 1; ++i)
    {
      CB::label& ld = examples[i]->l.cb;
      if (ld.costs.size() == 1) ld.costs[0].probability = 1.f;  // no importance weighting
    }

    multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset);
    ++_counter;
  }
  else
    multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);

  v_array<ACTION_SCORE::action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = static_cast<uint32_t>(preds.size());

  // The actual parameter $\gamma$ used in the SquareCB.
  const float gamma = _gamma_scale * static_cast<float>(std::pow(_counter, _gamma_exponent));

  // RegCB action set parameters
  const float max_range = _max_cb_cost - _min_cb_cost;
  // threshold on empirical loss difference
  const float delta =
      _c0 * std::log(static_cast<float>(num_actions * _counter)) * static_cast<float>(std::pow(max_range, 2));

  // SquareCB Exploration
  if (!is_learn)
  {
    if (!_elim)  // Vanilla variant (perform SquareCB exploration over all actions)
    {
      size_t a_min = 0;
      float min_cost = preds[0].score;
      // Compute highest-scoring action
      for (size_t a = 0; a < num_actions; ++a)
      {
        if (preds[a].score < min_cost)
        {
          a_min = a;
          min_cost = preds[a].score;
        }
      }
      // Compute probabilities using SquareCB rule.
      float total_weight = 0;
      float pa = 0;
      for (size_t a = 0; a < num_actions; ++a)
      {
        if (a == a_min) continue;
        pa = 1.f / (num_actions + gamma * (preds[a].score - min_cost));
        preds[a].score = pa;
        total_weight += pa;
      }
      preds[a_min].score = 1.f - total_weight;
    }
    else  // elimination variant
    {
      get_cost_ranges(delta, base, examples, /*min_only=*/false);

      float min_max_cost = FLT_MAX;
      for (size_t a = 0; a < num_actions; ++a)
        if (_max_costs[a] < min_max_cost) min_max_cost = _max_costs[a];

      size_t a_min = 0;
      size_t num_surviving_actions = 0;
      float min_cost = FLT_MAX;
      // Compute plausible / surviving actions.
      for (size_t a = 0; a < num_actions; ++a)
      {
        if (preds[a].score < min_cost && _min_costs[preds[a].action] <= min_max_cost)
        {
          a_min = a;
          min_cost = preds[a].score;
          num_surviving_actions += 1;
        }
      }
      float total_weight = 0;
      float pa = 0;
      // // Compute probabilities for surviving actions using SquareCB rule.
      for (size_t a = 0; a < num_actions; ++a)
      {
        if (_min_costs[preds[a].action] > min_max_cost) { preds[a].score = 0; }
        else
        {
          if (a == a_min) continue;
          pa = 1.f / (num_surviving_actions + gamma * (preds[a].score - min_cost));
          preds[a].score = pa;
          total_weight += pa;
        }
      }
      preds[a_min].score = 1.f - total_weight;
    }
  }
}

void cb_explore_adf_squarecb::save_load(io_buf& io, bool read, bool text)
{
  if (io.num_files() == 0) { return; }
  if (!read || _model_file_version >= VERSION_FILE_WITH_SQUARE_CB_SAVE_RESUME)
  {
    std::stringstream msg;
    if (!read) { msg << "cb squarecb adf storing example counter:  = " << _counter << "\n"; }
    bin_text_read_write_fixed_validated(io, reinterpret_cast<char*>(&_counter), sizeof(_counter), "", read, msg, text);
  }
}

base_learner* setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  bool squarecb = false;
  std::string type_string = "mtr";

  // Basic SquareCB parameters
  float gamma_scale = 1.;
  float gamma_exponent = 0.;

  // Perform SquareCB exploration over RegCB-style disagreement sets
  bool elim = false;
  float c0 = 0.;
  float min_cb_cost = 0.;
  float max_cb_cost = 0.;

  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (SquareCB)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("squarecb", squarecb).keep().necessary().help("SquareCB exploration"))
      .add(make_option("gamma_scale", gamma_scale)
               .keep()
               .default_value(10.f)
               .help("Sets SquareCB greediness parameter to gamma=[gamma_scale]*[num examples]^1/2"))
      .add(make_option("gamma_exponent", gamma_exponent)
               .keep()
               .default_value(.5f)
               .help("Exponent on [num examples] in SquareCB greediness parameter gamma."))
      .add(make_option("elim", elim)
               .keep()
               .help("Only perform SquareCB exploration over plausible actions (computed via RegCB strategy)"))
      .add(make_option("mellowness", c0)
               .keep()
               .default_value(0.001f)
               .help("Mellowness parameter c_0 for computing plausible action set. Only used with --elim"))
      .add(make_option("cb_min_cost", min_cb_cost)
               .keep()
               .default_value(0.f)
               .help("Lower bound on cost. Only used with --elim"))
      .add(make_option("cb_max_cost", max_cb_cost)
               .keep()
               .default_value(1.f)
               .help("Upper bound on cost. Only used with --elim"))
      .add(make_option("cb_type", type_string)
               .keep()
               .help("contextual bandit method to use in {ips,dr,mtr}. Default: mtr"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }
  if (type_string != "mtr")
  {
    *(all.trace_message) << "warning: bad cb_type, SquareCB only supports mtr; resetting to mtr." << std::endl;
    options.replace("cb_type", "mtr");
  }

  // Set explore_type
  size_t problem_multiplier = 1;

  multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_squarecb>;
  auto data = VW::make_unique<explore_type>(
      with_metrics, gamma_scale, gamma_exponent, elim, c0, min_cb_cost, max_cb_cost, all.model_file_ver);
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-squarecb")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(explore_type::finish_multiline_example)
                .set_print_example(explore_type::print_multiline_example)
                .set_persist_metrics(explore_type::persist_metrics)
                .set_save_load(explore_type::save_load)
                .build();
  return make_base(*l);
}

}  // namespace squarecb
}  // namespace cb_explore_adf
}  // namespace VW
