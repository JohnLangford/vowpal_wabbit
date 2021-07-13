// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <unordered_map>
#include "reductions.h"
#include "correctedMath.h"

#include "io/logger.h"

using namespace VW::config;
namespace logger = VW::io::logger;

namespace MARGINAL
{
struct expert
{
  float regret;
  float abs_regret;
  float weight;
};

typedef std::pair<double, double> marginal;
typedef std::pair<expert, expert> expert_pair;

struct data
{
  float initial_numerator;
  float initial_denominator;
  float decay;
  bool update_before_learn;
  bool unweighted_marginals;
  bool id_features[256];
  features temp[256];  // temporary storage when reducing.
  std::unordered_map<uint64_t, marginal> marginals;

  // bookkeeping variables for experts
  bool compete;
  float feature_pred;        // the prediction computed from using all the features
  float average_pred;        // the prediction of the expert
  float net_weight;          // normalizer for expert weights
  float net_feature_weight;  // the net weight on the feature-based expert
  float alg_loss;            // temporary storage for the loss of the current marginal-based predictor
  std::unordered_map<uint64_t, expert_pair>
      expert_state;  // pair of weights on marginal and feature based predictors, one per marginal feature

  vw* all;
};

float get_adanormalhedge_weights(float R, float C)
{
  float Rpos = R > 0 ? R : 0.f;
  if (C == 0. || Rpos == 0.) return 0;
  return 2 * Rpos * correctedExp(Rpos * Rpos / (3 * C)) / (3 * C);
}

template <bool is_learn>
void make_marginal(data& sm, example& ec)
{
  uint64_t mask = sm.all->weights.mask();
  float label = ec.l.simple.label;
  vw& all = *sm.all;
  sm.alg_loss = 0.;
  sm.net_weight = 0.;
  sm.net_feature_weight = 0.;
  sm.average_pred = 0.;

  for (example::iterator i = ec.begin(); i != ec.end(); ++i)
  {
    namespace_index n = i.index();
    if (sm.id_features[n])
    {
      std::swap(sm.temp[n], *i);
      features& f = *i;
      f.clear();
      for (features::iterator j = sm.temp[n].begin(); j != sm.temp[n].end(); ++j)
      {
        float first_value = j.value();
        uint64_t first_index = j.index() & mask;
        if (++j == sm.temp[n].end())
        {
          logger::log_warn("warning: id feature namespace has {} features. Should be a multiple of 2",
                           sm.temp[n].size());
          break;
        }
        float second_value = j.value();
        uint64_t second_index = j.index() & mask;
        if (first_value != 1. || second_value != 1.)
        {
          logger::log_warn("warning: bad id features, must have value 1.");
          continue;
        }
        uint64_t key = second_index + ec.ft_offset;
        if (sm.marginals.find(key) == sm.marginals.end())  // need to initialize things.
        {
          sm.marginals.insert(std::make_pair(key, std::make_pair(sm.initial_numerator, sm.initial_denominator)));
          if (sm.compete)
          {
            expert e = {0, 0, 1.};
            sm.expert_state.insert(std::make_pair(key, std::make_pair(e, e)));
          }
        }
        float marginal_pred = static_cast<float>(sm.marginals[key].first / sm.marginals[key].second);
        f.push_back(marginal_pred, first_index);
        if (!sm.temp[n].space_names.empty()) f.space_names.push_back(sm.temp[n].space_names[2 * (f.size() - 1)]);

        if (sm.compete)  // compute the prediction from the marginals using the weights
        {
          float weight = sm.expert_state[key].first.weight;
          sm.average_pred += weight * marginal_pred;
          sm.net_weight += weight;
          sm.net_feature_weight += sm.expert_state[key].second.weight;
          if (is_learn) sm.alg_loss += weight * all.loss->getLoss(all.sd, marginal_pred, label);
        }
      }
    }
  }
}

void undo_marginal(data& sm, example& ec)
{
  for (example::iterator i = ec.begin(); i != ec.end(); ++i)
  {
    namespace_index n = i.index();
    if (sm.id_features[n]) std::swap(sm.temp[n], *i);
  }
}

template <bool is_learn>
void compute_expert_loss(data& sm, example& ec)
{
  vw& all = *sm.all;
  // add in the feature-based expert and normalize,
  float label = ec.l.simple.label;

  if (sm.net_weight + sm.net_feature_weight > 0.)
    sm.average_pred += sm.net_feature_weight * sm.feature_pred;
  else
  {
    sm.net_feature_weight = 1.;
    sm.average_pred = sm.feature_pred;
  }
  float inv_weight = 1.0f / (sm.net_weight + sm.net_feature_weight);
  sm.average_pred *= inv_weight;
  ec.pred.scalar = sm.average_pred;
  ec.partial_prediction = sm.average_pred;

  if (is_learn)
  {
    sm.alg_loss += sm.net_feature_weight * all.loss->getLoss(all.sd, sm.feature_pred, label);
    sm.alg_loss *= inv_weight;
  }
}

void update_marginal(data& sm, example& ec)
{
  vw& all = *sm.all;
  uint64_t mask = sm.all->weights.mask();
  float label = ec.l.simple.label;
  float weight = ec.weight;
  if (sm.unweighted_marginals) weight = 1.;

  for (example::iterator i = ec.begin(); i != ec.end(); ++i)
  {
    namespace_index n = i.index();
    if (sm.id_features[n])
      for (features::iterator j = sm.temp[n].begin(); j != sm.temp[n].end(); ++j)
      {
        if (++j == sm.temp[n].end()) break;

        uint64_t second_index = j.index() & mask;
        uint64_t key = second_index + ec.ft_offset;
        marginal& m = sm.marginals[key];

        if (sm.compete)  // now update weights, before updating marginals
        {
          expert_pair& e = sm.expert_state[key];
          float regret1 = sm.alg_loss - all.loss->getLoss(all.sd, static_cast<float>(m.first / m.second), label);
          float regret2 = sm.alg_loss - all.loss->getLoss(all.sd, sm.feature_pred, label);

          e.first.regret += regret1 * weight;
          e.first.abs_regret += regret1 * regret1 * weight;  // fabs(regret1);
          e.first.weight = get_adanormalhedge_weights(e.first.regret, e.first.abs_regret);
          e.second.regret += regret2 * weight;
          e.second.abs_regret += regret2 * regret2 * weight;  // fabs(regret2);
          e.second.weight = get_adanormalhedge_weights(e.second.regret, e.second.abs_regret);
        }

        m.first = m.first * (1. - sm.decay) + ec.l.simple.label * weight;
        m.second = m.second * (1. - sm.decay) + weight;
      }
  }
}

template <bool is_learn>
void predict_or_learn(data& sm, VW::LEARNER::single_learner& base, example& ec)
{
  make_marginal<is_learn>(sm, ec);
  if (is_learn)
    if (sm.update_before_learn)
    {
      base.predict(ec);
      float pred = ec.pred.scalar;
      if (sm.compete)
      {
        sm.feature_pred = pred;
        compute_expert_loss<is_learn>(sm, ec);
      }
      undo_marginal(sm, ec);
      update_marginal(sm, ec);  // update features before learning.
      make_marginal<is_learn>(sm, ec);
      base.learn(ec);
      ec.pred.scalar = pred;
    }
    else
    {
      base.learn(ec);
      if (sm.compete)
      {
        sm.feature_pred = ec.pred.scalar;
        compute_expert_loss<is_learn>(sm, ec);
      }
      update_marginal(sm, ec);
    }
  else
  {
    base.predict(ec);
    float pred = ec.pred.scalar;
    if (sm.compete)
    {
      sm.feature_pred = pred;
      compute_expert_loss<is_learn>(sm, ec);
    }
  }

  // undo marginalization
  undo_marginal(sm, ec);
}

void save_load(data& sm, io_buf& io, bool read, bool text)
{
  uint64_t stride_shift = sm.all->weights.stride_shift();

  if (io.num_files() == 0) return;
  std::stringstream msg;
  uint64_t total_size;
  if (!read)
  {
    total_size = static_cast<uint64_t>(sm.marginals.size());
    msg << "marginals size = " << total_size << "\n";
  }
  bin_text_read_write_fixed_validated(
      io, reinterpret_cast<char*>(&total_size), sizeof(total_size), "", read, msg, text);

  auto iter = sm.marginals.begin();
  for (size_t i = 0; i < total_size; ++i)
  {
    uint64_t index;
    if (!read)
    {
      index = iter->first >> stride_shift;
      msg << index << ":";
    }
    bin_text_read_write_fixed(io, reinterpret_cast<char*>(&index), sizeof(index), "", read, msg, text);
    double numerator;
    if (!read)
    {
      numerator = iter->second.first;
      msg << numerator << ":";
    }
    bin_text_read_write_fixed(io, reinterpret_cast<char*>(&numerator), sizeof(numerator), "", read, msg, text);
    double denominator;
    if (!read)
    {
      denominator = iter->second.second;
      msg << denominator << "\n";
    }
    bin_text_read_write_fixed(io, reinterpret_cast<char*>(&denominator), sizeof(denominator), "", read, msg, text);
    if (read)
      sm.marginals.insert(std::make_pair(index << stride_shift, std::make_pair(numerator, denominator)));
    else
      ++iter;
  }

  if (sm.compete)
  {
    if (!read)
    {
      total_size = static_cast<uint64_t>(sm.expert_state.size());
      msg << "expert_state size = " << total_size << "\n";
    }
    bin_text_read_write_fixed_validated(
        io, reinterpret_cast<char*>(&total_size), sizeof(total_size), "", read, msg, text);

    auto exp_iter = sm.expert_state.begin();
    for (size_t i = 0; i < total_size; ++i)
    {
      uint64_t index;
      if (!read)
      {
        index = exp_iter->first >> stride_shift;
        msg << index << ":";
      }
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&index), sizeof(index), "", read, msg, text);
      float r1 = 0;
      float c1 = 0;
      float w1 = 0;
      float r2 = 0;
      float c2 = 0;
      float w2 = 0;
      if (!read)
      {
        r1 = exp_iter->second.first.regret;
        c1 = exp_iter->second.first.abs_regret;
        w1 = exp_iter->second.first.weight;
        r2 = exp_iter->second.second.regret;
        c2 = exp_iter->second.second.abs_regret;
        w2 = exp_iter->second.second.weight;
        msg << r1 << ":";
      }
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&r1), sizeof(r1), "", read, msg, text);
      if (!read) msg << c1 << ":";
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&c1), sizeof(c1), "", read, msg, text);
      if (!read) msg << w1 << ":";
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&w1), sizeof(w1), "", read, msg, text);
      if (!read) msg << r2 << ":";
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&r2), sizeof(r2), "", read, msg, text);
      if (!read) msg << c2 << ":";
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&c2), sizeof(c2), "", read, msg, text);
      if (!read) msg << w2 << ":";
      bin_text_read_write_fixed(io, reinterpret_cast<char*>(&w2), sizeof(w2), "", read, msg, text);

      if (read)
      {
        expert e1 = {r1, c1, w1};
        expert e2 = {r2, c2, w2};
        sm.expert_state.insert(std::make_pair(index << stride_shift, std::make_pair(e1, e2)));
      }
      else
        ++exp_iter;
    }
  }
}
}  // namespace MARGINAL

using namespace MARGINAL;

VW::LEARNER::base_learner* marginal_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  free_ptr<MARGINAL::data> d = scoped_calloc_or_throw<MARGINAL::data>();
  std::string marginal;

  option_group_definition marginal_options("Marginal options");
  marginal_options.add(
      make_option("marginal", marginal).keep().necessary().help("substitute marginal label estimates for ids"));
  marginal_options.add(
      make_option("initial_denominator", d->initial_denominator).default_value(1.f).help("initial denominator"));
  marginal_options.add(
      make_option("initial_numerator", d->initial_numerator).default_value(0.5f).help("initial numerator"));
  marginal_options.add(make_option("compete", d->compete).help("enable competition with marginal features"));
  marginal_options.add(
      make_option("update_before_learn", d->update_before_learn).help("update marginal values before learning"));
  marginal_options.add(make_option("unweighted_marginals", d->unweighted_marginals)
                           .help("ignore importance weights when computing marginals"));
  marginal_options.add(
      make_option("decay", d->decay).default_value(0.f).help("decay multiplier per event (1e-3 for example)"));

  if (!options.add_parse_and_check_necessary(marginal_options)) { return nullptr; }

  d->all = &all;

  for (size_t u = 0; u < 256; u++)
    if (marginal.find(static_cast<char>(u)) != std::string::npos) d->id_features[u] = true;

  VW::LEARNER::learner<MARGINAL::data, example>& ret = init_learner(d, as_singleline(setup_base(options, all)),
      predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(marginal_setup), true);
  ret.set_save_load(save_load);

  return make_base(ret);
}
