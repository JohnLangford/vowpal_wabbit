// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cmath>

#include "vw.h"
#include "reductions.h"
#include "gd.h"
#include "cb_algs.h"
#include "io_buf.h"
#include "cb.h"
#include "shared_data.h"

#include "io/logger.h"

using namespace VW::LEARNER;
using namespace CB_ALGS;
using namespace VW::config;

namespace logger = VW::io::logger;

namespace MWT
{
struct policy_data
{
  double cost = 0.0;
  uint32_t action = 0;
  bool seen = false;
};

struct mwt
{
  bool namespaces[256];        // the set of namespaces to evaluate.
  v_array<policy_data> evals;  // accrued losses of features.
  std::pair<bool, CB::cb_class> optional_observation;
  v_array<uint64_t> policies;
  double total;
  uint32_t num_classes;
  bool learn;

  v_array<namespace_index> indices;  // excluded namespaces
  features feature_space[256];
  vw* all;
};

void value_policy(mwt& c, float val, uint64_t index)  // estimate the value of a single feature.
{
  if (val < 0 || std::floor(val) != val) logger::log_error("error {} is not a valid action", val);

  uint32_t value = static_cast<uint32_t>(val);
  uint64_t new_index = (index & c.all->weights.mask()) >> c.all->weights.stride_shift();

  if (!c.evals[new_index].seen)
  {
    c.evals[new_index].seen = true;
    c.policies.push_back(new_index);
  }

  c.evals[new_index].action = value;
}

template <bool learn, bool exclude, bool is_learn>
void predict_or_learn(mwt& c, single_learner& base, example& ec)
{
  c.optional_observation = get_observed_cost_cb(ec.l.cb);

  if (c.optional_observation.first)
  {
    c.total++;
    // For each nonzero feature in observed namespaces, check it's value.
    for (unsigned char ns : ec.indices)
      if (c.namespaces[ns]) GD::foreach_feature<mwt, value_policy>(c.all, ec.feature_space[ns], c);
    for (uint64_t policy : c.policies)
    {
      c.evals[policy].cost += get_cost_estimate(c.optional_observation.second, c.evals[policy].action);
      c.evals[policy].action = 0;
    }
  }

  VW_WARNING_STATE_PUSH
  VW_WARNING_DISABLE_CPP_17_LANG_EXT
  if VW_STD17_CONSTEXPR (exclude || learn)
  {
    c.indices.clear();
    uint32_t stride_shift = c.all->weights.stride_shift();
    uint64_t weight_mask = c.all->weights.mask();
    for (unsigned char ns : ec.indices)
      if (c.namespaces[ns])
      {
        c.indices.push_back(ns);
        if (learn)
        {
          c.feature_space[ns].clear();
          for (features::iterator& f : ec.feature_space[ns])
          {
            uint64_t new_index =
                ((f.index() & weight_mask) >> stride_shift) * c.num_classes + static_cast<uint64_t>(f.value());
            c.feature_space[ns].push_back(1, new_index << stride_shift);
          }
        }
        std::swap(c.feature_space[ns], ec.feature_space[ns]);
      }
  }
  VW_WARNING_STATE_POP

  // modify the predictions to use a vector with a score for each evaluated feature.
  v_array<float> preds = ec.pred.scalars;

  if (learn)
  {
    if (is_learn)
      base.learn(ec);
    else
      base.predict(ec);
  }

  VW_WARNING_STATE_PUSH
  VW_WARNING_DISABLE_CPP_17_LANG_EXT
  if VW_STD17_CONSTEXPR (exclude || learn)
    while (!c.indices.empty())
    {
      unsigned char ns = c.indices.back();
      c.indices.pop_back();
      std::swap(c.feature_space[ns], ec.feature_space[ns]);
    }
  VW_WARNING_STATE_POP

  // modify the predictions to use a vector with a score for each evaluated feature.
  preds.clear();
  if (learn) preds.push_back(static_cast<float>(ec.pred.multiclass));
  for (uint64_t index : c.policies)
    preds.push_back(static_cast<float>(c.evals[index].cost) / static_cast<float>(c.total));

  ec.pred.scalars = preds;
}

void print_scalars(VW::io::writer* f, v_array<float>& scalars, v_array<char>& tag)
{
  if (f != nullptr)
  {
    std::stringstream ss;

    for (size_t i = 0; i < scalars.size(); i++)
    {
      if (i > 0) ss << ' ';
      ss << scalars[i];
    }
    for (size_t i = 0; i < tag.size(); i++)
    {
      if (i == 0) ss << ' ';
      ss << tag[i];
    }
    ss << '\n';
    ssize_t len = ss.str().size();
    ssize_t t = f->write(ss.str().c_str(), static_cast<unsigned int>(len));
    if (t != len) logger::errlog_error("write error: {}", VW::strerror_to_string(errno));
  }
}

void finish_example(vw& all, mwt& c, example& ec)
{
  float loss = 0.;
  if (c.learn)
    if (c.optional_observation.first)
      loss = get_cost_estimate(c.optional_observation.second, static_cast<uint32_t>(ec.pred.scalars[0]));
  all.sd->update(ec.test_only, c.optional_observation.first, loss, 1.f, ec.get_num_features());

  for (auto& sink : all.final_prediction_sink) print_scalars(sink.get(), ec.pred.scalars, ec.tag);

  if (c.learn)
  {
    v_array<float> temp = ec.pred.scalars;
    ec.pred.multiclass = static_cast<uint32_t>(temp[0]);
    CB::print_update(all, c.optional_observation.first, ec, nullptr, false, nullptr);
    ec.pred.scalars = temp;
  }
  VW::finish_example(all, ec);
}

void save_load(mwt& c, io_buf& model_file, bool read, bool text)
{
  if (model_file.num_files() == 0) return;

  std::stringstream msg;

  // total
  msg << "total: " << c.total;
  bin_text_read_write_fixed_validated(
      model_file, reinterpret_cast<char*>(&c.total), sizeof(c.total), "", read, msg, text);

  // policies
  size_t policies_size = c.policies.size();
  bin_text_read_write_fixed_validated(
      model_file, reinterpret_cast<char*>(&policies_size), sizeof(policies_size), "", read, msg, text);

  if (read) { c.policies.resize_but_with_stl_behavior(policies_size); }
  else
  {
    msg << "policies: ";
    for (feature_index& policy : c.policies) msg << policy << " ";
  }

  bin_text_read_write_fixed_validated(model_file, reinterpret_cast<char*>(c.policies.begin()),
      policies_size * sizeof(feature_index), "", read, msg, text);

  // c.evals is already initialized nicely to the same size as the regressor.
  for (feature_index& policy : c.policies)
  {
    policy_data& pd = c.evals[policy];
    if (read) msg << "evals: " << policy << ":" << pd.action << ":" << pd.cost << " ";
    bin_text_read_write_fixed_validated(
        model_file, reinterpret_cast<char*>(&c.evals[policy]), sizeof(policy_data), "", read, msg, text);
  }
}
}  // namespace MWT
using namespace MWT;

base_learner* mwt_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto c = scoped_calloc_or_throw<mwt>();
  std::string s;
  bool exclude_eval = false;
  option_group_definition new_options("Multiworld Testing Options");
  new_options.add(make_option("multiworld_test", s).keep().necessary().help("Evaluate features as a policies"))
      .add(make_option("learn", c->num_classes).help("Do Contextual Bandit learning on <n> classes."))
      .add(make_option("exclude_eval", exclude_eval).help("Discard mwt policy features before learning"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  for (char i : s) c->namespaces[static_cast<unsigned char>(i)] = true;
  c->all = &all;

  c->evals.resize_but_with_stl_behavior(all.length());
  all.example_parser->lbl_parser = CB::cb_label;

  if (c->num_classes > 0)
  {
    c->learn = true;

    if (!options.was_supplied("cb"))
    {
      std::stringstream ss;
      ss << c->num_classes;
      options.insert("cb", ss.str());
    }
  }

  // default to legacy cb implementation
  options.insert("cb_force_legacy", "");

  learner<mwt, example>* l;
  if (c->learn)
    if (exclude_eval)
      l = &init_learner(c, as_singleline(setup_base(options, all)), predict_or_learn<true, true, true>,
          predict_or_learn<true, true, false>, 1, prediction_type_t::scalars,
          all.get_setupfn_name(mwt_setup) + "-no_eval", true);
    else
      l = &init_learner(c, as_singleline(setup_base(options, all)), predict_or_learn<true, false, true>,
          predict_or_learn<true, false, false>, 1, prediction_type_t::scalars,
          all.get_setupfn_name(mwt_setup) + "-eval", true);
  else
    l = &init_learner(c, as_singleline(setup_base(options, all)), predict_or_learn<false, false, true>,
        predict_or_learn<false, false, false>, 1, prediction_type_t::scalars, all.get_setupfn_name(mwt_setup), true);

  l->set_save_load(save_load);
  l->set_finish_example(finish_example);
  return make_base(*l);
}
