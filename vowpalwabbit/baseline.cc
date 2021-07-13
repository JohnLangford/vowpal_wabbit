// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "baseline.h"

#include <cfloat>
#include <cerrno>

#include "reductions.h"
#include "vw.h"
#include "shared_data.h"

using namespace VW::LEARNER;
using namespace VW::config;

namespace
{
const float max_multiplier = 1000.f;
}  // namespace

namespace BASELINE
{
void set_baseline_enabled(example* ec)
{
  if (!baseline_enabled(ec)) { ec->indices.push_back(baseline_enabled_message_namespace); }
}

void reset_baseline_disabled(example* ec)
{
  auto it = std::find(ec->indices.begin(), ec->indices.end(), baseline_enabled_message_namespace);
  if (it != ec->indices.end()) { ec->indices.erase(it); }
}

bool baseline_enabled(example* ec)
{
  auto it = std::find(ec->indices.begin(), ec->indices.end(), baseline_enabled_message_namespace);
  return it != ec->indices.end();
}
}  // namespace BASELINE

struct baseline
{
  example* ec;
  vw* all;
  bool lr_scaling;  // whether to scale baseline learning rate based on max label
  float lr_multiplier;
  bool global_only;  // only use a global constant for the baseline
  bool global_initialized;
  bool check_enabled;  // only use baseline when the example contains enabled flag

  ~baseline()
  {
    if (ec) VW::dealloc_examples(ec, 1);
  }
};

void init_global(baseline& data)
{
  if (!data.global_only) return;
  // use a separate global constant
  data.ec->indices.push_back(constant_namespace);
  // different index from constant to avoid conflicts
  data.ec->feature_space[constant_namespace].push_back(
      1, ((constant - 17) * data.all->wpp) << data.all->weights.stride_shift());
  data.ec->reset_total_sum_feat_sq();
  data.ec->num_features++;
}

template <bool is_learn>
void predict_or_learn(baseline& data, single_learner& base, example& ec)
{
  // no baseline if check_enabled is true and example contains flag
  if (data.check_enabled && !BASELINE::baseline_enabled(&ec))
  {
    if (is_learn)
      base.learn(ec);
    else
      base.predict(ec);
    return;
  }

  // always do a full prediction, for safety in accurate predictive validation
  if (data.global_only)
  {
    if (!data.global_initialized)
    {
      init_global(data);
      data.global_initialized = true;
    }
    VW::copy_example_metadata(data.ec, &ec);
    base.predict(*data.ec);
    auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
    simple_red_features.initial = data.ec->pred.scalar;
    base.predict(ec);
  }
  else
    base.predict(ec);

  if (is_learn)
  {
    const float pred = ec.pred.scalar;  // save 'safe' prediction

    // now learn
    data.ec->l.simple = ec.l.simple;
    if (!data.global_only)
    {
      // move label & constant features data over to baseline example
      VW::copy_example_metadata(data.ec, &ec);
      VW::move_feature_namespace(data.ec, &ec, constant_namespace);
    }

    // regress baseline on label
    if (data.lr_scaling)
    {
      float multiplier = data.lr_multiplier;
      if (multiplier == 0)
      {
        multiplier = std::max(0.0001f, std::max(std::abs(data.all->sd->min_label), std::abs(data.all->sd->max_label)));
        if (multiplier > max_multiplier) multiplier = max_multiplier;
      }
      data.all->eta *= multiplier;
      base.learn(*data.ec);
      data.all->eta /= multiplier;
    }
    else
      base.learn(*data.ec);

    // regress residual
    auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
    simple_red_features.initial = data.ec->pred.scalar;
    base.learn(ec);

    if (!data.global_only)
    {
      // move feature data back to the original example
      VW::move_feature_namespace(&ec, data.ec, constant_namespace);
    }

    // return the safe prediction
    ec.pred.scalar = pred;
  }
}

float sensitivity(baseline& data, base_learner& base, example& ec)
{
  // no baseline if check_enabled is true and example contains flag
  if (data.check_enabled && !BASELINE::baseline_enabled(&ec)) return base.sensitivity(ec);

  if (!data.global_only) THROW("sensitivity for baseline without --global_only not implemented");

  // sensitivity of baseline term
  VW::copy_example_metadata(data.ec, &ec);
  data.ec->l.simple.label = ec.l.simple.label;
  data.ec->pred.scalar = ec.pred.scalar;
  const float baseline_sens = base.sensitivity(*data.ec);

  // sensitivity of residual
  as_singleline(&base)->predict(*data.ec);
  auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  simple_red_features.initial = data.ec->pred.scalar;
  const float sens = base.sensitivity(ec);
  return baseline_sens + sens;
}

base_learner* baseline_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<baseline>();
  bool baseline_option = false;
  std::string loss_function;

  option_group_definition new_options("Baseline options");
  new_options
      .add(make_option("baseline", baseline_option)
               .keep()
               .necessary()
               .help("Learn an additive baseline (from constant features) and a residual separately in regression."))
      .add(make_option("lr_multiplier", data->lr_multiplier).help("learning rate multiplier for baseline model"))
      .add(make_option("global_only", data->global_only)
               .keep()
               .help("use separate example with only global constant for baseline predictions"))
      .add(make_option("check_enabled", data->check_enabled)
               .keep()
               .help("only use baseline when the example contains enabled flag"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // initialize baseline example
  data->ec = VW::alloc_examples(1);
  data->ec->interactions = &all.interactions;

  data->all = &all;

  auto loss_function_type = all.loss->getType();
  if (loss_function_type != "logistic") data->lr_scaling = true;

  auto base = as_singleline(setup_base(options, all));

  learner<baseline, example>& l =
      init_learner(data, base, predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(baseline_setup));

  l.set_sensitivity(sensitivity);

  return make_base(l);
}
