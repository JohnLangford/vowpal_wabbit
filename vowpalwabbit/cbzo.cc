// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <random>
#include <cfloat>

#include "gd.h"
#include "io_buf.h"
#include "parse_regressor.h"
#include "cbzo.h"
#include "vw.h"
#include "vw_math.h"
#include "prob_dist_cont.h"
#include "shared_data.h"

using namespace VW::LEARNER;
using namespace VW::config;
using VW::continuous_actions::probability_density_function;

namespace VW
{
namespace cbzo
{
constexpr uint8_t constant_policy = 0;
constexpr uint8_t linear_policy = 1;

struct cbzo
{
  float radius;
  vw* all;
  bool min_prediction_supplied, max_prediction_supplied;
};

struct linear_update_data
{
  float mult;
  float part_grad;
  vw* all;
};

// uint64_t index variant of VW::get_weight
inline float get_weight(vw& all, uint64_t index, uint32_t offset)
{
  return (&all.weights[(index) << all.weights.stride_shift()])[offset];
}

// uint64_t index variant of VW::set_weight
inline void set_weight(vw& all, uint64_t index, uint32_t offset, float value)
{
  (&all.weights[(index) << all.weights.stride_shift()])[offset] = value;
}

float l1_grad(vw& all, uint64_t fi)
{
  if (all.no_bias && fi == constant) return 0.0f;

  float fw = get_weight(all, fi, 0);
  return fw >= 0.0f ? all.l1_lambda : -all.l1_lambda;
}

float l2_grad(vw& all, uint64_t fi)
{
  if (all.no_bias && fi == constant) return 0.0f;

  float fw = get_weight(all, fi, 0);
  return all.l2_lambda * fw;
}

inline void accumulate_dotprod(float& dotprod, float x, float& fw) { dotprod += x * fw; }

inline float constant_inference(vw& all)
{
  float wt = get_weight(all, constant, 0);
  return wt;
}

float linear_inference(vw& all, example& ec)
{
  float dotprod = 0;
  GD::foreach_feature<float, accumulate_dotprod>(all, ec, dotprod);
  return dotprod;
}

template <uint8_t policy>
float inference(vw& all, example& ec)
{
  if (policy == constant_policy)
    return constant_inference(all);

  else if (policy == linear_policy)
    return linear_inference(all, ec);

  else
    THROW("Unknown policy encountered: " << policy);
}

template <bool feature_mask_off>
void constant_update(cbzo& data, example& ec)
{
  float fw = get_weight(*data.all, constant, 0);
  if (feature_mask_off || fw != 0.0f)
  {
    float action_centroid = inference<constant_policy>(*data.all, ec);
    float grad = ec.l.cb_cont.costs[0].cost / (ec.l.cb_cont.costs[0].action - action_centroid);
    float update = -data.all->eta * (grad + l1_grad(*data.all, constant) + l2_grad(*data.all, constant));

    set_weight(*data.all, constant, 0, fw + update);
  }
}

template <bool feature_mask_off>
void linear_per_feature_update(linear_update_data& upd_data, float x, uint64_t fi)
{
  float fw = get_weight(*upd_data.all, fi, 0);

  if (feature_mask_off || fw != 0.0f)
  {
    float update = upd_data.mult * (upd_data.part_grad * x + (l1_grad(*upd_data.all, fi) + l2_grad(*upd_data.all, fi)));
    set_weight(*upd_data.all, fi, 0, fw + update);
  }
}

template <bool feature_mask_off>
void linear_update(cbzo& data, example& ec)
{
  float mult = -data.all->eta;

  float action_centroid = inference<linear_policy>(*data.all, ec);
  float part_grad = ec.l.cb_cont.costs[0].cost / (ec.l.cb_cont.costs[0].action - action_centroid);

  linear_update_data upd_data;
  upd_data.mult = mult;
  upd_data.part_grad = part_grad;
  upd_data.all = data.all;

  GD::foreach_feature<linear_update_data, uint64_t, linear_per_feature_update<feature_mask_off>>(
      *data.all, ec, upd_data);
}

template <uint8_t policy, bool feature_mask_off>
void update_weights(cbzo& data, example& ec)
{
  if (policy == constant_policy)
    constant_update<feature_mask_off>(data, ec);

  else if (policy == linear_policy)
    linear_update<feature_mask_off>(data, ec);

  else
    THROW("Unknown policy encountered: " << policy)
}

void set_minmax(shared_data* sd, float label, bool min_fixed, bool max_fixed)
{
  if (!min_fixed) sd->min_label = std::min(label, sd->min_label);
  if (!max_fixed) sd->max_label = std::max(label, sd->max_label);
}

inline std::string get_pred_repr(example& ec)
{
  return continuous_actions::to_string(ec.pred.pdf, false, std::numeric_limits<float>::max_digits10);
}

void print_audit_features(vw& all, example& ec)
{
  if (all.audit) all.print_text_by_ref(all.stdout_adapter.get(), get_pred_repr(ec), ec.tag);

  GD::print_features(all, ec);
}

// Returns a value close to x and greater than it
inline float close_greater_value(float x)
{
  if (x != 0.f) return std::nextafter(x, std::numeric_limits<float>::infinity());

  // When x==0, nextafter returns a very small value, thus use this instead
  return 1e-5f;
}

// Returns a value close to x and lesser than it
inline float close_lesser_value(float x)
{
  if (x != 0.f) return std::nextafter(x, -std::numeric_limits<float>::infinity());

  // When x==0, nextafter returns a very small value, thus use this instead
  return -1e-5f;
}

// Approximates a uniform pmf over two values 'a' and 'b' as a 2 spike pdf
void approx_pmf_to_pdf(float a, float b, probability_density_function& pdf)
{
  float left = close_lesser_value(a), right = close_greater_value(a);
  float pdf_val = static_cast<float>(0.5 / (right - left));
  pdf.push_back({left, right, pdf_val});

  left = close_lesser_value(b);
  right = close_greater_value(b);
  pdf_val = static_cast<float>(0.5 / (right - left));
  pdf.push_back({left, right, pdf_val});
}

template <uint8_t policy, bool audit_or_hash_inv>
void predict(cbzo& data, base_learner&, example& ec)
{
  ec.pred.pdf.clear();

  float action_centroid = inference<policy>(*data.all, ec);
  set_minmax(data.all->sd, action_centroid, data.min_prediction_supplied, data.max_prediction_supplied);
  action_centroid = std::min(std::max(action_centroid, data.all->sd->min_label), data.all->sd->max_label);

  approx_pmf_to_pdf(action_centroid - data.radius, action_centroid + data.radius, ec.pred.pdf);

  if (audit_or_hash_inv) print_audit_features(*data.all, ec);
}

template <uint8_t policy, bool feature_mask_off, bool audit_or_hash_inv>
void learn(cbzo& data, base_learner& base, example& ec)
{
  // update_weights() doesn't require predict() to be called. It is called
  // to respect --audit, --invert_hash, --predictions for train examples
  predict<policy, audit_or_hash_inv>(data, base, ec);
  update_weights<policy, feature_mask_off>(data, ec);
}

inline void save_load_regressor(vw& all, io_buf& model_file, bool read, bool text)
{
  GD::save_load_regressor(all, model_file, read, text);
}

void save_load(cbzo& data, io_buf& model_file, bool read, bool text)
{
  vw& all = *data.all;
  if (read)
  {
    initialize_regressor(all);
    if (data.all->initial_constant != 0.0f) set_weight(all, constant, 0, data.all->initial_constant);
  }
  if (model_file.num_files() > 0) save_load_regressor(all, model_file, read, text);
}

bool is_labeled(example& ec) { return (!ec.l.cb_cont.costs.empty() && ec.l.cb_cont.costs[0].action != FLT_MAX); }

void report_progress(vw& all, example& ec)
{
  const auto& costs = ec.l.cb_cont.costs;
  all.sd->update(ec.test_only, is_labeled(ec), costs.empty() ? 0.0f : costs[0].cost, ec.weight, ec.get_num_features());
  all.sd->weighted_labels += ec.weight;

  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.logger.quiet)
  {
    all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass,
        ec.test_only ? "unknown" : to_string(costs[0]), get_pred_repr(ec), ec.get_num_features(), all.progress_add,
        all.progress_arg);
  }
}

void output_prediction(vw& all, example& ec)
{
  std::string pred_repr = get_pred_repr(ec);
  for (auto& sink : all.final_prediction_sink) all.print_text_by_ref(sink.get(), pred_repr, ec.tag);
}

void finish_example(vw& all, cbzo&, example& ec)
{
  report_progress(all, ec);
  output_prediction(all, ec);
  VW::finish_example(all, ec);
}

void (*get_learn(vw& all, uint8_t policy, bool feature_mask_off))(cbzo&, base_learner&, example&)
{
  if (policy == constant_policy)
    if (feature_mask_off)
      if (all.audit || all.hash_inv)
        return learn<constant_policy, true, true>;
      else
        return learn<constant_policy, true, false>;

    else if (all.audit || all.hash_inv)
      return learn<constant_policy, false, true>;
    else
      return learn<constant_policy, false, false>;

  else if (policy == linear_policy)
    if (feature_mask_off)
      if (all.audit || all.hash_inv)
        return learn<linear_policy, true, true>;
      else
        return learn<linear_policy, true, false>;

    else if (all.audit || all.hash_inv)
      return learn<linear_policy, false, true>;
    else
      return learn<linear_policy, false, false>;

  else
    THROW("Unknown policy encountered: " << policy)
}

void (*get_predict(vw& all, uint8_t policy))(cbzo&, base_learner&, example&)
{
  if (policy == constant_policy)
    if (all.audit || all.hash_inv)
      return predict<constant_policy, true>;
    else
      return predict<constant_policy, false>;

  else if (policy == linear_policy)
    if (all.audit || all.hash_inv)
      return predict<linear_policy, true>;
    else
      return predict<linear_policy, false>;

  else
    THROW("Unknown policy encountered: " << policy)
}

base_learner* setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<cbzo>();

  std::string policy_str;
  bool cbzo_option = false;

  option_group_definition new_options("Continuous Action Contextual Bandit using Zeroth-Order Optimization");
  new_options
      .add(make_option("cbzo", cbzo_option)
               .keep()
               .necessary()
               .help("Solve 1-slot Continuous Action Contextual Bandit using Zeroth-Order Optimization"))
      .add(make_option("policy", policy_str).default_value("linear").keep().help("Policy/Model to Learn"))
      .add(make_option("radius", data->radius).default_value(0.1f).keep(all.save_resume).help("Exploration Radius"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  bool feature_mask_off = true;
  if (options.was_supplied("feature_mask")) feature_mask_off = false;

  uint8_t policy;
  if (policy_str.compare("constant") == 0)
    policy = constant_policy;
  else if (policy_str.compare("linear") == 0)
    policy = linear_policy;
  else
    THROW("policy must be in {'constant', 'linear'}");

  if (policy == constant_policy)
  {
    if (options.was_supplied("noconstant")) THROW("constant policy can't be learnt when --noconstant is used")

    if (!feature_mask_off)
      *(all.trace_message)
          << "warning: feature_mask used with constant policy (where there is only one weight to learn)." << std::endl;
  }

  all.example_parser->lbl_parser = cb_continuous::the_label_parser;
  data->all = &all;
  data->min_prediction_supplied = options.was_supplied("min_prediction");
  data->max_prediction_supplied = options.was_supplied("max_prediction");

  learner<cbzo, example>& l = init_learner(data, get_learn(all, policy, feature_mask_off), get_predict(all, policy), 0,
      prediction_type_t::pdf, all.get_setupfn_name(setup));

  l.set_save_load(save_load);
  l.set_finish_example(finish_example);

  return make_base(l);
}

}  // namespace cbzo
}  // namespace VW
