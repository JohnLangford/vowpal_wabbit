// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include "crossplat_compat.h"

#include <cfloat>

#if !defined(VW_NO_INLINE_SIMD)
#  if !defined(__SSE2__) && (defined(_M_AMD64) || defined(_M_X64))
#    define __SSE2__
#  endif

#  if defined(__ARM_NEON__)
#    include <arm_neon.h>
#  elif defined(__SSE2__)
#    include <xmmintrin.h>
#  endif
#endif

#include "accumulate.h"
#include "debug_log.h"
#include "gd.h"
#include "reductions.h"
#include "vw.h"
#include "shared_data.h"

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::gd
#include "io/logger.h"

#define VERSION_SAVE_RESUME_FIX "7.10.1"
#define VERSION_PASS_UINT64 "8.3.3"

using namespace VW::LEARNER;
using namespace VW::config;

namespace logger = VW::io::logger;

// todo:
// 4. Factor various state out of vw&
namespace GD
{
struct gd
{
  //  double normalized_sum_norm_x;
  double total_weight;
  size_t no_win_counter;
  size_t early_stop_thres;
  float initial_constant;
  float neg_norm_power;
  float neg_power_t;
  float sparse_l2;
  float update_multiplier;
  void (*predict)(gd&, base_learner&, example&);
  void (*learn)(gd&, base_learner&, example&);
  void (*update)(gd&, base_learner&, example&);
  float (*sensitivity)(gd&, base_learner&, example&);
  void (*multipredict)(gd&, base_learner&, example&, size_t, size_t, polyprediction*, bool);
  bool adaptive_input;
  bool normalized_input;
  bool adax;
  vw* all;  // parallel, features, parameters
};

void sync_weights(vw& all);

inline float quake_InvSqrt(float x)
{
  // Carmack/Quake/SGI fast method:
  float xhalf = 0.5f * x;
  static_assert(sizeof(int) == sizeof(float), "Floats and ints are converted between, they must be the same size.");
  int i = reinterpret_cast<int&>(x);  // store floating-point bits in integer
  i = 0x5f3759d5 - (i >> 1);          // initial guess for Newton's method
  x = reinterpret_cast<float&>(i);    // convert new bits into float
  x = x * (1.5f - xhalf * x * x);     // One round of Newton's method
  return x;
}

static inline float InvSqrt(float x)
{
#if !defined(VW_NO_INLINE_SIMD)
#  if defined(__ARM_NEON__)
  // Propagate into vector
  float32x2_t v1 = vdup_n_f32(x);
  // Estimate
  float32x2_t e1 = vrsqrte_f32(v1);
  // N-R iteration 1
  float32x2_t e2 = vmul_f32(e1, vrsqrts_f32(v1, vmul_f32(e1, e1)));
  // N-R iteration 2
  float32x2_t e3 = vmul_f32(e2, vrsqrts_f32(v1, vmul_f32(e2, e2)));
  // Extract result
  return vget_lane_f32(e3, 0);
#  elif defined(__SSE2__)
  __m128 eta = _mm_load_ss(&x);
  eta = _mm_rsqrt_ss(eta);
  _mm_store_ss(&x, eta);
#  else
  x = quake_InvSqrt(x);
#  endif
#else
  x = quake_InvSqrt(x);
#endif

  return x;
}
VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_CPP_17_LANG_EXT
template <bool sqrt_rate, bool feature_mask_off, size_t adaptive, size_t normalized, size_t spare>
inline void update_feature(float& update, float x, float& fw)
{
  weight* w = &fw;
  bool modify = x < FLT_MAX && x > -FLT_MAX && (feature_mask_off || fw != 0.);
  if (modify)
  {
    if VW_STD17_CONSTEXPR (spare != 0) { x *= w[spare]; }
    w[0] += update * x;
  }
}

// this deals with few nonzero features vs. all nonzero features issues.
template <bool sqrt_rate, size_t adaptive, size_t normalized>
float average_update(float total_weight, float normalized_sum_norm_x, float neg_norm_power)
{
  if VW_STD17_CONSTEXPR (normalized != 0)
  {
    if (sqrt_rate)
    {
      float avg_norm = (total_weight / normalized_sum_norm_x);
      if (adaptive)
        return std::sqrt(avg_norm);
      else
        return avg_norm;
    }
    else
      return powf((normalized_sum_norm_x / total_weight), neg_norm_power);
  }
  return 1.f;
}

template <bool sqrt_rate, bool feature_mask_off, size_t adaptive, size_t normalized, size_t spare>
void train(gd& g, example& ec, float update)
{
  if VW_STD17_CONSTEXPR (normalized != 0) { update *= g.update_multiplier; }
  VW_DBG(ec) << "gd: train() spare=" << spare << std::endl;
  foreach_feature<float, update_feature<sqrt_rate, feature_mask_off, adaptive, normalized, spare> >(*g.all, ec, update);
}

void end_pass(gd& g)
{
  vw& all = *g.all;
  if (all.save_resume)
  {
    // TODO work out a better system to update state that will be saved in the model.
    if (all.sd->gravity != 0.)
    {
      g.all->options->replace("l1_state", std::to_string(all.sd->gravity));
      g.all->options->get_typed_option<double>("l1_state").value(all.sd->gravity);
    }
    if (all.sd->contraction != 1.)
    {
      g.all->options->replace("l2_state", std::to_string(all.sd->contraction));
      g.all->options->get_typed_option<double>("l2_state").value(all.sd->contraction);
    }
  }
  else
    sync_weights(all);
  if (all.all_reduce != nullptr)
  {
    if (all.weights.adaptive)
      accumulate_weighted_avg(all, all.weights);
    else
      accumulate_avg(all, all.weights, 0);
  }
  all.eta *= all.eta_decay_rate;
  if (all.save_per_pass) save_predictor(all, all.final_regressor_name, all.current_pass);

  if (!all.holdout_set_off)
  {
    if (summarize_holdout_set(all, g.no_win_counter)) finalize_regressor(all, all.final_regressor_name);
    if ((g.early_stop_thres == g.no_win_counter) &&
        ((all.check_holdout_every_n_passes <= 1) || ((all.current_pass % all.check_holdout_every_n_passes) == 0)))
      set_done(all);
  }
}

#include <algorithm>

struct string_value
{
  float v;
  std::string s;
  friend bool operator<(const string_value& first, const string_value& second);
};

bool operator<(const string_value& first, const string_value& second) { return fabsf(first.v) > fabsf(second.v); }

struct audit_results
{
  vw& all;
  const uint64_t offset;
  std::vector<std::string> ns_pre;
  std::vector<string_value> results;
  audit_results(vw& p_all, const size_t p_offset) : all(p_all), offset(p_offset) {}
};

inline void audit_interaction(audit_results& dat, const audit_strings* f)
{
  if (f == nullptr)
  {
    if (!dat.ns_pre.empty()) { dat.ns_pre.pop_back(); }

    return;
  }

  std::string ns_pre;
  if (!dat.ns_pre.empty()) ns_pre += '*';

  if (f->first != "" && ((f->first) != " "))
  {
    ns_pre.append(f->first);
    ns_pre += '^';
  }

  if (f->second != "") { ns_pre.append(f->second); }

  if (!ns_pre.empty()) { dat.ns_pre.push_back(ns_pre); }
}

inline void audit_feature(audit_results& dat, const float ft_weight, const uint64_t ft_idx)
{
  parameters& weights = dat.all.weights;
  uint64_t index = ft_idx & weights.mask();
  size_t stride_shift = weights.stride_shift();

  std::string ns_pre;
  for (std::string& s : dat.ns_pre) ns_pre += s;

  if (dat.all.audit)
  {
    std::ostringstream tempstream;
    tempstream << ':' << (index >> stride_shift) << ':' << ft_weight << ':'
               << trunc_weight(weights[index], static_cast<float>(dat.all.sd->gravity)) *
            static_cast<float>(dat.all.sd->contraction);

    if (weights.adaptive)  // adaptive
      tempstream << '@' << (&weights[index])[1];

    string_value sv = {weights[index] * ft_weight, ns_pre + tempstream.str()};
    dat.results.push_back(sv);
  }

  if ((dat.all.current_pass == 0 || dat.all.training == false) && dat.all.hash_inv)
  {
    // for invert_hash

    if (dat.offset != 0)
    {
      // otherwise --oaa output no features for class > 0.
      std::ostringstream tempstream;
      tempstream << '[' << (dat.offset >> stride_shift) << ']';
      ns_pre += tempstream.str();
    }
    const auto strided_index = index >> stride_shift;
    if (!dat.all.index_name_map.count(strided_index))
      dat.all.index_name_map.insert(std::make_pair(strided_index, ns_pre));
  }
}

void print_lda_features(vw& all, example& ec)
{
  parameters& weights = all.weights;
  uint32_t stride_shift = weights.stride_shift();
  size_t count = 0;
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket) { count += fs.feats.size(); }
  }

  // TODO: Where should audit stuff output to?
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket)
    {
      for (const auto& f : fs.feats.audit_range())
      {
        std::cout << '\t' << f.audit()->first << '^' << f.audit()->second << ':'
                  << ((f.index() >> stride_shift) & all.parse_mask) << ':' << f.value();
        for (size_t k = 0; k < all.lda; k++) std::cout << ':' << (&weights[f.index()])[k];
      }
    }
  }
  std::cout << " total of " << count << " features." << std::endl;
}

void print_features(vw& all, example& ec)
{
  if (all.lda > 0)
    print_lda_features(all, ec);
  else
  {
    audit_results dat(all, ec.ft_offset);

    for (auto& bucket : ec)
    {
      for (auto& fs : bucket)
      {
        if (fs.feats.space_names.size() > 0)
          for (const auto& f : fs.feats.audit_range())
          {
            audit_interaction(dat, f.audit());
            audit_feature(dat, f.value(), f.index() + ec.ft_offset);
            audit_interaction(dat, nullptr);
          }
        else
        {
          for (const auto& f : fs.feats) { audit_feature(dat, f.value(), f.index() + ec.ft_offset); }
        }
        }
    }
    size_t num_interacted_features = 0;
    INTERACTIONS::generate_interactions<audit_results, const uint64_t, audit_feature, true, audit_interaction>(
        all, ec, dat, num_interacted_features);

    stable_sort(dat.results.begin(), dat.results.end());
    if (all.audit)
    {
      for (string_value& sv : dat.results) std::cout << '\t' << sv.s;
      std::cout << std::endl;
    }
  }
}

void print_audit_features(vw& all, example& ec)
{
  if (all.audit) print_result_by_ref(all.stdout_adapter.get(), ec.pred.scalar, -1, ec.tag);
  fflush(stdout);
  print_features(all, ec);
}

float finalize_prediction(shared_data* sd, vw_logger&, float ret)
{
  if (std::isnan(ret))
  {
    ret = 0.;
    logger::errlog_warn("NAN prediction in example {0}, forcing {1}", sd->example_number + 1, ret);
    return ret;
  }
  if (ret > sd->max_label) return sd->max_label;
  if (ret < sd->min_label) return sd->min_label;
  return ret;
}

struct trunc_data
{
  float prediction;
  float gravity;
};

inline void vec_add_trunc(trunc_data& p, const float fx, float& fw)
{
  p.prediction += trunc_weight(fw, p.gravity) * fx;
}

inline float trunc_predict(vw& all, example& ec, double gravity, size_t& num_interacted_features)
{
  const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  trunc_data temp = {simple_red_features.initial, static_cast<float>(gravity)};
  foreach_feature<trunc_data, vec_add_trunc>(all, ec, temp, num_interacted_features);
  return temp.prediction;
}

inline void vec_add_print(float& p, const float fx, float& fw)
{
  // TODO: partial line logging. This function isn't actually called from anywhere though?
  p += fw * fx;
  std::cerr << " + " << fw << "*" << fx;
}

template <bool l1, bool audit>
void predict(gd& g, base_learner&, example& ec)
{
  VW_DBG(ec) << "gd.predict(): ex#=" << ec.example_counter << ", offset=" << ec.ft_offset << std::endl;

  vw& all = *g.all;
  size_t num_interacted_features = 0;
  if (l1)
    ec.partial_prediction = trunc_predict(all, ec, all.sd->gravity, num_interacted_features);
  else
    ec.partial_prediction = inline_predict(all, ec, num_interacted_features);

  ec.num_features_from_interactions = num_interacted_features;
  ec.partial_prediction *= static_cast<float>(all.sd->contraction);
  ec.pred.scalar = finalize_prediction(all.sd, all.logger, ec.partial_prediction);

  VW_DBG(ec) << "gd: predict() " << scalar_pred_to_string(ec) << features_to_string(ec) << std::endl;

  if (audit) print_audit_features(all, ec);
}

template <class T>
inline void vec_add_trunc_multipredict(multipredict_info<T>& mp, const float fx, uint64_t fi)
{
  size_t index = fi;
  for (size_t c = 0; c < mp.count; c++, index += mp.step)
    mp.pred[c].scalar += fx * trunc_weight(mp.weights[index], mp.gravity);
}

template <bool l1, bool audit>
void multipredict(
    gd& g, base_learner&, example& ec, size_t count, size_t step, polyprediction* pred, bool finalize_predictions)
{
  vw& all = *g.all;
  for (size_t c = 0; c < count; c++)
  {
    const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
    pred[c].scalar = simple_red_features.initial;
  }

  size_t num_features_from_interactions = 0;
  if (g.all->weights.sparse)
  {
    multipredict_info<sparse_parameters> mp = {
        count, step, pred, g.all->weights.sparse_weights, static_cast<float>(all.sd->gravity)};
    if (l1)
      foreach_feature<multipredict_info<sparse_parameters>, uint64_t, vec_add_trunc_multipredict>(
          all, ec, mp, num_features_from_interactions);
    else
      foreach_feature<multipredict_info<sparse_parameters>, uint64_t, vec_add_multipredict>(
          all, ec, mp, num_features_from_interactions);
  }
  else
  {
    multipredict_info<dense_parameters> mp = {
        count, step, pred, g.all->weights.dense_weights, static_cast<float>(all.sd->gravity)};
    if (l1)
      foreach_feature<multipredict_info<dense_parameters>, uint64_t, vec_add_trunc_multipredict>(
          all, ec, mp, num_features_from_interactions);
    else
      foreach_feature<multipredict_info<dense_parameters>, uint64_t, vec_add_multipredict>(
          all, ec, mp, num_features_from_interactions);
  }
  ec.num_features_from_interactions = num_features_from_interactions;

  if (all.sd->contraction != 1.)
    for (size_t c = 0; c < count; c++) pred[c].scalar *= static_cast<float>(all.sd->contraction);
  if (finalize_predictions)
    for (size_t c = 0; c < count; c++) pred[c].scalar = finalize_prediction(all.sd, all.logger, pred[c].scalar);
  if (audit)
  {
    for (size_t c = 0; c < count; c++)
    {
      ec.pred.scalar = pred[c].scalar;
      print_audit_features(all, ec);
      ec.ft_offset += static_cast<uint64_t>(step);
    }
    ec.ft_offset -= static_cast<uint64_t>(step * count);
  }
}

struct power_data
{
  float minus_power_t;
  float neg_norm_power;
};

template <bool sqrt_rate, size_t adaptive, size_t normalized>
inline float compute_rate_decay(power_data& s, float& fw)
{
  weight* w = &fw;
  float rate_decay = 1.f;
  if (adaptive)
  {
    if (sqrt_rate)
      rate_decay = InvSqrt(w[adaptive]);
    else
      rate_decay = powf(w[adaptive], s.minus_power_t);
  }
  if VW_STD17_CONSTEXPR (normalized != 0)
  {
    if (sqrt_rate)
    {
      float inv_norm = 1.f / w[normalized];
      if (adaptive)
        rate_decay *= inv_norm;
      else
        rate_decay *= inv_norm * inv_norm;
    }
    else
      rate_decay *= powf(w[normalized] * w[normalized], s.neg_norm_power);
  }
  return rate_decay;
}

struct norm_data
{
  float grad_squared;
  float pred_per_update;
  float norm_x;
  power_data pd;
  float extra_state[4];
};

constexpr float x_min = 1.084202e-19f;
constexpr float x2_min = x_min * x_min;
constexpr float x2_max = FLT_MAX;

template <bool sqrt_rate, bool feature_mask_off, size_t adaptive, size_t normalized, size_t spare, bool stateless>
inline void pred_per_update_feature(norm_data& nd, float x, float& fw)
{
  bool modify = feature_mask_off || fw != 0.;
  if (modify)
  {
    weight* w = &fw;
    float x2 = x * x;
    if (x2 < x2_min)
    {
      x = (x > 0) ? x_min : -x_min;
      x2 = x2_min;
    }
    if (stateless)  // we must not modify the parameter state so introduce a shadow version.
    {
      nd.extra_state[0] = w[0];
      nd.extra_state[adaptive] = w[adaptive];
      nd.extra_state[normalized] = w[normalized];
      w = nd.extra_state;
    }
    if (adaptive) w[adaptive] += nd.grad_squared * x2;
    if VW_STD17_CONSTEXPR (normalized != 0)
    {
      float x_abs = fabsf(x);
      if (x_abs > w[normalized])  // new scale discovered
      {
        if (w[normalized] >
            0.)  // If the normalizer is > 0 then rescale the weight so it's as if the new scale was the old scale.
        {
          if (sqrt_rate)
          {
            float rescale = w[normalized] / x_abs;
            w[0] *= (adaptive ? rescale : rescale * rescale);
          }
          else
          {
            float rescale = x_abs / w[normalized];
            w[0] *= powf(rescale * rescale, nd.pd.neg_norm_power);
          }
        }
        w[normalized] = x_abs;
      }
      float norm_x2 = x2 / (w[normalized] * w[normalized]);
      if (x2 > x2_max)
      {
        norm_x2 = 1;
        logger::errlog_error("your features have too much magnitude");
      }
      nd.norm_x += norm_x2;
    }
    w[spare] = compute_rate_decay<sqrt_rate, adaptive, normalized>(nd.pd, w[0]);
    nd.pred_per_update += x2 * w[spare];
  }
}

bool global_print_features = false;
template <bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive, size_t normalized, size_t spare,
    bool stateless>
float get_pred_per_update(gd& g, example& ec)
{
  // We must traverse the features in _precisely_ the same order as during training.
  label_data& ld = ec.l.simple;
  vw& all = *g.all;

  float grad_squared = ec.weight;
  if (!adax) grad_squared *= all.loss->getSquareGrad(ec.pred.scalar, ld.label);

  if (grad_squared == 0 && !stateless) return 1.;

  norm_data nd = {grad_squared, 0., 0., {g.neg_power_t, g.neg_norm_power}, {0}};
  foreach_feature<norm_data,
      pred_per_update_feature<sqrt_rate, feature_mask_off, adaptive, normalized, spare, stateless> >(all, ec, nd);
  if VW_STD17_CONSTEXPR (normalized != 0)
  {
    if (!stateless)
    {
      g.all->normalized_sum_norm_x += (static_cast<double>(ec.weight)) * nd.norm_x;
      g.total_weight += ec.weight;
      g.update_multiplier = average_update<sqrt_rate, adaptive, normalized>(
          static_cast<float>(g.total_weight), static_cast<float>(g.all->normalized_sum_norm_x), g.neg_norm_power);
    }
    else
    {
      float nsnx = (static_cast<float>(g.all->normalized_sum_norm_x)) + ec.weight * nd.norm_x;
      float tw = static_cast<float>(g.total_weight) + ec.weight;
      g.update_multiplier = average_update<sqrt_rate, adaptive, normalized>(tw, nsnx, g.neg_norm_power);
    }
    nd.pred_per_update *= g.update_multiplier;
  }
  return nd.pred_per_update;
}

template <bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive, size_t normalized, size_t spare,
    bool stateless>
float sensitivity(gd& g, example& ec)
{
  if VW_STD17_CONSTEXPR (adaptive || normalized)
    return get_pred_per_update<sqrt_rate, feature_mask_off, adax, adaptive, normalized, spare, stateless>(g, ec);
  else
  {
    _UNUSED(g);
    return ec.get_total_sum_feat_sq();
  }
}
VW_WARNING_STATE_POP

template <size_t adaptive>
float get_scale(gd& g, example& /* ec */, float weight)
{
  float update_scale = g.all->eta * weight;
  if (!adaptive)
  {
    float t = static_cast<float>(
        g.all->sd->t + weight - g.all->sd->weighted_holdout_examples - g.all->sd->weighted_unlabeled_examples);
    update_scale *= powf(t, g.neg_power_t);
  }
  return update_scale;
}

template <bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive, size_t normalized, size_t spare>
float sensitivity(gd& g, base_learner& /* base */, example& ec)
{
  return get_scale<adaptive>(g, ec, 1.) *
      sensitivity<sqrt_rate, feature_mask_off, adax, adaptive, normalized, spare, true>(g, ec);
}

template <bool sparse_l2, bool invariant, bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive,
    size_t normalized, size_t spare>
float compute_update(gd& g, example& ec)
{
  // invariant: not a test label, importance weight > 0
  const label_data& ld = ec.l.simple;
  vw& all = *g.all;

  float update = 0.;
  ec.updated_prediction = ec.pred.scalar;
  if (all.loss->getLoss(all.sd, ec.pred.scalar, ld.label) > 0.)
  {
    float pred_per_update = sensitivity<sqrt_rate, feature_mask_off, adax, adaptive, normalized, spare, false>(g, ec);
    float update_scale = get_scale<adaptive>(g, ec, ec.weight);
    if (invariant)
      update = all.loss->getUpdate(ec.pred.scalar, ld.label, update_scale, pred_per_update);
    else
      update = all.loss->getUnsafeUpdate(ec.pred.scalar, ld.label, update_scale);
    // changed from ec.partial_prediction to ld.prediction
    ec.updated_prediction += pred_per_update * update;

    if (all.reg_mode && std::fabs(update) > 1e-8)
    {
      double dev1 = all.loss->first_derivative(all.sd, ec.pred.scalar, ld.label);
      double eta_bar = (fabs(dev1) > 1e-8) ? (-update / dev1) : 0.0;
      if (fabs(dev1) > 1e-8) all.sd->contraction *= (1. - all.l2_lambda * eta_bar);
      update /= static_cast<float>(all.sd->contraction);
      all.sd->gravity += eta_bar * all.l1_lambda;
    }
  }

  if (sparse_l2) update -= g.sparse_l2 * ec.pred.scalar;

  if (std::isnan(update))
  {
    logger::errlog_warn("update is NAN, replacing with 0");
    update = 0.;
  }

  return update;
}

template <bool sparse_l2, bool invariant, bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive,
    size_t normalized, size_t spare>
void update(gd& g, base_learner&, example& ec)
{
  // invariant: not a test label, importance weight > 0
  float update;
  if ((update = compute_update<sparse_l2, invariant, sqrt_rate, feature_mask_off, adax, adaptive, normalized, spare>(
           g, ec)) != 0.)
    train<sqrt_rate, feature_mask_off, adaptive, normalized, spare>(g, ec, update);

  if (g.all->sd->contraction < 1e-9 || g.all->sd->gravity > 1e3)  // updating weights now to avoid numerical instability
    sync_weights(*g.all);
}

template <bool sparse_l2, bool invariant, bool sqrt_rate, bool feature_mask_off, bool adax, size_t adaptive,
    size_t normalized, size_t spare>
void learn(gd& g, base_learner& base, example& ec)
{
  // invariant: not a test label, importance weight > 0
  assert(ec.l.simple.label != FLT_MAX);
  assert(ec.weight > 0.);
  g.predict(g, base, ec);
  update<sparse_l2, invariant, sqrt_rate, feature_mask_off, adax, adaptive, normalized, spare>(g, base, ec);
}

void sync_weights(vw& all)
{
  // todo, fix length dependence
  if (all.sd->gravity == 0. && all.sd->contraction == 1.)  // to avoid unnecessary weight synchronization
    return;

  if (all.weights.sparse)
    for (weight& w : all.weights.sparse_weights)
      w = trunc_weight(w, static_cast<float>(all.sd->gravity)) * static_cast<float>(all.sd->contraction);
  else
    for (weight& w : all.weights.dense_weights)
      w = trunc_weight(w, static_cast<float>(all.sd->gravity)) * static_cast<float>(all.sd->contraction);

  all.sd->gravity = 0.;
  all.sd->contraction = 1.;
}

size_t write_index(io_buf& model_file, std::stringstream& msg, bool text, uint32_t num_bits, uint64_t i)
{
  size_t brw;
  uint32_t old_i = 0;

  msg << i;

  if (num_bits < 31)
  {
    old_i = static_cast<uint32_t>(i);
    brw = bin_text_write_fixed(model_file, reinterpret_cast<char*>(&old_i), sizeof(old_i), msg, text);
  }
  else
    brw = bin_text_write_fixed(model_file, reinterpret_cast<char*>(&i), sizeof(i), msg, text);

  return brw;
}

template <class T>
void save_load_regressor(vw& all, io_buf& model_file, bool read, bool text, T& weights)
{
  size_t brw = 1;

  if (all.print_invert)  // write readable model with feature names
  {
    std::stringstream msg;

    for (auto it = weights.begin(); it != weights.end(); ++it)
    {
      const auto weight_value = *it;
      if (*it != 0.f)
      {
        const auto weight_index = it.index() >> weights.stride_shift();

        const auto map_it = all.index_name_map.find(weight_index);
        if (map_it != all.index_name_map.end())
        {
          msg << map_it->second;
          bin_text_write_fixed(model_file, nullptr /*unused*/, 0 /*unused*/, msg, true);
        }

        msg << ":" << weight_index << ":" << weight_value << "\n";
        bin_text_write_fixed(model_file, nullptr /*unused*/, 0 /*unused*/, msg, true);
      }
    }
    return;
  }

  uint64_t i = 0;
  uint32_t old_i = 0;
  uint64_t length = static_cast<uint64_t>(1) << all.num_bits;
  if (read) do
    {
      brw = 1;
      if (all.num_bits < 31)  // backwards compatible
      {
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(&old_i), sizeof(old_i), "");
        i = old_i;
      }
      else
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(&i), sizeof(i), "");
      if (brw > 0)
      {
        if (i >= length)
          THROW("Model content is corrupted, weight vector index " << i << " must be less than total vector length "
                                                                   << length);
        weight* v = &weights.strided_index(i);
        brw += model_file.bin_read_fixed(reinterpret_cast<char*>(&(*v)), sizeof(*v), "");
      }
    } while (brw > 0);
  else  // write
    for (typename T::iterator v = weights.begin(); v != weights.end(); ++v)
      if (*v != 0.)
      {
        i = v.index() >> weights.stride_shift();
        std::stringstream msg;

        brw = write_index(model_file, msg, text, all.num_bits, i);
        msg << ":" << *v << "\n";
        brw += bin_text_write_fixed(model_file, (char*)&(*v), sizeof(*v), msg, text);
      }
}

void save_load_regressor(vw& all, io_buf& model_file, bool read, bool text)
{
  if (all.weights.sparse)
    save_load_regressor(all, model_file, read, text, all.weights.sparse_weights);
  else
    save_load_regressor(all, model_file, read, text, all.weights.dense_weights);
}

template <class T>
void save_load_online_state(
    vw& all, io_buf& model_file, bool read, bool text, gd* g, std::stringstream& msg, uint32_t ftrl_size, T& weights)
{
  uint64_t length = static_cast<uint64_t>(1) << all.num_bits;

  uint64_t i = 0;
  uint32_t old_i = 0;
  size_t brw = 1;

  if (read) do
    {
      brw = 1;
      if (all.num_bits < 31)  // backwards compatible
      {
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(&old_i), sizeof(old_i), "");
        i = old_i;
      }
      else
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(&i), sizeof(i), "");
      if (brw > 0)
      {
        if (i >= length)
          THROW("Model content is corrupted, weight vector index " << i << " must be less than total vector length "
                                                                   << length);
        weight buff[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (ftrl_size > 0)
          brw += model_file.bin_read_fixed(reinterpret_cast<char*>(buff), sizeof(buff[0]) * ftrl_size, "");
        else if (g == nullptr || (!g->adaptive_input && !g->normalized_input))
          brw += model_file.bin_read_fixed(reinterpret_cast<char*>(buff), sizeof(buff[0]), "");
        else if ((g->adaptive_input && !g->normalized_input) || (!g->adaptive_input && g->normalized_input))
          brw += model_file.bin_read_fixed(reinterpret_cast<char*>(buff), sizeof(buff[0]) * 2, "");
        else  // adaptive and normalized
          brw += model_file.bin_read_fixed(reinterpret_cast<char*>(buff), sizeof(buff[0]) * 3, "");
        uint32_t stride = 1 << weights.stride_shift();
        weight* v = &weights.strided_index(i);
        for (size_t j = 0; j < stride; j++) v[j] = buff[j];
      }
    } while (brw > 0);
  else  // write binary or text
    for (typename T::iterator v = weights.begin(); v != weights.end(); ++v)
    {
      i = v.index() >> weights.stride_shift();

      if (all.print_invert)  // write readable model with feature names
      {
        if (*v != 0.f)
        {
          const auto map_it = all.index_name_map.find(i);
          if (map_it != all.index_name_map.end())
          {
            msg << map_it->second << ":";
            bin_text_write_fixed(model_file, nullptr /*unused*/, 0 /*unused*/, msg, true);
          }
        }
      }

      if (ftrl_size == 3)
      {
        if (*v != 0. || (&(*v))[1] != 0. || (&(*v))[2] != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << " " << (&(*v))[1] << " " << (&(*v))[2] << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), 3 * sizeof(*v), msg, text);
        }
      }
      else if (ftrl_size == 4)
      {
        if (*v != 0. || (&(*v))[1] != 0. || (&(*v))[2] != 0. || (&(*v))[3] != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << " " << (&(*v))[1] << " " << (&(*v))[2] << " " << (&(*v))[3] << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), 4 * sizeof(*v), msg, text);
        }
      }
      else if (ftrl_size == 6)
      {
        if (*v != 0. || (&(*v))[1] != 0. || (&(*v))[2] != 0. || (&(*v))[3] != 0. || (&(*v))[4] != 0. ||
            (&(*v))[5] != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << " " << (&(*v))[1] << " " << (&(*v))[2] << " " << (&(*v))[3] << " " << (&(*v))[4] << " "
              << (&(*v))[5] << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), 6 * sizeof(*v), msg, text);
        }
      }
      else if (g == nullptr || (!all.weights.adaptive && !all.weights.normalized))
      {
        if (*v != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), sizeof(*v), msg, text);
        }
      }
      else if ((all.weights.adaptive && !all.weights.normalized) || (!all.weights.adaptive && all.weights.normalized))
      {
        // either adaptive or normalized
        if (*v != 0. || (&(*v))[1] != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << " " << (&(*v))[1] << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), 2 * sizeof(*v), msg, text);
        }
      }
      else
      {
        // adaptive and normalized
        if (*v != 0. || (&(*v))[1] != 0. || (&(*v))[2] != 0.)
        {
          brw = write_index(model_file, msg, text, all.num_bits, i);
          msg << ":" << *v << " " << (&(*v))[1] << " " << (&(*v))[2] << "\n";
          brw += bin_text_write_fixed(model_file, (char*)&(*v), 3 * sizeof(*v), msg, text);
        }
      }
    }
}

void save_load_online_state(
    vw& all, io_buf& model_file, bool read, bool text, double& total_weight, gd* g, uint32_t ftrl_size)
{
  std::stringstream msg;

  msg << "initial_t " << all.initial_t << "\n";
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&all.initial_t), sizeof(all.initial_t), "", read, msg, text);

  msg << "norm normalizer " << all.normalized_sum_norm_x << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.normalized_sum_norm_x),
      sizeof(all.normalized_sum_norm_x), "", read, msg, text);

  msg << "t " << all.sd->t << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->t), sizeof(all.sd->t), "", read, msg, text);

  msg << "sum_loss " << all.sd->sum_loss << "\n";
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&all.sd->sum_loss), sizeof(all.sd->sum_loss), "", read, msg, text);

  msg << "sum_loss_since_last_dump " << all.sd->sum_loss_since_last_dump << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->sum_loss_since_last_dump),
      sizeof(all.sd->sum_loss_since_last_dump), "", read, msg, text);

  float dump_interval = all.sd->dump_interval;
  msg << "dump_interval " << dump_interval << "\n";
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&dump_interval), sizeof(dump_interval), "", read, msg, text);
  if (!read || (all.training && all.preserve_performance_counters))  // update dump_interval from input model
    all.sd->dump_interval = dump_interval;

  msg << "min_label " << all.sd->min_label << "\n";
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&all.sd->min_label), sizeof(all.sd->min_label), "", read, msg, text);

  msg << "max_label " << all.sd->max_label << "\n";
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&all.sd->max_label), sizeof(all.sd->max_label), "", read, msg, text);

  msg << "weighted_labeled_examples " << all.sd->weighted_labeled_examples << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->weighted_labeled_examples),
      sizeof(all.sd->weighted_labeled_examples), "", read, msg, text);

  msg << "weighted_labels " << all.sd->weighted_labels << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->weighted_labels),
      sizeof(all.sd->weighted_labels), "", read, msg, text);

  msg << "weighted_unlabeled_examples " << all.sd->weighted_unlabeled_examples << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->weighted_unlabeled_examples),
      sizeof(all.sd->weighted_unlabeled_examples), "", read, msg, text);

  msg << "example_number " << all.sd->example_number << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->example_number),
      sizeof(all.sd->example_number), "", read, msg, text);

  msg << "total_features " << all.sd->total_features << "\n";
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->total_features),
      sizeof(all.sd->total_features), "", read, msg, text);

  if (!read || all.model_file_ver >= VERSION_SAVE_RESUME_FIX)
  {
    // restore some data to allow --save_resume work more accurate

    // fix average loss
    msg << "total_weight " << total_weight << "\n";
    bin_text_read_write_fixed(
        model_file, reinterpret_cast<char*>(&total_weight), sizeof(total_weight), "", read, msg, text);

    // fix "loss since last" for first printed out example details
    msg << "sd::oec.weighted_labeled_examples " << all.sd->old_weighted_labeled_examples << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&all.sd->old_weighted_labeled_examples),
        sizeof(all.sd->old_weighted_labeled_examples), "", read, msg, text);

    // fix "number of examples per pass"
    msg << "current_pass " << all.current_pass << "\n";
    if (all.model_file_ver >= VERSION_PASS_UINT64)
      bin_text_read_write_fixed(
          model_file, reinterpret_cast<char*>(&all.current_pass), sizeof(all.current_pass), "", read, msg, text);
    else  // backwards compatiblity.
    {
      size_t temp_pass = static_cast<size_t>(all.current_pass);
      bin_text_read_write_fixed(
          model_file, reinterpret_cast<char*>(&temp_pass), sizeof(temp_pass), "", read, msg, text);
      all.current_pass = temp_pass;
    }
  }

  if (read &&
      (!all.training ||
          !all.preserve_performance_counters))  // reset various things so that we report test set performance properly
  {
    all.sd->sum_loss = 0;
    all.sd->sum_loss_since_last_dump = 0;
    all.sd->weighted_labeled_examples = 0.;
    all.sd->weighted_labels = 0.;
    all.sd->weighted_unlabeled_examples = 0.;
    all.sd->old_weighted_labeled_examples = 0.;
    all.sd->example_number = 0;
    all.sd->total_features = 0;
    all.current_pass = 0;
  }
  if (all.weights.sparse)
    save_load_online_state(all, model_file, read, text, g, msg, ftrl_size, all.weights.sparse_weights);
  else
    save_load_online_state(all, model_file, read, text, g, msg, ftrl_size, all.weights.dense_weights);
}

void save_load(gd& g, io_buf& model_file, bool read, bool text)
{
  vw& all = *g.all;
  if (read)
  {
    initialize_regressor(all);

    if (all.weights.adaptive && all.initial_t > 0)
    {
      float init_weight = all.initial_weight;
      float init_t = all.initial_t;
      auto initial_gd_weight_initializer = [init_weight, init_t](weight* weights, uint64_t /*index*/) {
        weights[0] = init_weight;
        weights[1] = init_t;
      };

      all.weights.set_default(initial_gd_weight_initializer);

      // for adaptive update, we interpret initial_t as previously seeing initial_t fake datapoints, all with squared
      // gradient=1 NOTE: this is not invariant to the scaling of the data (i.e. when combined with normalized). Since
      // scaling the data scales the gradient, this should ideally be feature_range*initial_t, or something like that.
      // We could potentially fix this by just adding this base quantity times the current range to the sum of gradients
      // stored in memory at each update, and always start sum of gradients to 0, at the price of additional additions
      // and multiplications during the update...
    }
    if (g.initial_constant != 0.0) VW::set_weight(all, constant, 0, g.initial_constant);
  }

  if (model_file.num_files() > 0)
  {
    bool resume = all.save_resume;
    std::stringstream msg;
    msg << ":" << resume << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&resume), sizeof(resume), "", read, msg, text);
    if (resume)
    {
      if (read && all.model_file_ver < VERSION_SAVE_RESUME_FIX)
        *(all.trace_message)
            << std::endl
            << "WARNING: --save_resume functionality is known to have inaccuracy in model files version less than "
            << VERSION_SAVE_RESUME_FIX << std::endl
            << std::endl;
      save_load_online_state(all, model_file, read, text, g.total_weight, &g);
    }
    else
      save_load_regressor(all, model_file, read, text);
  }
  if (!all.training)  // If the regressor was saved as --save_resume, then when testing we want to materialize the
                      // weights.
    sync_weights(all);
}

template <bool sparse_l2, bool invariant, bool sqrt_rate, bool feature_mask_off, uint64_t adaptive, uint64_t normalized,
    uint64_t spare, uint64_t next>
uint64_t set_learn(vw& all, gd& g)
{
  all.normalized_idx = normalized;
  if (g.adax)
  {
    g.learn = learn<sparse_l2, invariant, sqrt_rate, feature_mask_off, true, adaptive, normalized, spare>;
    g.update = update<sparse_l2, invariant, sqrt_rate, feature_mask_off, true, adaptive, normalized, spare>;
    g.sensitivity = sensitivity<sqrt_rate, feature_mask_off, true, adaptive, normalized, spare>;
    return next;
  }
  else
  {
    g.learn = learn<sparse_l2, invariant, sqrt_rate, feature_mask_off, false, adaptive, normalized, spare>;
    g.update = update<sparse_l2, invariant, sqrt_rate, feature_mask_off, false, adaptive, normalized, spare>;
    g.sensitivity = sensitivity<sqrt_rate, feature_mask_off, false, adaptive, normalized, spare>;
    return next;
  }
}

template <bool sparse_l2, bool invariant, bool sqrt_rate, uint64_t adaptive, uint64_t normalized, uint64_t spare,
    uint64_t next>
uint64_t set_learn(vw& all, bool feature_mask_off, gd& g)
{
  all.normalized_idx = normalized;
  if (feature_mask_off)
    return set_learn<sparse_l2, invariant, sqrt_rate, true, adaptive, normalized, spare, next>(all, g);
  else
    return set_learn<sparse_l2, invariant, sqrt_rate, false, adaptive, normalized, spare, next>(all, g);
}

template <bool invariant, bool sqrt_rate, uint64_t adaptive, uint64_t normalized, uint64_t spare, uint64_t next>
uint64_t set_learn(vw& all, bool feature_mask_off, gd& g)
{
  if (g.sparse_l2 > 0.f)
    return set_learn<true, invariant, sqrt_rate, adaptive, normalized, spare, next>(all, feature_mask_off, g);
  else
    return set_learn<false, invariant, sqrt_rate, adaptive, normalized, spare, next>(all, feature_mask_off, g);
}

template <bool sqrt_rate, uint64_t adaptive, uint64_t normalized, uint64_t spare, uint64_t next>
uint64_t set_learn(vw& all, bool feature_mask_off, gd& g)
{
  if (all.invariant_updates)
    return set_learn<true, sqrt_rate, adaptive, normalized, spare, next>(all, feature_mask_off, g);
  else
    return set_learn<false, sqrt_rate, adaptive, normalized, spare, next>(all, feature_mask_off, g);
}

template <bool sqrt_rate, uint64_t adaptive, uint64_t spare>
uint64_t set_learn(vw& all, bool feature_mask_off, gd& g)
{
  // select the appropriate learn function based on adaptive, normalization, and feature mask
  if (all.weights.normalized)
    return set_learn<sqrt_rate, adaptive, adaptive + 1, adaptive + 2, adaptive + 3>(all, feature_mask_off, g);
  else
    return set_learn<sqrt_rate, adaptive, 0, spare, spare + 1>(all, feature_mask_off, g);
}

template <bool sqrt_rate>
uint64_t set_learn(vw& all, bool feature_mask_off, gd& g)
{
  if (all.weights.adaptive)
    return set_learn<sqrt_rate, 1, 2>(all, feature_mask_off, g);
  else
    return set_learn<sqrt_rate, 0, 0>(all, feature_mask_off, g);
}

uint64_t ceil_log_2(uint64_t v)
{
  if (v == 0)
    return 0;
  else
    return 1 + ceil_log_2(v >> 1);
}

base_learner* setup(options_i& options, vw& all)
{
  auto g = scoped_calloc_or_throw<gd>();

  bool sgd = false;
  bool adaptive = false;
  bool adax = false;
  bool invariant = false;
  bool normalized = false;

  option_group_definition new_options("Gradient Descent options");
  new_options.add(make_option("sgd", sgd).help("use regular stochastic gradient descent update.").keep(all.save_resume))
      .add(make_option("adaptive", adaptive).help("use adaptive, individual learning rates.").keep(all.save_resume))
      .add(make_option("adax", adax).help("use adaptive learning rates with x^2 instead of g^2x^2"))
      .add(make_option("invariant", invariant).help("use safe/importance aware updates.").keep(all.save_resume))
      .add(make_option("normalized", normalized).help("use per feature normalized updates").keep(all.save_resume))
      .add(make_option("sparse_l2", g->sparse_l2).default_value(0.f).help("use per feature normalized updates"))
      .add(make_option("l1_state", all.sd->gravity)
               .keep(all.save_resume)
               .default_value(0.)
               .help("use per feature normalized updates"))
      .add(make_option("l2_state", all.sd->contraction)
               .keep(all.save_resume)
               .default_value(1.)
               .help("use per feature normalized updates"));
  options.add_and_parse(new_options);

  g->all = &all;
  g->all->normalized_sum_norm_x = 0;
  g->no_win_counter = 0;
  g->total_weight = 0.;
  all.weights.adaptive = true;
  all.weights.normalized = true;
  g->neg_norm_power = (all.weights.adaptive ? (all.power_t - 1.f) : -1.f);
  g->neg_power_t = -all.power_t;

  if (all.initial_t > 0)  // for the normalized update: if initial_t is bigger than 1 we interpret this as if we had
                          // seen (all.initial_t) previous fake datapoints all with norm 1
  {
    g->all->normalized_sum_norm_x = all.initial_t;
    g->total_weight = all.initial_t;
  }

  bool feature_mask_off = true;
  if (options.was_supplied("feature_mask")) feature_mask_off = false;

  if (!all.holdout_set_off)
  {
    all.sd->holdout_best_loss = FLT_MAX;
    g->early_stop_thres = options.get_typed_option<size_t>("early_terminate").value();
  }

  g->initial_constant = all.initial_constant;

  if (sgd || adaptive || invariant || normalized)
  {
    // nondefault
    all.weights.adaptive = adaptive;
    all.invariant_updates = all.training && invariant;
    all.weights.normalized = normalized;

    if (!options.was_supplied("learning_rate") && !options.was_supplied("l") &&
        !(all.weights.adaptive && all.weights.normalized))
      all.eta = 10;  // default learning rate to 10 for non default update rule

    // if not using normalized or adaptive, default initial_t to 1 instead of 0
    if (!all.weights.adaptive && !all.weights.normalized)
    {
      if (!options.was_supplied("initial_t"))
      {
        all.sd->t = 1.f;
        all.initial_t = 1.f;
      }
      all.eta *= powf(static_cast<float>(all.sd->t), all.power_t);
    }
  }
  else
  {
    all.invariant_updates = all.training;
  }
  g->adaptive_input = all.weights.adaptive;
  g->normalized_input = all.weights.normalized;

  all.weights.adaptive = all.weights.adaptive && all.training;
  all.weights.normalized = all.weights.normalized && all.training;

  if (adax) g->adax = all.training && adax;

  if (g->adax && !all.weights.adaptive) THROW("Cannot use adax without adaptive");

  if (pow(static_cast<double>(all.eta_decay_rate), static_cast<double>(all.numpasses)) < 0.0001)
    *(all.trace_message) << "Warning: the learning rate for the last pass is multiplied by: "
                         << pow(static_cast<double>(all.eta_decay_rate), static_cast<double>(all.numpasses))
                         << " adjust --decay_learning_rate larger to avoid this." << std::endl;

  if (all.reg_mode % 2)
    if (all.audit || all.hash_inv)
    {
      g->predict = predict<true, true>;
      g->multipredict = multipredict<true, true>;
    }
    else
    {
      g->predict = predict<true, false>;
      g->multipredict = multipredict<true, false>;
    }
  else if (all.audit || all.hash_inv)
  {
    g->predict = predict<false, true>;
    g->multipredict = multipredict<false, true>;
  }
  else
  {
    g->predict = predict<false, false>;
    g->multipredict = multipredict<false, false>;
  }

  uint64_t stride;
  if (all.power_t == 0.5)
    stride = set_learn<true>(all, feature_mask_off, *g.get());
  else
    stride = set_learn<false>(all, feature_mask_off, *g.get());

  all.weights.stride_shift(static_cast<uint32_t>(ceil_log_2(stride - 1)));

  gd* bare = g.get();
  learner<gd, example>& ret = init_learner(g, g->learn, bare->predict,
      (static_cast<uint64_t>(1) << all.weights.stride_shift()), all.get_setupfn_name(setup), true);
  ret.set_sensitivity(bare->sensitivity);
  ret.set_multipredict(bare->multipredict);
  ret.set_update(bare->update);
  ret.set_save_load(save_load);
  ret.set_end_pass(end_pass);
  return make_base(ret);
}

}  // namespace GD
