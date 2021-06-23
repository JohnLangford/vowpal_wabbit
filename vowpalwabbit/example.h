// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "v_array.h"
#include "no_label.h"
#include "simple_label.h"
#include "multiclass.h"
#include "multilabel.h"
#include "cost_sensitive.h"
#include "cb.h"
#include "constant.h"
#include "feature_group.h"
#include "action_score.h"
#include "example_predict.h"
#include "conditional_contextual_bandit.h"
#include "continuous_actions_reduction_features.h"
#include "ccb_label.h"
#include "slates_label.h"
#include "decision_scores.h"
#include "cb_continuous_label.h"
#include "prob_dist_cont.h"
#include "active_multiclass_prediction.h"

#include <cstdint>
#include <vector>
#include <iostream>

struct vw;
namespace VW
{
void copy_example_data(example* dst, const example* src);
void setup_example(vw& all, example* ae);
}  // namespace VW

struct polylabel
{
  no_label::no_label empty;
  label_data simple;
  MULTICLASS::label_t multi;
  COST_SENSITIVE::label cs;
  CB::label cb;
  VW::cb_continuous::continuous_label cb_cont;
  CCB::label conditional_contextual_bandit;
  VW::slates::label slates;
  CB_EVAL::label cb_eval;
  MULTILABEL::labels multilabels;
};

struct polyprediction
{
  polyprediction() = default;
  ~polyprediction() = default;

  polyprediction(polyprediction&&) = default;
  polyprediction& operator=(polyprediction&&) = default;

  polyprediction(const polyprediction&) = delete;
  polyprediction& operator=(const polyprediction&) = delete;

  float scalar = 0.f;
  v_array<float> scalars;           // a sequence of scalar predictions
  ACTION_SCORE::action_scores a_s;  // a sequence of classes with scores.  Also used for probabilities.
  VW::decision_scores_t decision_scores;
  uint32_t multiclass;
  MULTILABEL::labels multilabels;
  float prob = 0.f;                                          // for --probabilities --csoaa_ldf=mc
  VW::continuous_actions::probability_density_function pdf;  // probability density defined over an action range
  VW::continuous_actions::probability_density_function_value pdf_value;  // probability density value for a given action
  VW::active_multiclass_prediction active_multiclass;
};

float calculate_total_sum_features_squared(bool permutations, example& ec);

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_DEPRECATED_USAGE
struct example : public example_predict  // core example datatype.
{
  example() = default;
  VW_WARNING_STATE_PUSH
  VW_WARNING_DISABLE_DEPRECATED_USAGE
  ~example()
  {
    if (passthrough)
    {
      delete passthrough;
      passthrough = nullptr;
    }
  }
  VW_WARNING_STATE_POP

  example(const example&) = delete;
  example& operator=(const example&) = delete;
  example(example&& other) = default;
  example& operator=(example&& other) = default;

  // input fields
  polylabel l;

  // output prediction
  polyprediction pred;

  float weight = 1.f;  // a relative importance weight for the example, default = 1
  v_array<char> tag;   // An identifier for the example.
  size_t example_counter = 0;

  // helpers
  size_t num_features = 0;  // precomputed, cause it's fast&easy.
  size_t num_features_from_interactions = 0;
  float partial_prediction = 0.f;  // shared data for prediction.
  float updated_prediction = 0.f;  // estimated post-update prediction.
  float loss = 0.f;

  // This value is only used for gd's sensitivity call, but it is costly to
  // calculate. Therefore it is calculated only when needed. Anything that
  // modifies the feature_groups in this example should invalidate this value
  // with reset_total_sum_feat_sq() to ensure it gets recalculated if needed.
  float total_sum_feat_sq = 0.f;
  float confidence = 0.f;
  features* passthrough =
      nullptr;  // if a higher-up reduction wants access to internal state of lower-down reductions, they go here

  bool test_only = false;
  bool end_pass = false;  // special example indicating end of pass.
  bool sorted = false;    // Are the features sorted or not?
  bool is_newline = false;

  // Deprecating a field can make deprecated warnings hard to track down through implicit usage in the constructor.
  // This is deprecated, but we won't mark it so we don't have those issues.
  // VW_DEPRECATED(
  //     "in_use has been removed, examples taken from the pool are assumed to be in use if there is a reference to
  //     them. " "Standalone examples are by definition always in use.")
  bool in_use = true;

  size_t get_num_features() const noexcept { return num_features + num_features_from_interactions; }

  float get_total_sum_feat_sq()
  {
    if (!total_sum_feat_sq_calculated)
    {
      total_sum_feat_sq = calculate_total_sum_features_squared(use_permutations, *this);
      total_sum_feat_sq_calculated = true;
    }
    return total_sum_feat_sq;
  }

  void reset_total_sum_feat_sq()
  {
    total_sum_feat_sq = 0.f;
    total_sum_feat_sq_calculated = false;
  }

  friend void VW::copy_example_data(example* dst, const example* src);
  friend void VW::setup_example(vw& all, example* ae);

private:
  bool total_sum_feat_sq_calculated = false;
  bool use_permutations = false;
};
VW_WARNING_STATE_POP

struct vw;

struct flat_example
{
  polylabel l;
  reduction_features _reduction_features;

  size_t tag_len;
  char* tag;  // An identifier for the example.

  size_t example_counter;
  uint64_t ft_offset;
  float global_weight;

  size_t num_features;      // precomputed, cause it's fast&easy.
  float total_sum_feat_sq;  // precomputed, cause it's kind of fast & easy.
  features fs;              // all the features
};

flat_example* flatten_example(vw& all, example* ec);
flat_example* flatten_sort_example(vw& all, example* ec);
void free_flatten_example(flat_example* fec);

inline int example_is_newline(example const& ec) { return ec.is_newline; }

inline bool valid_ns(char c) { return !(c == '|' || c == ':'); }

inline void add_passthrough_feature_magic(example& ec, uint64_t magic, uint64_t i, float x)
{
  if (ec.passthrough) ec.passthrough->push_back(x, (FNV_prime * magic) ^ i);
}

#define add_passthrough_feature(ec, i, x) \
  add_passthrough_feature_magic(ec, __FILE__[0] * 483901 + __FILE__[1] * 3417 + __FILE__[2] * 8490177, i, x);

typedef std::vector<example*> multi_ex;

namespace VW
{
void return_multiple_example(vw& all, v_array<example*>& examples);

typedef example& (*example_factory_t)(void*);

}  // namespace VW

std::string simple_label_to_string(const example& ec);
std::string cb_label_to_string(const example& ec);
std::string scalar_pred_to_string(const example& ec);
std::string a_s_pred_to_string(const example& ec);
std::string prob_dist_pred_to_string(const example& ec);
std::string multiclass_pred_to_string(const example& ec);
std::string debug_depth_indent_string(const multi_ex& ec);
std::string debug_depth_indent_string(const example& ec);
std::string debug_depth_indent_string(int32_t stack_depth);
std::string cb_label_to_string(const example& ec);
