// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner_no_throw.h"
#include "cb.h"
#include "action_score.h"
#include "guard.h"

// forward declaration
namespace VW
{
namespace config
{
struct options_i;
}
}  // namespace VW

namespace VW
{
namespace pmf_to_pdf
{
LEARNER::base_learner* setup(config::options_i& options, vw& all);
struct reduction
{
  void predict(example& ec)
  {
    auto swap_label = VW::swap_guard(ec.l.cb, temp_lbl_cb);

    const auto& reduction_features = ec._reduction_features.template get<VW::continuous_actions::reduction_features>();
    if (first_only && reduction_features.is_chosen_action_set())
    {
      float chosen_action = reduction_features.chosen_action;
      const float continuous_range = max_value - min_value;
      const float unit_range = continuous_range / num_actions;

      // discretize chosen action
      const float ac = (chosen_action - min_value) / unit_range;
      auto action = std::min(num_actions - 1, static_cast<uint32_t>(std::floor(ac)));

      temp_pred_a_s.clear();
      temp_pred_a_s.push_back({action, 1.f});
    }
    else
    {
      // scope for saving / restoring prediction
      auto save_prediction = VW::swap_guard(ec.pred.a_s, temp_pred_a_s);
      _p_base->predict(ec);
    }
    transform_prediction(ec);
  }
  void learn(example& ec);
  void init(uint32_t num_actions_, float bandwidth_, float min_value_, float max_value_, bool first_only_,
      bool bandwidth_supplied)
  {
    num_actions = num_actions_;
    bandwidth = bandwidth_;
    min_value = min_value_;
    max_value = max_value_;
    first_only = first_only_;

    float leaf_width = (max_value - min_value) / (num_actions);  // aka unit range
    float half_leaf_width = leaf_width / 2.f;

    if (!bandwidth_supplied) { bandwidth = half_leaf_width; }

    // Translate user provided bandwidth which is in terms of continuous action range (max_value - min_value)
    // to the internal tree bandwidth which is in terms of #actions
    if (bandwidth <= half_leaf_width) { tree_bandwidth = 0; }
    else if (std::fmod((bandwidth), leaf_width) == 0)
    {
      tree_bandwidth = static_cast<uint32_t>((bandwidth) / leaf_width);
    }
    else
    {
      tree_bandwidth = static_cast<uint32_t>((bandwidth) / leaf_width) + 1;
    }
  }

  ~reduction()
  {
    temp_lbl_cb.costs.delete_v();
    temp_pred_a_s.delete_v();
  }

  std::vector<float> pdf_lim;
  uint32_t num_actions;
  uint32_t tree_bandwidth;
  float bandwidth;  // radius
  float min_value;
  float max_value;
  bool first_only;
  LEARNER::single_learner* _p_base;

private:
  void transform_prediction(example& ec)
  {
    const float continuous_range = max_value - min_value;
    const float unit_range = continuous_range / num_actions;

    size_t n = temp_pred_a_s.size();
    assert(n != 0);

    auto score = temp_pred_a_s[0].score;
    // map discrete action (predicted tree leaf) to the continuous value of the centre of the leaf
    auto centre = min_value + temp_pred_a_s[0].action * unit_range + unit_range / 2.0f;

    // if zero bandwidth -> stay inside leaf by smoothing around unit_range / 2 (leaf range is unit_range)
    auto b = !bandwidth ? unit_range / 2.0f : bandwidth;

    pdf_lim.clear();
    if (centre - b != min_value) pdf_lim.push_back(min_value);

    uint32_t l = 0;
    uint32_t r = 0;
    while (l < n || r < n)
    {
      if (centre >= b)
      {
        if (l == n || centre + b < centre - b)
        {
          auto val = std::min(centre + b, max_value);
          pdf_lim.push_back(val);
          r++;
        }
        else if (r == n || centre - b < centre + b)
        {
          auto val = std::max(centre - b, min_value);
          if ((!pdf_lim.empty() && pdf_lim.back() != val) || pdf_lim.empty()) { pdf_lim.push_back(val); }
          l++;
        }
        else if (centre - b == centre + b)
        {
          auto val = std::max(centre - b, min_value);
          if ((!pdf_lim.empty() && pdf_lim.back() != val) || pdf_lim.empty()) { pdf_lim.push_back(val); }
          l++;
          r++;
        }
      }
      else
      {
        // centre < b so lower limit should be min_value (already added to pdf_lim)
        // so need to add centre + b
        auto val = std::min(centre + b, max_value);
        pdf_lim.push_back(val);
        l++;
        r++;
      }
    }

    if (pdf_lim.back() != max_value) pdf_lim.push_back(max_value);

    auto& p_dist = ec.pred.pdf;
    p_dist.clear();

    size_t m = pdf_lim.size();
    l = 0;
    for (uint32_t i = 0; i < m - 1; i++)
    {
      float p = 0;
      // there are 2 ways of knowing that we are entering the pdf limits of the chosen action and thus need to assign a
      // probability: (1) if centre - b < min_value -> pdf_lim would be 'min_value' or
      // (2) pdf_lim is 'centre - b'
      if (l < n && (((centre - min_value) < b && pdf_lim[i] == min_value) || pdf_lim[i] == centre - b))
      {
        // default: 2 * b : 'centre - b' to 'centre + b'
        float actual_b = std::min(max_value, centre + b) - std::max(min_value, centre - b);
        p += score / actual_b;
        l++;
      }
      const float left = pdf_lim[i];
      const float right = pdf_lim[i + 1];

      p_dist.push_back({left, right, p});
    }
  }

  CB::label temp_lbl_cb;
  ACTION_SCORE::action_scores temp_pred_a_s;
};

inline void predict(pmf_to_pdf::reduction& data, VW::LEARNER::single_learner&, example& ec) { data.predict(ec); }

inline void learn(pmf_to_pdf::reduction& data, VW::LEARNER::single_learner&, example& ec);

}  // namespace pmf_to_pdf
}  // namespace VW
