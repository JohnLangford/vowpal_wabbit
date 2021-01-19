// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "v_array.h"
#include "prob_dist_cont.h"

#include <cmath>

namespace VW
{
namespace continuous_actions
{
struct reduction_features
{
  probability_density_function pdf;
  float chosen_action;
  bool is_chosen_action_set() const { return !std::isnan(chosen_action); }
  bool is_pdf_set() const { return pdf.pdf.size() > 0; }

  reduction_features()
  {
    pdf.pdf = v_init<pdf_segment>();
    chosen_action = std::numeric_limits<float>::quiet_NaN();
  }

  ~reduction_features() { pdf.pdf.delete_v(); }

  void clear()
  {
    pdf.pdf.clear();
    chosen_action = std::numeric_limits<float>::quiet_NaN();
  }
};

}  // namespace continuous_actions
}  // namespace VW