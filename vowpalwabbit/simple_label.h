// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include <cstddef>
#include <cfloat>

struct example;
struct vw;

struct label_data
{
  float label = 0.f;

  label_data() { reset_to_default(); }

  label_data(float label) : label(label) {}

  void reset_to_default() { label = FLT_MAX; }
};

struct simple_label_reduction_features
{
  float weight;
  float initial;

  simple_label_reduction_features() { reset_to_default(); }
  simple_label_reduction_features(float weight, float initial) : weight(weight), initial(initial) {}
  void reset_to_default() noexcept
  {
    weight = 1.f;
    initial = 0.f;
  }
};

void return_simple_example(vw& all, void*, example& ec);
bool summarize_holdout_set(vw& all, size_t& no_win_counter);
void print_update(vw& all, example& ec);
void output_and_account_example(vw& all, example& ec);
