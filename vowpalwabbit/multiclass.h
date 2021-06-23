// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
#include "label_parser.h"

struct example;
struct vw;

namespace MULTICLASS
{
struct label_t
{
  uint32_t label;
  float weight;

  label_t() { reset_to_default(); }

  label_t(uint32_t label, float weight) : label(label), weight(weight) {}

  void reset_to_default()
  {
    label = std::numeric_limits<uint32_t>::max();
    weight = 1.f;
  }
};

extern label_parser mc_label;

void print_update_with_probability(vw& all, example& ec, uint32_t prediction);
void print_update_with_score(vw& all, example& ec, uint32_t prediction);

void finish_example(vw& all, example& ec, bool update_loss);

bool test_label(const label_t& ld);

template <class T>
void finish_example(vw& all, T&, example& ec)
{
  finish_example(all, ec, true);
}

template <class T>
void finish_example_without_loss(vw& all, T&, example& ec)
{
  finish_example(all, ec, false);
}
}  // namespace MULTICLASS
