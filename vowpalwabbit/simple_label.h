/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#pragma once
#include "label_parser.h"

struct example;
struct vw;

struct label_data
{
  float label;
  // only used for serialization and parsing.  example.weight is used for computation
  // DeSerialized/Parsed values are copied into example in VW::setup_example()
  float serialized_weight;
  // Only used for serialization and parsing.  example.initial is used for computation
  // DeSerialized/Parsed values are copied into example in VW::setup_example()
  float serialized_initial;
};

void return_simple_example(vw& all, void*, example& ec);

extern label_parser simple_label_parser;

bool summarize_holdout_set(vw& all, size_t& no_win_counter);
void print_update(vw& all, example& ec);
void output_and_account_example(vw& all, example& ec);

namespace VW {
  const float NA_0 = 0.f;   // constant to signal intitializing unused member
  const float NA_1 = 1.f;    // constant to signal intitializing unused member
}