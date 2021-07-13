// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "reductions_fwd.h"

#include "slates_label.h"
#include "learner.h"

#include <string>
#include <vector>

namespace VW
{
namespace slates
{
struct slates_data
{
private:
  std::vector<label> _stashed_labels;

  /*
  The primary job of this reduction is to convert slate labels to a form CCB can process.
  Note: Shared must come before action and slot, and all action examples must come before
  slot examples.
  This is an example of this conversion in the form of textual labels:
    slates shared 0.8
    slates action 0
    slates action 0
    slates action 0
    slates action 1
    slates action 1
    slates slot 1:0.8
    slates slot 0:0.6

    ccb shared
    ccb action
    ccb action
    ccb action
    ccb action
    ccb action
    ccb slot 1:0.8:0.8 0,1,2
    ccb slot 3:0.8:0.6 3,4
  */
  template <bool is_learn>
  void learn_or_predict(VW::LEARNER::multi_learner& base, multi_ex& examples);

public:
  void learn(VW::LEARNER::multi_learner& base, multi_ex& examples);
  void predict(VW::LEARNER::multi_learner& base, multi_ex& examples);
};

VW::LEARNER::base_learner* slates_setup(VW::setup_base_fn&, VW::config::options_i& options, vw& all);
std::string generate_slates_label_printout(const std::vector<example*>& slots);
}  // namespace slates
}  // namespace VW