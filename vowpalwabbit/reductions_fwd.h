// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
// Forward declaration header to declare the basic components used by VW reductions

#include <vector>
#include "v_array.h"

// forward declarations
struct example;
using multi_ex = std::vector<example*>;
struct random_state;
struct vw;

namespace VW
{
namespace LEARNER
{
template <class T, class E>
struct learner;
using base_learner = learner<char, char>;
using single_learner = learner<char, example>;
using multi_learner = learner<char, multi_ex>;
}  // namespace LEARNER
namespace config
{
struct options_i;
}  // namespace config

struct setup_base_fn
{
  virtual VW::LEARNER::base_learner* operator()(VW::config::options_i&, vw&) = 0;

  virtual ~setup_base_fn() = default;
};
}  // namespace VW
