// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once
// This is the interface for a learning algorithm

#include <iostream>
#include <memory>

#include "memory.h"
#include "multiclass.h"
#include "parser.h"

#include "future_compat.h"
#include "example.h"
#include <memory>
#include "scope_exit.h"
#include "learner_no_throw.h"

const char* to_string(prediction_type_t prediction_type);

namespace VW
{
/// \brief Contains the VW::LEARNER::learner object and utilities for
/// interacting with it.
namespace LEARNER
{
template <class T, class E>
struct learner;

/// \brief Used to type erase the object and pass around common type.
using base_learner = learner<char, char>;

/// \brief Used for reductions that process single ::example objects at at time.
/// It type erases the specific reduction object type.
using single_learner = learner<char, example>;

/// \brief Used for multiline examples where there are several ::example objects
/// required to describe the overall example. It type erases the specific
/// reduction object type.
using multi_learner = learner<char, multi_ex>;

void generic_driver(vw& all);
void generic_driver(const std::vector<vw*>& alls);
void generic_driver_onethread(vw& all);

template <class T, class E>
inline void learner<T, E>::print_example(vw& all, E& ec)
{
  debug_log_message(ec, "print_example");

  if (finish_example_fd.print_example_f == nullptr) THROW("fatal: learner did not register print example fn: " + name);

  finish_example_fd.print_example_f(all, finish_example_fd.data, (void*)&ec);
}

template <class T, class E>
base_learner* learner<T, E>::get_learner_by_name_prefix(std::string reduction_name)
{
  if (name.find(reduction_name) != std::string::npos) { return (base_learner*)this; }
  else
  {
    if (learn_fd.base != nullptr)
      return learn_fd.base->get_learner_by_name_prefix(reduction_name);
    else
      THROW("fatal: could not find in learner chain: " << reduction_name);
  }
}

// multiclass reduction
template <class T, class E, class L>
learner<T, E>& init_multiclass_learner(free_ptr<T>& dat, L* base, void (*learn)(T&, L&, E&),
    void (*predict)(T&, L&, E&), parser* p, size_t ws, const std::string& name,
    prediction_type_t pred_type = prediction_type_t::multiclass, bool learn_returns_prediction = false)
{
  learner<T, E>& l =
      learner<T, E>::init_learner(dat.get(), base, learn, predict, ws, pred_type, name, learn_returns_prediction);

  dat.release();
  l.set_finish_example(MULTICLASS::finish_example<T>);
  p->lbl_parser = MULTICLASS::mc_label;
  return l;
}

template <class T, class E, class L>
learner<T, E>& init_cost_sensitive_learner(free_ptr<T>& dat, L* base, void (*learn)(T&, L&, E&),
    void (*predict)(T&, L&, E&), parser* p, size_t ws, const std::string& name,
    prediction_type_t pred_type = prediction_type_t::multiclass, bool learn_returns_prediction = false)
{
  learner<T, E>& l =
      learner<T, E>::init_learner(dat.get(), base, learn, predict, ws, pred_type, name, learn_returns_prediction);
  dat.release();
  l.set_finish_example(COST_SENSITIVE::finish_example);
  p->lbl_parser = COST_SENSITIVE::cs_label;
  return l;
}

template <class T, class E>
multi_learner* as_multiline(learner<T, E>* l)
{
  if (l->is_multiline)  // Tried to use a singleline reduction as a multiline reduction
    return (multi_learner*)(l);
  THROW("Tried to use a singleline reduction as a multiline reduction");
}

template <class T, class E>
single_learner* as_singleline(learner<T, E>* l)
{
  if (!l->is_multiline)  // Tried to use a multiline reduction as a singleline reduction
    return (single_learner*)(l);
  THROW("Tried to use a multiline reduction as a singleline reduction");
}

template <bool is_learn>
void multiline_learn_or_predict(multi_learner& base, multi_ex& examples, const uint64_t offset, const uint32_t id = 0)
{
  std::vector<uint64_t> saved_offsets;
  saved_offsets.reserve(examples.size());
  for (auto ec : examples)
  {
    saved_offsets.push_back(ec->ft_offset);
    ec->ft_offset = offset;
  }

  // Guard example state restore against throws
  auto restore_guard = VW::scope_exit([&saved_offsets, &examples] {
    for (size_t i = 0; i < examples.size(); i++) { examples[i]->ft_offset = saved_offsets[i]; }
  });

  if (is_learn)
    base.learn(examples, id);
  else
    base.predict(examples, id);
}
}  // namespace LEARNER
}  // namespace VW
