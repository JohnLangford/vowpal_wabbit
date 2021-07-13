// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

typedef unsigned char namespace_index;

#include "constant.h"
#include "future_compat.h"
#include "reduction_features.h"
#include "feature_group.h"
#include "v_array.h"
#include "namespaced_feature_store.h"

#include <vector>
#include <set>
#include <unordered_set>
#include <array>
// Mutex cannot be used in managed C++, tell the compiler that this is unmanaged even if included in a managed
// project.
#ifdef _M_CEE
#  pragma managed(push, off)
#  undef _M_CEE
#  include <mutex>
#  define _M_CEE 001
#  pragma managed(pop)
#else
#  include <mutex>
#endif

struct indices_proxy_obj
{
  VW::namespaced_feature_store* feature_space;

  std::vector<namespace_index>::const_iterator begin() const { return feature_space->indices().begin(); }
  std::vector<namespace_index>::const_iterator end() const { return feature_space->indices().end(); }
  // TODO this needs to be fixed to be resilient to duplicated indices
  size_t size() const { return feature_space->indices().size(); }
  bool empty() const { return feature_space->indices().empty(); }
};

struct example_predict
{
  using iterator = VW::namespaced_feature_store::iterator;

  example_predict() { indices.feature_space = &feature_space; }
  ~example_predict() = default;
  example_predict(const example_predict&) = delete;
  example_predict& operator=(const example_predict&) = delete;
  example_predict(example_predict&& other) = default;
  example_predict& operator=(example_predict&& other) = default;

  /// If feature_space is modified this iterator is invalidated.
  inline iterator begin() { return feature_space.begin(); }
  /// If feature_space is modified this iterator is invalidated.
  inline iterator end() { return feature_space.end(); }

  indices_proxy_obj indices;
  VW::namespaced_feature_store feature_space;

  uint64_t ft_offset = 0;  // An offset for all feature values.

  // Interactions are specified by this struct's interactions vector of vectors of unsigned characters, where each
  // vector is an interaction and each char is a namespace.
  std::vector<std::vector<namespace_index>>* interactions = nullptr;
  reduction_features _reduction_features;

  // Used for debugging reductions.  Keeps track of current reduction level.
  uint32_t _debug_current_reduction_depth = 0;
};

// make sure we have an exception safe version of example_predict

class VW_DEPRECATED(
    "example_predict is now RAII based. That class can be used instead. This will be removed in VW 9.0.")
    safe_example_predict : public example_predict
{
public:
  safe_example_predict();
  ~safe_example_predict() = default;

  void clear();
};

std::string features_to_string(const example_predict& ec);
std::string depth_indent_string(const example_predict& ec);
std::string depth_indent_string(int32_t stack_depth);
