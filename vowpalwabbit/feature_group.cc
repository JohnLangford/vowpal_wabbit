// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "feature_group.h"

#include "v_array.h"

#include <vector>
#include <algorithm>
#include <utility>

struct feature_slice  // a helper struct for functions using the set {v,i,space_name}
{
  feature_value x;
  feature_index weight_index;
  audit_strings space_name;
};

void features::free_space_names(size_t i) { space_names.erase(space_names.begin() + i, space_names.end()); }

void features::clear()
{
  sum_feat_sq = 0.f;
  index = 0;
  hash = 0;
  values.clear();
  indicies.clear();
  space_names.clear();
}

void features::truncate_to(const iterator& pos) { truncate_to(std::distance(begin(), pos)); }

void features::truncate_to(const audit_iterator& pos) { truncate_to(std::distance(audit_begin(), pos)); }

void features::truncate_to(size_t i)
{
  values.resize_but_with_stl_behavior(i);
  if (indicies.end() != indicies.begin()) { indicies.resize_but_with_stl_behavior(i); }

  if (space_names.size() > i) { space_names.erase(space_names.begin() + i, space_names.end()); }
}

void features::concat(const features& other)
{
  if (other.empty()) { return; }

  // Conditions to check:
  //  - !empty() && audit && other.audit -> push val, idx, audit
  //  - !empty() && audit && !other.audit -> fail
  //  - !empty() && !audit && other.audit -> fail
  //  - !empty() && !audit && !other.audit -> push val, idx
  //  - empty() && other.audit -> push val, idx, audit
  //  - empty() && !other.audit -> push val, idx

  if (!empty() && (space_names.empty() != other.space_names.empty()))
  { THROW_OR_RETURN_VOID("Cannot merge two feature groups if one has audit info and the other does not."); }
  values.insert(values.end(), other.values.begin(), other.values.end());
  indicies.insert(indicies.end(), other.indicies.begin(), other.indicies.end());

  if (!other.space_names.empty())
  { space_names.insert(space_names.end(), other.space_names.begin(), other.space_names.end()); }
}

void features::push_back(feature_value v, feature_index i)
{
  values.push_back(v);
  indicies.push_back(i);
  sum_feat_sq += v * v;
}

bool features::sort(uint64_t parse_mask)
{
  if (indicies.empty()) { return false; }

  if (!space_names.empty())
  {
    std::vector<feature_slice> slice;
    slice.reserve(indicies.size());
    for (size_t i = 0; i < indicies.size(); i++)
    { slice.push_back({values[i], indicies[i] & parse_mask, space_names[i]}); }
    // The comparator should return true if the first element is less than the second.
    std::sort(slice.begin(), slice.end(), [](const feature_slice& first, const feature_slice& second) {
      return (first.weight_index < second.weight_index) ||
          ((first.weight_index == second.weight_index) && (first.x < second.x));
    });

    for (size_t i = 0; i < slice.size(); i++)
    {
      values[i] = slice[i].x;
      indicies[i] = slice[i].weight_index;
      space_names[i] = slice[i].space_name;
    }
  }
  else
  {
    std::vector<feature> slice;
    slice.reserve(indicies.size());

    for (size_t i = 0; i < indicies.size(); i++) { slice.emplace_back(values[i], indicies[i] & parse_mask); }
    // The comparator should return true if the first element is less than the second.
    std::sort(slice.begin(), slice.end(), [](const feature& first, const feature& second) {
      return (first.weight_index < second.weight_index) ||
          ((first.weight_index == second.weight_index) && (first.x < second.x));
    });
    for (size_t i = 0; i < slice.size(); i++)
    {
      values[i] = slice[i].x;
      indicies[i] = slice[i].weight_index;
    }
  }
  return true;
}

void features::deep_copy_from(const features& src)
{
  values = src.values;
  indicies = src.indicies;
  space_names = src.space_names;
  sum_feat_sq = src.sum_feat_sq;
}
