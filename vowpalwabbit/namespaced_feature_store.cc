// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "namespaced_feature_store.h"
#include "vw_exception.h"

using namespace VW;

// If no feature group already exists a default one will be created.
// Creating new feature groups will invalidate any pointers or references held.
features& namespaced_feature_store::get_or_create(namespace_index ns_index, uint64_t hash)
{
  auto* existing_group = get_or_null(ns_index, hash);
  if (existing_group == nullptr)
  {
    if (!_saved_feature_group_nodes.empty())
    {
      _feature_groups[ns_index].splice(
          _feature_groups[ns_index].end(), _saved_feature_group_nodes, _saved_feature_group_nodes.begin());
      _feature_groups[ns_index].back().hash = hash;
      _feature_groups[ns_index].back().index = ns_index;
    }
    else
    {
      _feature_groups[ns_index].emplace_back(features{}, hash, ns_index);
    }

    if (_feature_groups[ns_index].size() == 1) { _legacy_indices_existing.push_back(ns_index); }
    return _feature_groups[ns_index].back().feats;
  }

  return *existing_group;
}

void namespaced_feature_store::remove(namespace_index ns_index, uint64_t hash)
{
  auto& bucket = _feature_groups[ns_index];
  auto it = std::find_if(
      bucket.begin(), bucket.end(), [hash](const namespaced_features& group) { return group.hash == hash; });
  if (it == bucket.end()) { return; }
  it->feats.clear();
  _saved_feature_group_nodes.splice(_saved_feature_group_nodes.end(), _feature_groups[ns_index], it);
  if (bucket.empty())
  {
    auto idx_it = std::find(_legacy_indices_existing.begin(), _legacy_indices_existing.end(), ns_index);
    if (idx_it != _legacy_indices_existing.end()) { _legacy_indices_existing.erase(idx_it); }
  }
}

void namespaced_feature_store::clear()
{
  for (auto& ns_index : _legacy_indices_existing)
  {
    for (auto& namespaced_feat_group : _feature_groups[ns_index]) { namespaced_feat_group.feats.clear(); }
    _saved_feature_group_nodes.splice(_saved_feature_group_nodes.end(), _feature_groups[ns_index]);
  }
  _legacy_indices_existing.clear();
}

VW::chained_proxy_iterator<namespaced_feature_store::list_iterator, features::audit_iterator>
namespaced_feature_store::index_flat_begin(namespace_index ns_index)
{
  auto begin_it = _feature_groups[ns_index].begin();
  auto end_it = _feature_groups[ns_index].end();
  features::audit_iterator inner_it;
  // If the range is empty we must default construct the inner iterator as dereferencing an end pointer (What begin_it
  // is here) is not valid.
  if (begin_it == end_it) { inner_it = features::audit_iterator{}; }
  else
  {
    --end_it;
    inner_it = begin_it->feats.audit_begin();
  }
  // end_it always points to the last valid outer iterator instead of the actual end iterator of the outer collection.
  // This is because the end chained_proxy_iterator points to the end iterator of the last valid item of the outer
  // collection.
  return {begin_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_feature_store::list_iterator, features::audit_iterator>
namespaced_feature_store::index_flat_end(namespace_index ns_index)
{
  auto begin_it = _feature_groups[ns_index].begin();
  auto end_it = _feature_groups[ns_index].end();
  features::audit_iterator inner_it;
  if (begin_it == end_it) { inner_it = features::audit_iterator{}; }
  else
  {
    --end_it;
    inner_it = end_it->feats.audit_end();
  }

  return {end_it, end_it, inner_it};
}
