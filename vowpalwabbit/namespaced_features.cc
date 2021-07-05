// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "namespaced_features.h"
#include "vw_exception.h"

using namespace VW;

const std::vector<namespace_index>& namespaced_features::get_indices() const { return _namespace_indices; }

const uint64_t& namespaced_features::get_last_hash() const { return _namespace_hashes.back(); }

namespace_index namespaced_features::get_index_for_hash(uint64_t hash) const
{
  auto it = std::find(_namespace_hashes.begin(), _namespace_hashes.end(), hash);
  if (it == _namespace_hashes.end())
  {
#ifdef VW_NOEXCEPT
    return {};
#else
    THROW("No index found for hash: " << hash);
#endif
  }
  auto existing_index = std::distance(_namespace_hashes.begin(), it);
  return _namespace_indices[existing_index];
}

std::pair<namespaced_features::indexed_iterator, namespaced_features::indexed_iterator>
namespaced_features::get_namespace_index_groups(namespace_index ns_index)
{
  return std::make_pair(namespace_index_begin(ns_index), namespace_index_end(ns_index));
}

std::pair<namespaced_features::const_indexed_iterator, namespaced_features::const_indexed_iterator>
namespaced_features::get_namespace_index_groups(namespace_index ns_index) const
{
  return std::make_pair(namespace_index_begin(ns_index), namespace_index_end(ns_index));
}


void namespaced_features::remove_feature_group(uint64_t hash)
{
  auto it = std::find(_namespace_hashes.begin(), _namespace_hashes.end(), hash);
  if (it == _namespace_hashes.end()) { return; }
  auto existing_index = std::distance(_namespace_hashes.begin(), it);

  auto& obj_to_remove = _feature_groups[existing_index];
  obj_to_remove.clear();
  _saved_feature_groups.reclaim_object(std::move(obj_to_remove));

  // Remove item from each vector at this index.
  _feature_groups.erase(_feature_groups.begin() + existing_index);
  _namespace_indices.erase(_namespace_indices.begin() + existing_index);
  _namespace_hashes.erase(_namespace_hashes.begin() + existing_index);

  for (auto& index_vec : _legacy_indices_to_index_mapping)
  {
    if (!index_vec.empty())
    {
      // Remove this index from ns_index mappings if it exists
      auto it = std::find(index_vec.begin(), index_vec.end(), existing_index);
      if (it != index_vec.end()) { index_vec.erase(it); }

      // Shift down any index that came after this one.
      for (auto& idx : index_vec)
      {
        if (idx > existing_index) { idx -= 1; }
      }
    }
  }
}

void namespaced_features::clear()
{
  for (auto& feat_group : _feature_groups)
  {
    feat_group.clear();
    _saved_feature_groups.reclaim_object(std::move(feat_group));
  }
  _feature_groups.clear();
  _namespace_indices.clear();
  _namespace_hashes.clear();
  for (auto& index_vec : _legacy_indices_to_index_mapping) { index_vec.clear(); }
}

generic_range<namespaced_features::indexed_iterator> namespaced_features::namespace_index_range(
    namespace_index ns_index)
{
  return {namespace_index_begin(ns_index), namespace_index_end(ns_index)};
}

generic_range<namespaced_features::const_indexed_iterator> namespaced_features::namespace_index_range(
    namespace_index ns_index) const
{
  return {namespace_index_begin(ns_index), namespace_index_end(ns_index)};
}

VW::chained_proxy_iterator<namespaced_features::indexed_iterator, features::audit_iterator>
namespaced_features::namespace_index_begin_proxy(namespace_index ns_index)
{
  auto begin_it = namespace_index_begin(ns_index);
  auto end_it = namespace_index_end(ns_index);
  features::audit_iterator inner_it;
  // If the range is empty we must default construct the inner iterator as dereferencing an end pointer (What begin_it
  // is here) is not valid.
  if (begin_it == end_it)
  {
    --end_it;
    inner_it = features::audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*begin_it).audit_begin();
  }
  // end_it always points to the last valid outer iterator instead of the actual end iterator of the outer collection.
  // This is because the end chained_proxy_iterator points to the end iterator of the last valid item of the outer
  // collection.
  return {begin_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_features::indexed_iterator, features::audit_iterator>
namespaced_features::namespace_index_end_proxy(namespace_index ns_index)
{
  auto begin_it = namespace_index_begin(ns_index);
  auto end_it = namespace_index_end(ns_index);
  features::audit_iterator inner_it;
  if (begin_it == end_it)
  {
    --end_it;
    inner_it = features::audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*end_it).audit_end();
  }

  return {end_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_features::const_indexed_iterator, features::const_audit_iterator>
namespaced_features::namespace_index_begin_proxy(namespace_index ns_index) const
{
  auto begin_it = namespace_index_cbegin(ns_index);
  auto end_it = namespace_index_cend(ns_index);
  features::const_audit_iterator inner_it;
  if (begin_it == end_it)
  {
    --end_it;
    inner_it = features::const_audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*begin_it).audit_cbegin();
  }
  return {begin_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_features::const_indexed_iterator, features::const_audit_iterator>
namespaced_features::namespace_index_end_proxy(namespace_index ns_index) const
{
  auto begin_it = namespace_index_cbegin(ns_index);
  auto end_it = namespace_index_cend(ns_index);
  features::const_audit_iterator inner_it;
  if (begin_it == end_it)
  {
    --end_it;
    inner_it = features::const_audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*end_it).audit_cend();
  }
  return {end_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_features::const_indexed_iterator, features::const_audit_iterator>
namespaced_features::namespace_index_cbegin_proxy(namespace_index ns_index) const
{
  auto begin_it = namespace_index_cbegin(ns_index);
  auto end_it = namespace_index_cend(ns_index);

  features::const_audit_iterator inner_it;
  if (begin_it == end_it)
  {
    inner_it = features::const_audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*begin_it).audit_cbegin();
  }

  return {begin_it, end_it, inner_it};
}

VW::chained_proxy_iterator<namespaced_features::const_indexed_iterator, features::const_audit_iterator>
namespaced_features::namespace_index_cend_proxy(namespace_index ns_index) const
{
  auto begin_it = namespace_index_cbegin(ns_index);
  auto end_it = namespace_index_cend(ns_index);
  features::const_audit_iterator inner_it;
  if (begin_it == end_it)
  {
    inner_it = features::const_audit_iterator{};
  }
  else
  {
    --end_it;
    inner_it = (*end_it).audit_cend();
  }
  return {end_it, end_it, inner_it};
}

namespaced_features::indexed_iterator namespaced_features::namespace_index_begin(namespace_index ns_index)
{
  auto& index_vec = _legacy_indices_to_index_mapping[ns_index];
  return {index_vec.begin(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}

namespaced_features::indexed_iterator namespaced_features::namespace_index_end(namespace_index ns_index)
{
  auto& index_vec = _legacy_indices_to_index_mapping[ns_index];
  return {index_vec.end(), _feature_groups.data(), _namespace_indices.data(),
      _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_begin(namespace_index ns_index) const
{
  auto& index_vec = _legacy_indices_to_index_mapping[ns_index];
  return {index_vec.begin(), const_cast<const features*>(_feature_groups.data()), _namespace_indices.data(),
      _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_end(namespace_index ns_index) const
{
  auto& index_vec = _legacy_indices_to_index_mapping[ns_index];
  return {index_vec.end(), const_cast<const features*>(_feature_groups.data()),
      _namespace_indices.data(), _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_cbegin(namespace_index ns_index) const
{
  return namespace_index_begin(ns_index);
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_cend(namespace_index ns_index) const
{
  return namespace_index_end(ns_index);
}
