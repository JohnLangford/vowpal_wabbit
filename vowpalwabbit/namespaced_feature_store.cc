// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "namespaced_feature_store.h"
#include "vw_exception.h"

using namespace VW;

features* namespaced_features::get_feature_group(uint64_t hash)
{
  auto it = _hash_to_index_mapping.find(hash);
  if (it == _hash_to_index_mapping.end()) { return nullptr; }

  return &_feature_groups[it->second];
}

const features* namespaced_features::get_feature_group(uint64_t hash) const
{
  auto it = _hash_to_index_mapping.find(hash);
  if (it == _hash_to_index_mapping.end()) { return nullptr; }

  return &_feature_groups[it->second];
}

const std::set<namespace_index>& namespaced_features::get_indices() const { return _contained_indices; }

namespace_index namespaced_features::get_index_for_hash(uint64_t hash) const
{
  auto it = _hash_to_index_mapping.find(hash);
  if (it == _hash_to_index_mapping.end())
  {
#ifdef VW_NOEXCEPT
    return {};
#else
    THROW("No index found for hash: " << hash);
#endif
  }
  return _namespace_indices[it->second];
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

features& namespaced_features::get_or_create_feature_group(uint64_t hash, namespace_index ns_index)
{
  auto* existing_group = get_feature_group(hash);
  if (existing_group == nullptr)
  {
    _feature_groups.emplace_back();
    _namespace_indices.push_back(ns_index);
    _namespace_hashes.push_back(hash);
    auto new_index = _feature_groups.size() - 1;
    _hash_to_index_mapping[hash] = new_index;
    _legacy_indices_to_index_mapping[ns_index].push_back(new_index);
    // If size is 1, that means this is the first time the ns_index is added and we should add it to the set.
    if (_legacy_indices_to_index_mapping[ns_index].size() == 1) { _contained_indices.insert(ns_index); }
    existing_group = &_feature_groups.back();
  }

  return *existing_group;
}

// This operation is only allowed in code that allows exceptions.
// get_feature_group should be used instead for noexcept code
#ifndef VW_NOEXCEPT
const features& namespaced_features::operator[](uint64_t hash) const
{
  auto* existing_group = get_feature_group(hash);
  if (existing_group == nullptr) { THROW("No group found for hash: " << hash); }
  return *existing_group;
}
#endif

#ifndef VW_NOEXCEPT
features& namespaced_features::operator[](uint64_t hash)
{
  auto* existing_group = get_feature_group(hash);
  if (existing_group == nullptr) { THROW("No group found for hash: " << hash); }
  return *existing_group;
}
#endif

void namespaced_features::remove_feature_group(uint64_t hash)
{
  if (_hash_to_index_mapping.count(hash) == 0) { return; }
  auto existing_index = _hash_to_index_mapping[hash];

  // Remove item from each vector at this index.
  _feature_groups.erase(_feature_groups.begin() + existing_index);
  _namespace_indices.erase(_namespace_indices.begin() + existing_index);
  _namespace_hashes.erase(_namespace_hashes.begin() + existing_index);
  _hash_to_index_mapping.erase(hash);

  for (auto& kv : _legacy_indices_to_index_mapping)
  {
    auto& index_vec = kv.second;
    // Remove this index from ns_index mappings if it exists
    auto it = std::find(index_vec.begin(), index_vec.end(), existing_index);
    if (it != index_vec.end()) { index_vec.erase(it); }

    // Shift down any index that came after this one.
    for (auto& idx : index_vec)
    {
      if (idx > existing_index) { idx -= 1; }
    }
  }

  // If any groups are left empty, remove them.
  for (auto it = _legacy_indices_to_index_mapping.begin(); it != _legacy_indices_to_index_mapping.end();)
  {
    if (it->second.empty())
    {
      // There are no more feature groups which correspond to this index.
      _contained_indices.erase(it->first);
      it = _legacy_indices_to_index_mapping.erase(it);
    }
    else
    {
      ++it;
    }
  }

  for (auto& kv : _hash_to_index_mapping)
  {
    if (kv.second > existing_index) { kv.second -= 1; }
  }
}

void namespaced_features::clear()
{
  _feature_groups.clear();
  _namespace_indices.clear();
  _namespace_hashes.clear();
  _legacy_indices_to_index_mapping.clear();
  _hash_to_index_mapping.clear();
  _contained_indices.clear();
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
namespaced_features::namespace_index_cend_proxy(namespace_index ns_index) const
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

namespaced_features::indexed_iterator namespaced_features::namespace_index_begin(namespace_index ns_index)
{
  auto it = _legacy_indices_to_index_mapping.find(ns_index);
  if (it == _legacy_indices_to_index_mapping.end())
  { return {nullptr, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()}; }
  return {it->second.data(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}

namespaced_features::indexed_iterator namespaced_features::namespace_index_end(namespace_index ns_index)
{
  auto it = _legacy_indices_to_index_mapping.find(ns_index);
  if (it == _legacy_indices_to_index_mapping.end())
  { return {nullptr, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()}; }
  return {it->second.data() + it->second.size(), _feature_groups.data(), _namespace_indices.data(),
      _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_begin(namespace_index ns_index) const
{
  auto it = _legacy_indices_to_index_mapping.find(ns_index);
  if (it == _legacy_indices_to_index_mapping.end())
  { return {nullptr, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()}; }
  return {it->second.data(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_end(namespace_index ns_index) const
{
  auto it = _legacy_indices_to_index_mapping.find(ns_index);
  if (it == _legacy_indices_to_index_mapping.end())
  { return {nullptr, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()}; }
  return {it->second.data() + it->second.size(), _feature_groups.data(), _namespace_indices.data(),
      _namespace_hashes.data()};
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_cbegin(namespace_index ns_index) const
{
  return namespace_index_begin(ns_index);
}

namespaced_features::const_indexed_iterator namespaced_features::namespace_index_cend(namespace_index ns_index) const
{
  return namespace_index_end(ns_index);
}

namespaced_features::iterator namespaced_features::begin()
{
  return {0, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
namespaced_features::iterator namespaced_features::end()
{
  return {_feature_groups.size(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
namespaced_features::const_iterator namespaced_features::begin() const
{
  return {0, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
namespaced_features::const_iterator namespaced_features::end() const
{
  return {_feature_groups.size(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
namespaced_features::const_iterator namespaced_features::cbegin() const
{
  return {0, _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
namespaced_features::const_iterator namespaced_features::cend() const
{
  return {_feature_groups.size(), _feature_groups.data(), _namespace_indices.data(), _namespace_hashes.data()};
}
