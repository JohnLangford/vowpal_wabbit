// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <set>

#include "feature_group.h"
#include "generic_range.h"
#include "chained_proxy_iterator.h"

typedef unsigned char namespace_index;

namespace VW
{
/// Insertion or removal will result in this value in invalidated.
template <typename FeaturesT, typename IndexT, typename HashT>
class iterator_t
{
  size_t _index;
  FeaturesT* _feature_groups;
  IndexT* _namespace_indices;
  HashT* _namespace_hashes;

public:
  iterator_t(size_t index, FeaturesT* feature_groups, IndexT* namespace_indices, HashT* namespace_hashes)
      : _index(index)
      , _feature_groups(feature_groups)
      , _namespace_indices(namespace_indices)
      , _namespace_hashes(namespace_hashes)
  {
  }
  FeaturesT& operator*() { return _feature_groups[_index]; }
  iterator_t& operator++()
  {
    ++_index;
    return *this;
  }

  IndexT index() { return _namespace_indices[_index]; }
  HashT hash() { return _namespace_hashes[_index]; }

  bool operator==(const iterator_t& rhs) { return _index == rhs._index; }
  bool operator!=(const iterator_t& rhs) { return _index != rhs._index; }
};

/// Insertion or removal will result in this value in invalidated.
template <typename IndicesT, typename FeaturesT, typename IndexT, typename HashT>
class indexed_iterator_t
{
  IndicesT* _indices;
  FeaturesT* _feature_groups;
  IndexT* _namespace_indices;
  HashT* _namespace_hashes;

public:
  using difference_type = std::ptrdiff_t;

  indexed_iterator_t(IndicesT* indices, FeaturesT* feature_groups, IndexT* namespace_indices, HashT* namespace_hashes)
      : _indices(indices)
      , _feature_groups(feature_groups)
      , _namespace_indices(namespace_indices)
      , _namespace_hashes(namespace_hashes)
  {
  }
  FeaturesT& operator*()
  {
#ifndef VW_NOEXCEPT
    if (_indices == nullptr) { THROW("Invalid iterator"); }
#endif
    return _feature_groups[*_indices];
  }
  indexed_iterator_t& operator++()
  {
    if (_indices != nullptr) { ++_indices; }
    return *this;
  }

  indexed_iterator_t& operator--()
  {
    if (_indices != nullptr) { --_indices; }
    return *this;
  }

  IndexT index()
  {
#ifndef VW_NOEXCEPT
    if (_indices == nullptr) { THROW("Invalid iterator"); }
#endif
    return _namespace_indices[*_indices];
  }
  HashT hash()
  {
#ifndef VW_NOEXCEPT
    if (_indices == nullptr) { THROW("Invalid iterator"); }
#endif
    return _namespace_hashes[*_indices];
  }

  friend bool operator<(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    return lhs._indices < rhs._indices;
  }

  friend bool operator>(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    return lhs._indices > rhs._indices;
  }

  friend bool operator<=(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs) { return !(lhs > rhs); }
  friend bool operator>=(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs) { return !(lhs < rhs); }

  friend difference_type operator-(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    assert(lhs._indices >= rhs._indices);
    return lhs._indices - rhs._indices;
  }

  bool operator==(const indexed_iterator_t& rhs) const { return _indices == rhs._indices; }
  bool operator!=(const indexed_iterator_t& rhs) const { return _indices != rhs._indices; }
};

/// namespace_index - 1 byte namespace identifier. Either the first character of the namespace or a reserved namespace
/// identifier namespace_hash - 8 byte hash
struct namespaced_features
{
  using iterator = iterator_t<features, namespace_index, uint64_t>;
  using const_iterator = iterator_t<const features, const namespace_index, const uint64_t>;
  using indexed_iterator = indexed_iterator_t<size_t, features, namespace_index, uint64_t>;
  using const_indexed_iterator =
      indexed_iterator_t<const size_t, const features, const namespace_index, const uint64_t>;

  namespaced_features() = default;
  ~namespaced_features() = default;
  namespaced_features(const namespaced_features&) = default;
  namespaced_features& operator=(const namespaced_features&) = default;
  namespaced_features(namespaced_features&& other) = default;
  namespaced_features& operator=(namespaced_features&& other) = default;

  inline size_t size() const { return _feature_groups.size(); }
  inline bool empty() const { return _feature_groups.empty(); }

  // Returns nullptr if not found.
  features* get_feature_group(uint64_t hash);
  // Returns nullptr if not found.
  const features* get_feature_group(uint64_t hash) const;

  const std::set<namespace_index>& get_indices() const;
  namespace_index get_index_for_hash(uint64_t hash) const;

  // The following are experimental and may be superseded with namespace_index_begin_proxy
  // Returns empty range if not found
  std::pair<indexed_iterator, indexed_iterator> get_namespace_index_groups(namespace_index ns_index);
  // Returns empty range if not found
  std::pair<const_indexed_iterator, const_indexed_iterator> get_namespace_index_groups(namespace_index ns_index) const;

  // If a feature group already exists in this "slot" it will be merged
  template <typename FeaturesT>
  features& merge_feature_group(FeaturesT&& ftrs, uint64_t hash, namespace_index ns_index);

  // If no feature group already exists a default one will be created.
  // Creating new feature groups will invalidate any pointers or references held.
  features& get_or_create_feature_group(uint64_t hash, namespace_index ns_index);

#ifndef VW_NOEXCEPT
  // These will throw if the hash does not exist
  const features& operator[](uint64_t hash) const;
  features& operator[](uint64_t hash);
#endif

  // Removing a feature group will invalidate any pointers or references held.
  void remove_feature_group(uint64_t hash);

  void clear();

  // Experimental, hence the cumbersome names.
  // These iterators allow you to iterate over an entire namespace index as if it were a single feature group.
  VW::chained_proxy_iterator<indexed_iterator, features::audit_iterator> namespace_index_begin_proxy(
      namespace_index ns_index);
  VW::chained_proxy_iterator<indexed_iterator, features::audit_iterator> namespace_index_end_proxy(
      namespace_index ns_index);
  VW::chained_proxy_iterator<const_indexed_iterator, features::const_audit_iterator> namespace_index_begin_proxy(
      namespace_index ns_index) const;
  VW::chained_proxy_iterator<const_indexed_iterator, features::const_audit_iterator> namespace_index_end_proxy(
      namespace_index ns_index) const;
  VW::chained_proxy_iterator<const_indexed_iterator, features::const_audit_iterator> namespace_index_cbegin_proxy(
      namespace_index ns_index) const;
  VW::chained_proxy_iterator<const_indexed_iterator, features::const_audit_iterator> namespace_index_cend_proxy(
      namespace_index ns_index) const;

  // All of the following are experimental and may be superseded with the above proxies.
  generic_range<indexed_iterator> namespace_index_range(namespace_index ns_index);
  generic_range<const_indexed_iterator> namespace_index_range(namespace_index ns_index) const;
  indexed_iterator namespace_index_begin(namespace_index ns_index);
  indexed_iterator namespace_index_end(namespace_index ns_index);
  const_indexed_iterator namespace_index_begin(namespace_index ns_index) const;
  const_indexed_iterator namespace_index_end(namespace_index ns_index) const;
  const_indexed_iterator namespace_index_cbegin(namespace_index ns_index) const;
  const_indexed_iterator namespace_index_cend(namespace_index ns_index) const;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  const_iterator cbegin() const;
  const_iterator cend() const;

private:
  std::vector<features> _feature_groups;
  // Can have duplicate values.
  std::vector<namespace_index> _namespace_indices;
  // Should never have duplicate values.
  std::vector<uint64_t> _namespace_hashes;

  std::unordered_map<namespace_index, std::vector<size_t>> _legacy_indices_to_index_mapping;
  std::unordered_map<uint64_t, size_t> _hash_to_index_mapping;
  std::set<namespace_index> _contained_indices;
};

// If a feature group already exists in this "slot" it will be merged
// Creating new feature groups will invalidate any pointers or references held.
template <typename FeaturesT>
features& namespaced_features::merge_feature_group(FeaturesT&& ftrs, uint64_t hash, namespace_index ns_index)
{
  auto* existing_group = get_feature_group(hash);
  if (existing_group == nullptr)
  {
    _feature_groups.emplace_back(std::forward<FeaturesT>(ftrs));
    _namespace_indices.push_back(ns_index);
    _namespace_hashes.push_back(hash);
    auto new_index = _feature_groups.size() - 1;
    _hash_to_index_mapping[hash] = new_index;
    _legacy_indices_to_index_mapping[ns_index].push_back(new_index);
    // If size is 1, that means this is the first time the ns_index is added and we should add it to the set.
    if (_legacy_indices_to_index_mapping[ns_index].size() == 1) { _contained_indices.insert(ns_index); }
    existing_group = &_feature_groups.back();
  }
  else
  {
    existing_group->concat(std::forward<FeaturesT>(ftrs));
    auto existing_index = _hash_to_index_mapping[hash];
    // Should we ensure that this doesnt already exist under a DIFFERENT namespace_index?
    // However, his shouldn't be possible as ns_index depends on hash.
    auto& ns_indices_list = _legacy_indices_to_index_mapping[ns_index];
    if (std::find(ns_indices_list.begin(), ns_indices_list.end(), ns_index) == ns_indices_list.end())
    { ns_indices_list.push_back(existing_index); }
  }
  return *existing_group;
}

}  // namespace VW
