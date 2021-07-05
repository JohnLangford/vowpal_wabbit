// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <set>
#include <map>

#include "feature_group.h"
#include "generic_range.h"
#include "chained_proxy_iterator.h"
#include "object_pool.h"

typedef unsigned char namespace_index;



namespace VW
{
struct namespaced_features;

class quick_iterator_t
{
  features* _features;
  namespace_index* _indices;

public:
  quick_iterator_t(features* in_features, namespace_index* indices)
      : _features(in_features), _indices(indices)
  {
  }

  inline features& operator*() { return *_features; }
  inline namespace_index index() { return *_indices; }
  inline quick_iterator_t& operator++()
  {
    _features++;
    _indices++;
    return *this;
  }

  bool operator==(const quick_iterator_t& rhs) { return _features == rhs._features; }
  bool operator!=(const quick_iterator_t& rhs) { return _features != rhs._features; }
};


class quickest_iterator_t
{
  features* _features;

public:
  quickest_iterator_t(features* in_features)
      : _features(in_features)
  {
  }

  inline features& operator*() { return *_features; }
  inline quickest_iterator_t& operator++()
  {
    _features++;
    return *this;
  }

  bool operator==(const quickest_iterator_t& rhs) { return _features == rhs._features; }
  bool operator!=(const quickest_iterator_t& rhs) { return _features != rhs._features; }
};

/// Insertion or removal will result in this value in invalidated.
template <typename FeaturesT, typename IndexT, typename HashT, typename NamespaceFeaturesT>
class iterator_t
{
  size_t _index;
  NamespaceFeaturesT* _ns_features;

public:
  iterator_t(size_t index, NamespaceFeaturesT* ns_features)
      : _index(index)
      , _ns_features(ns_features)
  {
  }

  inline FeaturesT& operator*() { return _ns_features->_feature_groups[_index]; }
  inline iterator_t& operator++()
  {
    _index++;
    return *this;
  }

  inline iterator_t& operator+=(int i)
  {
    _index += i;
    return *this;
  }

  inline IndexT index() { return _ns_features->_namespace_indices[_index]; }
  inline HashT hash() { return _ns_features->_namespace_hashes[_index]; }

  bool operator==(const iterator_t& rhs) { return _index == rhs._index; }
  bool operator!=(const iterator_t& rhs) { return _index != rhs._index; }
};

/// Insertion or removal will result in this value in invalidated.
template <typename IndiceItT, typename FeaturesT, typename IndexT, typename HashT>
class indexed_iterator_t
{
  IndiceItT _indices_it;
  FeaturesT* _feature_groups;
  IndexT* _namespace_indices;
  HashT* _namespace_hashes;

public:
  using difference_type = std::ptrdiff_t;

  indexed_iterator_t(IndiceItT indices, FeaturesT* feature_groups, IndexT* namespace_indices, HashT* namespace_hashes)
      : _indices_it(indices)
      , _feature_groups(feature_groups)
      , _namespace_indices(namespace_indices)
      , _namespace_hashes(namespace_hashes)
  {
  }
  FeaturesT& operator*()
  {
    return _feature_groups[*_indices_it];
  }
  indexed_iterator_t& operator++()
  {
    _indices_it++;
    return *this;
  }

  indexed_iterator_t& operator--()
  {
    --_indices_it;
    return *this;
  }

  IndexT index()
  {
    return _namespace_indices[*_indices_it];
  }
  HashT hash()
  {
    return _namespace_hashes[*_indices_it];
  }

  friend bool operator<(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    return lhs._indices_it < rhs._indices_it;
  }

  friend bool operator>(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    return lhs._indices_it > rhs._indices_it;
  }

  friend bool operator<=(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs) { return !(lhs > rhs); }
  friend bool operator>=(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs) { return !(lhs < rhs); }

  friend difference_type operator-(const indexed_iterator_t& lhs, const indexed_iterator_t& rhs)
  {
    assert(lhs._indices_it >= rhs._indices_it);
    return lhs._indices_it - rhs._indices_it;
  }

  bool operator==(const indexed_iterator_t& rhs) const { return _indices_it == rhs._indices_it; }
  bool operator!=(const indexed_iterator_t& rhs) const { return _indices_it != rhs._indices_it; }
};

/// namespace_index - 1 byte namespace identifier. Either the first character of the namespace or a reserved namespace
/// identifier namespace_hash - 8 byte hash
struct namespaced_features
{
  using iterator = iterator_t<features, namespace_index, uint64_t, namespaced_features>;
  using const_iterator = iterator_t<const features, const namespace_index, const uint64_t, const namespaced_features>;
  using indexed_iterator = indexed_iterator_t<std::vector<size_t>::iterator, features, namespace_index, uint64_t>;
  using const_indexed_iterator =
      indexed_iterator_t<std::vector<size_t>::const_iterator, const features, const namespace_index, const uint64_t>;

  template <typename A, typename B, typename C, typename D> friend class iterator_t;

  namespaced_features() = default;
  ~namespaced_features() = default;

  // For copies we deep copy the features but not the pool
  namespaced_features(const namespaced_features& other)
  {
    _feature_groups = other._feature_groups;
    _namespace_indices = other._namespace_indices;
    _namespace_hashes = other._namespace_hashes;
    _legacy_indices_to_index_mapping = other._legacy_indices_to_index_mapping;
  }
  namespaced_features& operator=(const namespaced_features& other)
  {
    _feature_groups = other._feature_groups;
    _namespace_indices = other._namespace_indices;
    _namespace_hashes = other._namespace_hashes;
    _legacy_indices_to_index_mapping = other._legacy_indices_to_index_mapping;
    return *this;
  }
  namespaced_features(namespaced_features&& other) = default;
  namespaced_features& operator=(namespaced_features&& other) = default;

  inline size_t size() const { return _feature_groups.size(); }
  inline bool empty() const { return _feature_groups.empty(); }

  // Returns nullptr if not found.
  inline features* get_feature_group(uint64_t hash)
  {
    auto it = std::find(_namespace_hashes.begin(), _namespace_hashes.end(), hash);
    if (it == _namespace_hashes.end()) { return nullptr; }
    auto existing_index = std::distance(_namespace_hashes.begin(), it);

    return &_feature_groups[existing_index];
  }
  // Returns nullptr if not found.
  inline const features* get_feature_group(uint64_t hash) const
  {
    auto it = std::find(_namespace_hashes.begin(), _namespace_hashes.end(), hash);
    if (it == _namespace_hashes.end()) { return nullptr; }
    auto existing_index = std::distance(_namespace_hashes.begin(), it);

    return &_feature_groups[existing_index];
  }

  // Wil contains duplicates if there exists more than one feature group per index.
  const std::vector<namespace_index>& get_indices() const;
  const uint64_t& get_last_hash() const;

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
  inline features& get_or_create_feature_group(uint64_t hash, namespace_index ns_index)
  {
    auto* existing_group = get_feature_group(hash);
    if (existing_group == nullptr)
    {
      _feature_groups.push_back(features{});
      _saved_feature_groups.acquire_object(_feature_groups.back());
      _namespace_indices.push_back(ns_index);
      _namespace_hashes.push_back(hash);
      auto new_index = _feature_groups.size() - 1;
      _legacy_indices_to_index_mapping[ns_index].push_back(new_index);
      existing_group = &_feature_groups.back();
    }

    return *existing_group;
  }

  // This operation is only allowed in code that allows exceptions.
  // get_feature_group should be used instead for noexcept code
#ifndef VW_NOEXCEPT
  // These will throw if the hash does not exist
  inline const features& operator[](uint64_t hash) const
  {
    auto* existing_group = get_feature_group(hash);
    assert(existing_group != nullptr);
    return *existing_group;
  }
  inline features& operator[](uint64_t hash)
  {
    auto* existing_group = get_feature_group(hash);
    assert(existing_group != nullptr);
    return *existing_group;
  }
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

  inline quick_iterator_t quick_begin()
{
  return {_feature_groups.data(), _namespace_indices.data()};
}
inline quick_iterator_t quick_end()
{
  return {_feature_groups.data() + _feature_groups.size(), _namespace_indices.data()};
}

  inline quickest_iterator_t quickest_begin()
{
  return {_feature_groups.data()};
}
inline quickest_iterator_t quickest_end()
{
  return {_feature_groups.data() + _feature_groups.size()};
}

inline iterator begin()
{
  return {0, this};
}
inline iterator end()
{
  return {_feature_groups.size(), this};
}
inline const_iterator begin() const
{
  return {0, this};
}
inline const_iterator end() const
{
  return {_feature_groups.size(), this};
}
inline const_iterator cbegin() const
{
  return {0, this};
}
inline const_iterator cend() const
{
  return {_feature_groups.size(), this};
}


private:
  std::vector<features> _feature_groups;
  // Can have duplicate values.
  std::vector<namespace_index> _namespace_indices;
  // Should never have duplicate values.
  std::vector<uint64_t> _namespace_hashes;

  std::array<std::vector<size_t>, 256> _legacy_indices_to_index_mapping;
  VW::moved_object_pool<features> _saved_feature_groups;
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
    _legacy_indices_to_index_mapping[ns_index].push_back(new_index);
    existing_group = &_feature_groups.back();
  }
  else
  {
    existing_group->concat(std::forward<FeaturesT>(ftrs));
    auto it = std::find(_namespace_hashes.begin(), _namespace_hashes.end(), hash);
    auto existing_index = std::distance(_namespace_hashes.begin(), it);
    // Should we ensure that this doesnt already exist under a DIFFERENT namespace_index?
    // However, his shouldn't be possible as ns_index depends on hash.
    auto& ns_indices_list = _legacy_indices_to_index_mapping[ns_index];
    if (std::find(ns_indices_list.begin(), ns_indices_list.end(), ns_index) == ns_indices_list.end())
    { ns_indices_list.push_back(existing_index); }
  }
  return *existing_group;
}

}  // namespace VW
