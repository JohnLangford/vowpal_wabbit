// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "label_dictionary.h"

#include "reductions.h"
#include "cost_sensitive.h"

namespace LabelDict
{
void add_example_namespace(
    example& ec, namespace_index ns_index, uint64_t ns_hash, const features& source_feature_group)
{
  features& destination_feat_group = ec.feature_space.get_or_create(ns_index, ns_hash);
  destination_feat_group.concat(source_feature_group);
  ec.reset_total_sum_feat_sq();
  ec.num_features += source_feature_group.size();
}

void del_example_namespace(
    example& ec, namespace_index ns_index, uint64_t ns_hash, const features& source_feature_group)
{
  // print_update is called after this del_example_namespace,
  // so we need to keep the ec.num_features correct,
  // so shared features are included in the reported number of "current features"
  // ec.num_features -= numf;
  auto* del_target = ec.feature_space.get_or_null(ns_index, ns_hash);
  assert(del_target->size() >= source_feature_group.size());
  assert(del_target != nullptr);
  ec.reset_total_sum_feat_sq();
  ec.num_features -= source_feature_group.size();
  del_target->truncate_to(del_target->size() - source_feature_group.size());
  del_target->sum_feat_sq -= source_feature_group.sum_feat_sq;
  if (del_target->empty()) { ec.feature_space.remove(ns_index, ns_hash); }
}

void add_example_namespaces_from_example(example& target, const example& source)
{
  for (auto& bucket : const_cast<example&>(source))
  {
    for (auto it = bucket.begin(); it != bucket.end(); ++it)
    {
      if (it->index == constant_namespace) continue;
      add_example_namespace(target, it->index, it->hash, it->feats);
    }
  }
}

void del_example_namespaces_from_example(example& target, const example& source)
{
  for (auto& bucket : const_cast<example&>(source))
  {
    for (auto it = bucket.begin(); it != bucket.end(); ++it)
    {
      if (it->index == constant_namespace) continue;
      del_example_namespace(target, it->index, it->hash, it->feats);
    }
  }
}

void add_example_namespace_from_memory(label_feature_map& lfm, example& ec, size_t lab)
{
  auto res_iter = lfm.find(lab);
  if (res_iter == lfm.end()) return;
  add_example_namespace(ec, static_cast<unsigned char>('l'), static_cast<uint64_t>('l'), res_iter->second);
}

void del_example_namespace_from_memory(label_feature_map& lfm, example& ec, size_t lab)
{
  auto res_iter = lfm.find(lab);
  if (res_iter == lfm.end()) return;
  del_example_namespace(ec, static_cast<unsigned char>('l'), static_cast<uint64_t>('l'), res_iter->second);
}

void set_label_features(label_feature_map& lfm, size_t lab, const features& source_feature_group)
{
  if (lfm.find(lab) == lfm.end()) return;
  lfm.emplace(lab, source_feature_group);
}

}  // namespace LabelDict
