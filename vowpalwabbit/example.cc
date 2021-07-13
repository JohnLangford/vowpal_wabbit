// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cstdint>
#include <algorithm>

#include "example.h"
#include "gd.h"
#include "simple_label_parser.h"
#include "interactions.h"

float calculate_total_sum_features_squared(bool permutations, example& ec)
{
  float sum_features_squared = 0.f;
  for (auto& bucket : ec)
  {
    for (const auto& fs : bucket) { sum_features_squared += fs.feats.sum_feat_sq; }
  }

  size_t ignored_interacted_feature_count = 0;
  float calculated_sum_features_squared = 0.f;
  INTERACTIONS::eval_count_of_generated_ft(permutations, *ec.interactions, ec.feature_space,
      ignored_interacted_feature_count, calculated_sum_features_squared);
  sum_features_squared += calculated_sum_features_squared;
  return sum_features_squared;
}

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_DEPRECATED_USAGE
example::example() { in_use = true; }

example::~example()
{
  if (passthrough)
  {
    delete passthrough;
    passthrough = nullptr;
  }
}
VW_WARNING_STATE_POP

float collision_cleanup(features& fs)
{
  uint64_t last_index = static_cast<uint64_t>(-1);
  float sum_sq = 0.f;
  features::iterator pos = fs.begin();
  for (features::iterator& f : fs)
  {
    if (last_index == f.index())
      pos.value() += f.value();
    else
    {
      sum_sq += pos.value() * pos.value();
      ++pos;
      pos.value() = f.value();
      pos.index() = f.index();
      last_index = f.index();
    }
  }

  sum_sq += pos.value() * pos.value();
  fs.sum_feat_sq = sum_sq;
  ++pos;
  fs.truncate_to(pos);

  return sum_sq;
}

namespace VW
{
void copy_example_label(example* dst, example* src, void (*)(polylabel*, polylabel*))
{
  dst->l = src->l;
  dst->_reduction_features = src->_reduction_features;
}

void copy_example_label(example* dst, const example* src) { dst->l = src->l; }

void copy_example_metadata(example* dst, const example* src)
{
  dst->tag = src->tag;
  dst->example_counter = src->example_counter;

  dst->ft_offset = src->ft_offset;

  dst->partial_prediction = src->partial_prediction;
  if (src->passthrough == nullptr)
    dst->passthrough = nullptr;
  else
  {
    dst->passthrough = new features(*src->passthrough);
  }
  dst->loss = src->loss;
  dst->weight = src->weight;
  dst->confidence = src->confidence;
  dst->test_only = src->test_only;
  dst->end_pass = src->end_pass;
  dst->is_newline = src->is_newline;
  dst->sorted = src->sorted;
}

void copy_example_data(example* dst, const example* src)
{
  copy_example_metadata(dst, src);

  // copy feature data
  dst->feature_space = src->feature_space;
  dst->num_features = src->num_features;
  dst->total_sum_feat_sq = src->total_sum_feat_sq;
  dst->total_sum_feat_sq_calculated = src->total_sum_feat_sq_calculated;
  dst->use_permutations = src->use_permutations;
  dst->interactions = src->interactions;
  dst->_debug_current_reduction_depth = src->_debug_current_reduction_depth;
}

void copy_example_metadata(bool /* audit */, example* dst, example* src) { copy_example_metadata(dst, src); }

void copy_example_data(bool /* audit */, example* dst, example* src) { copy_example_data(dst, src); }

void copy_example_data(bool /* audit */, example* dst, example* src, void (*copy_label)(polylabel*, polylabel*))
{
  copy_example_data(dst, src);
  copy_example_label(dst, src, copy_label);
}

void copy_example_data(
    bool audit, example* dst, example* src, size_t /*label_size*/, void (*copy_label)(polylabel*, polylabel*))
{
  copy_example_data(audit, dst, src, copy_label);
}

void copy_example_data_with_label(example* dst, const example* src)
{
  copy_example_data(dst, src);
  copy_example_label(dst, src);
}

void move_feature_namespace(example* dst, example* src, namespace_index c)
{
  auto& group_list = src->feature_space.get_list(c);

  // Check if the range is empty.
  if (group_list.empty()) { return; }

  std::vector<std::pair<namespace_index, uint64_t>> hashes_to_remove;
  hashes_to_remove.reserve(group_list.size());

  for (auto& ns_fs : group_list)
  {
    src->num_features -= ns_fs.feats.size();
    dst->num_features += ns_fs.feats.size();
    dst->feature_space.get_or_create(ns_fs.index, ns_fs.hash) = std::move(ns_fs.feats);
    hashes_to_remove.emplace_back(ns_fs.index, ns_fs.hash);
  }
  for (auto idx_hash : hashes_to_remove) { src->feature_space.remove(idx_hash.first, idx_hash.second); }

  src->reset_total_sum_feat_sq();
  dst->reset_total_sum_feat_sq();
}

}  // namespace VW

struct features_and_source
{
  v_array<feature> feature_map;  // map to store sparse feature vectors
  uint32_t stride_shift;
  uint64_t mask;
};

void vec_store(features_and_source& p, float fx, uint64_t fi)
{
  p.feature_map.push_back(feature(fx, (fi >> p.stride_shift) & p.mask));
}

namespace VW
{
feature* get_features(vw& all, example* ec, size_t& feature_map_len)
{
  features_and_source fs;
  fs.stride_shift = all.weights.stride_shift();
  fs.mask = all.weights.mask() >> all.weights.stride_shift();
  GD::foreach_feature<features_and_source, uint64_t, vec_store>(all, *ec, fs);

  feature_map_len = fs.feature_map.size();
  return fs.feature_map.begin();
}

void return_features(feature* f) { free_it(f); }
}  // namespace VW

struct full_features_and_source
{
  features fs;
  uint32_t stride_shift;
  uint64_t mask;
};

void vec_ffs_store(full_features_and_source& p, float fx, uint64_t fi)
{
  p.fs.push_back(fx, (fi >> p.stride_shift) & p.mask);
}

flat_example* flatten_example(vw& all, example* ec)
{
  flat_example& fec = calloc_or_throw<flat_example>();
  fec.l = ec->l;
  fec._reduction_features = ec->_reduction_features;

  fec.tag_len = ec->tag.size();
  if (fec.tag_len > 0)
  {
    fec.tag = calloc_or_throw<char>(fec.tag_len + 1);
    memcpy(fec.tag, ec->tag.begin(), fec.tag_len);
  }

  fec.example_counter = ec->example_counter;
  fec.ft_offset = ec->ft_offset;
  fec.num_features = ec->num_features;

  full_features_and_source ffs;
  ffs.stride_shift = all.weights.stride_shift();
  if (all.weights.not_null())  // TODO:temporary fix. all.weights is not initialized at this point in some cases.
    ffs.mask = all.weights.mask() >> all.weights.stride_shift();
  else
    ffs.mask = static_cast<uint64_t>(LONG_MAX) >> all.weights.stride_shift();
  GD::foreach_feature<full_features_and_source, uint64_t, vec_ffs_store>(all, *ec, ffs);

  std::swap(fec.fs, ffs.fs);

  return &fec;
}

flat_example* flatten_sort_example(vw& all, example* ec)
{
  flat_example* fec = flatten_example(all, ec);
  fec->fs.sort(all.parse_mask);
  fec->total_sum_feat_sq = collision_cleanup(fec->fs);
  return fec;
}

void free_flatten_example(flat_example* fec)
{
  // note: The label memory should be freed by by freeing the original example.
  if (fec)
  {
    fec->fs.~features();
    if (fec->tag_len > 0) free(fec->tag);
    free(fec);
  }
}

std::string cb_label_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "[l.cb={";
  auto& costs = ec.l.cb.costs;
  for (auto c = costs.cbegin(); c != costs.cend(); ++c)
  {
    strstream << "{c=" << c->cost << ",a=" << c->action << ",p=" << c->probability << ",pp=" << c->partial_prediction
              << "}";
  }
  strstream << "}]";
  return strstream.str();
}

std::string simple_label_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "[l=" << ec.l.simple.label << ",w=" << ec.weight << "]";
  return strstream.str();
}

std::string scalar_pred_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "[p=" << ec.pred.scalar << ", pp=" << ec.partial_prediction << "]";
  return strstream.str();
}

std::string a_s_pred_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "ec.pred.a_s[";
  for (uint32_t i = 0; i < ec.pred.a_s.size(); i++)
  { strstream << "(" << i << " = " << ec.pred.a_s[i].action << ", " << ec.pred.a_s[i].score << ")"; }
  strstream << "]";
  return strstream.str();
}

std::string multiclass_pred_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "ec.pred.multiclass = " << ec.pred.multiclass;
  return strstream.str();
}

std::string prob_dist_pred_to_string(const example& ec)
{
  std::stringstream strstream;
  strstream << "ec.pred.prob_dist[";
  for (uint32_t i = 0; i < ec.pred.pdf.size(); i++)
  {
    strstream << "(" << i << " = " << ec.pred.pdf[i].left << "-" << ec.pred.pdf[i].right << ", "
              << ec.pred.pdf[i].pdf_value << ")";
  }
  strstream << "]";
  return strstream.str();
}

namespace VW
{
example* alloc_examples(size_t, size_t count)
{
  example* ec = calloc_or_throw<example>(count);
  if (ec == nullptr) return nullptr;
  for (size_t i = 0; i < count; i++) { new (ec + i) example; }
  return ec;
}

example* alloc_examples(size_t count) { return alloc_examples(0, count); }

void dealloc_example(void (*)(polylabel*), example& ec, void (*)(void*)) { ec.~example(); }

void dealloc_examples(example* example_ptr, size_t count)
{
  for (size_t i = 0; i < count; i++) { (example_ptr + i)->~example(); }
  free(example_ptr);
}

void finish_example(vw&, example&);
void clean_example(vw&, example&, bool rewind);

void finish_example(vw& all, multi_ex& ec_seq)
{
  for (example* ecc : ec_seq) VW::finish_example(all, *ecc);
}

void return_multiple_example(vw& all, v_array<example*>& examples)
{
  for (auto ec : examples) { clean_example(all, *ec, true); }
  examples.clear();
}

}  // namespace VW

std::string debug_depth_indent_string(const example& ec)
{
  return debug_depth_indent_string(ec._debug_current_reduction_depth);
}
std::string debug_depth_indent_string(const multi_ex& ec) { return debug_depth_indent_string(*ec[0]); }
