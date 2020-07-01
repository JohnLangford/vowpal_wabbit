#include "example_predict_builder.h"
#include "hash.h"
#include "hashstring.h"

namespace vw_slim
{
example_predict_builder::example_predict_builder(
    example_predict* ex, const char* namespace_name, uint32_t feature_index_num_bits)
    : _ex(ex)
{
  _feature_index_bit_mask = ((uint64_t)1 << feature_index_num_bits) - 1;
  add_namespace(namespace_name[0]);
  _namespace_hash = hashstring(namespace_name, strlen(namespace_name), 0);
}

example_predict_builder::example_predict_builder(
    example_predict* ex, namespace_index namespace_idx, uint32_t feature_index_num_bits)
    : _ex(ex), _namespace_hash(namespace_idx)
{
  _feature_index_bit_mask = ((uint64_t)1 << feature_index_num_bits) - 1;
  add_namespace(namespace_idx);
}

void example_predict_builder::add_namespace(namespace_index feature_group)
{
  _namespace_idx = feature_group;
  _ex->indices.unique_add_sorted(feature_group);
}

void example_predict_builder::push_feature_string(const char* feature_name, feature_value value)
{
  feature_index feature_hash =
      _feature_index_bit_mask & hashstring(feature_name, strlen(feature_name), _namespace_hash);
  _ex->feature_space[_namespace_idx].push_back(value, feature_hash);
}

void example_predict_builder::push_feature(feature_index feature_idx, feature_value value)
{
  _ex->feature_space[_namespace_idx].push_back(value, _namespace_hash + feature_idx);
}
};  // namespace vw_slim
