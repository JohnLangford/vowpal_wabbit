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
  _ex->feature_space.get_or_create(_namespace_idx, _namespace_hash);
}

example_predict_builder::example_predict_builder(
    example_predict* ex, namespace_index namespace_idx, uint32_t feature_index_num_bits)
    : _ex(ex), _namespace_hash(namespace_idx)
{
  _feature_index_bit_mask = ((uint64_t)1 << feature_index_num_bits) - 1;
  add_namespace(namespace_idx);
  _ex->feature_space.get_or_create(_namespace_idx, _namespace_hash);
}

void example_predict_builder::add_namespace(namespace_index feature_group)
{
  _namespace_idx = feature_group;
}

void example_predict_builder::push_feature_string(const char* feature_name, feature_value value)
{
  feature_index feature_hash =
      _feature_index_bit_mask & hashstring(feature_name, strlen(feature_name), _namespace_hash);
  _ex->feature_space.get_or_null(_namespace_hash)->push_back(value, feature_hash);
}

void example_predict_builder::push_feature(feature_index feature_idx, feature_value value)
{
  auto* fs = _ex->feature_space.get_or_null(_namespace_hash);
  fs->push_back(value, _namespace_hash + feature_idx);
}
};  // namespace vw_slim
