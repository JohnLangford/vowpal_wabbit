// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "kskip_ngram_transformer.h"

#include "io/logger.h"
#include "feature_group.h"

#include <memory>

namespace logger = VW::io::logger;

void add_grams(size_t ngram, size_t skip_gram, features& destination_group, features& source_group,
    size_t initial_length, std::vector<size_t>& gram_mask, size_t skips)
{
  if (ngram == 0 && gram_mask.back() < initial_length)
  {
    size_t last = initial_length - gram_mask.back();
    for (size_t i = 0; i < last; i++)
    {
      uint64_t new_index = source_group.indicies[i];
      for (size_t n = 1; n < gram_mask.size(); n++)
      { new_index = new_index * quadratic_constant + source_group.indicies[i + gram_mask[n]]; }

      destination_group.push_back(1., new_index);
      if (!destination_group.space_names.empty())
      {
        std::string feature_name(destination_group.space_names[i].second);
        for (size_t n = 1; n < gram_mask.size(); n++)
        {
          feature_name += std::string("^");
          feature_name += std::string(destination_group.space_names[i + gram_mask[n]].second);
        }
        destination_group.space_names.push_back(audit_strings(destination_group.space_names[i].first, feature_name));
      }
    }
  }
  if (ngram > 0)
  {
    gram_mask.push_back(gram_mask.back() + 1 + skips);
    add_grams(ngram - 1, skip_gram, destination_group, source_group, initial_length, gram_mask, 0);
    gram_mask.pop_back();
  }
  if (skip_gram > 0 && ngram > 0)
  { add_grams(ngram, skip_gram - 1, destination_group, source_group, initial_length, gram_mask, skips + 1); }
}

void compile_gram(const std::vector<std::string>& grams, std::array<uint32_t, NUM_NAMESPACES>& dest,
    const std::string& descriptor, bool /*quiet*/)
{
  for (const auto& gram : grams)
  {
    if (isdigit(gram[0]) != 0)
    {
      int n = atoi(gram.c_str());
      logger::errlog_info("Generating {0}-{1} for all namespaces.", n, descriptor);
      for (size_t j = 0; j < NUM_NAMESPACES; j++) { dest[j] = n; }
    }
    else if (gram.size() == 1)
    {
      logger::log_error("You must specify the namespace index before the n");
    }
    else
    {
      int n = atoi(gram.c_str() + 1);
      dest[static_cast<uint32_t>(static_cast<unsigned char>(*gram.c_str()))] = n;
      logger::errlog_info("Generating {0}-{1} for {2} namespaces.", n, descriptor, gram[0]);
    }
  }
}

void VW::kskip_ngram_transformer::generate_grams(example* ex)
{
  for (namespace_index index : ex->feature_space.indices())
  {
    auto& feat_group_list = ex->feature_space.get_list(index);
    features* destination_feature_group = &feat_group_list.front().feats;
    features* source_feature_group = nullptr;
    std::unique_ptr<features> generated_feature_group;

    if (feat_group_list.size() > 1)
    {
      logger::errlog_warn(
          "Sentence based ngram concatenates feature groups that start with the same letter. This is deprecated "
          "behavior and in future it will change to treating every namespace as a separate sentence even if the first "
          "character is the same.");
      generated_feature_group = VW::make_unique<features>();
      source_feature_group = generated_feature_group.get();
      for (auto& ns_fs : feat_group_list) { generated_feature_group->concat(ns_fs.feats); }
    }
    else
    {
      source_feature_group = &feat_group_list.front().feats;
    }

    size_t length = source_feature_group->size();
    for (size_t n = 1; n < ngram_definition[index]; n++)
    {
      gram_mask.clear();
      gram_mask.push_back(0);
      add_grams(n, skip_definition[index], *destination_feature_group, *source_feature_group, length, gram_mask, 0);
    }
  }
}

VW::kskip_ngram_transformer VW::kskip_ngram_transformer::build(
    const std::vector<std::string>& grams, const std::vector<std::string>& skips, bool quiet)
{
  kskip_ngram_transformer transformer(grams, skips);

  compile_gram(grams, transformer.ngram_definition, "grams", quiet);
  compile_gram(skips, transformer.skip_definition, "skips", quiet);
  return transformer;
}

VW::kskip_ngram_transformer::kskip_ngram_transformer(std::vector<std::string> grams, std::vector<std::string> skips)
    : initial_ngram_definitions(std::move(grams)), initial_skip_definitions(std::move(skips))
{
  ngram_definition.fill(0);
  skip_definition.fill(0);
}
