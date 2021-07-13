// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <sstream>
#include <cfloat>
#include "reductions.h"
#include "v_array.h"

#include "io/logger.h"

using namespace VW::config;

namespace logger = VW::io::logger;

struct interact
{
  unsigned char n1, n2;  // namespaces to interact
  features feat_store;
  vw* all;
  float n1_feat_sq;
  size_t num_features;
};

size_t get_ns_size(example& ex, namespace_index ns)
{
  auto begin = ex.feature_space.index_flat_begin(ns);
  auto end = ex.feature_space.index_flat_end(ns);
  return end - begin;
}

bool contains_valid_namespaces(example& ex, namespace_index n1, namespace_index n2)
{
  // first feature must be 1 so we're sure that the anchor feature is present
  if (get_ns_size(ex, n1) == 0 || get_ns_size(ex, n2) == 0) return false;

  if ((*ex.feature_space.index_flat_begin(n1)).value() != 1)
  {
    // Anchor feature must be a number instead of text so that the relative offsets functions correctly but I don't
    // think we are able to test for this here.
    logger::log_error("Namespace '{}' misses anchor feature with value 1", static_cast<char>(n1));
    return false;
  }

  if ((*ex.feature_space.index_flat_begin(n2)).value() != 1)
  {
    logger::log_error("Namespace '{}' misses anchor feature with value 1", static_cast<char>(n2));
    return false;
  }

  return true;
}

using multiply_iterator_t =
    VW::chained_proxy_iterator<VW::namespaced_feature_store::list_iterator, features::audit_iterator>;
void multiply(features& f_dest, multiply_iterator_t f_src1_begin, multiply_iterator_t f_src1_end,
    multiply_iterator_t f_src2_begin, multiply_iterator_t f_src2_end, uint64_t weight_mask)
{
  f_dest.clear();
  uint64_t base_id1 = (*f_src1_begin).index() & weight_mask;
  uint64_t base_id2 = (*f_src2_begin).index() & weight_mask;

  f_dest.push_back((*f_src1_begin).value() * (*f_src2_begin).value(), (*f_src1_begin).index());

  uint64_t prev_id1 = 0;
  uint64_t prev_id2 = 0;
  ++f_src1_begin;
  ++f_src2_begin;

  for (; f_src1_begin != f_src1_end && f_src2_begin != f_src2_end;)
  {
    // calculating the relative offset from the namespace offset used to match features
    uint64_t cur_id1 = static_cast<uint64_t>((((*f_src1_begin).index() & weight_mask) - base_id1) & weight_mask);
    uint64_t cur_id2 = static_cast<uint64_t>((((*f_src2_begin).index() & weight_mask) - base_id2) & weight_mask);

    // checking for sorting requirement
    if (cur_id1 < prev_id1)
    {
      logger::log_error("interact features are out of order: {0} < {1}. Skipping features.", cur_id1, prev_id1);
      return;
    }

    if (cur_id2 < prev_id2)
    {
      logger::log_error("interact features are out of order: {0} < {1}. Skipping features.", cur_id2, prev_id2);
      return;
    }

    if (cur_id1 == cur_id2)
    {
      f_dest.push_back((*f_src1_begin).value() * (*f_src2_begin).value(), (*f_src1_begin).index());
      ++f_src1_begin;
      ++f_src2_begin;
    }
    else if (cur_id1 < cur_id2)
      ++f_src1_begin;
    else
      ++f_src2_begin;

    prev_id1 = cur_id1;
    prev_id2 = cur_id2;
  }
}

template <bool is_learn, bool print_all>
void predict_or_learn(interact& in, VW::LEARNER::single_learner& base, example& ec)
{
  if (!contains_valid_namespaces(ec, in.n1, in.n2))
  {
    if (is_learn)
      base.learn(ec);
    else
      base.predict(ec);

    return;
  }

  in.num_features = ec.num_features;
  ec.num_features -= get_ns_size(ec, in.n1);
  ec.num_features -= get_ns_size(ec, in.n2);

  in.feat_store.clear();
  multiply(in.feat_store, ec.feature_space.index_flat_begin(in.n1), ec.feature_space.index_flat_end(in.n1),
      ec.feature_space.index_flat_begin(in.n2), ec.feature_space.index_flat_end(in.n2), in.all->weights.mask());
  ec.reset_total_sum_feat_sq();
  ec.num_features = in.feat_store.size();

  std::vector<uint64_t> hashes_removed;
  std::vector<namespace_index> index_removed;
  std::vector<features> removed_features;

  auto remove_index = [&hashes_removed, &removed_features, &index_removed, &ec](namespace_index index) {
    auto& group_list = ec.feature_space.get_list(index);
    for (auto& group : group_list)
    {
      hashes_removed.push_back(group.hash);
      index_removed.push_back(group.index);
      removed_features.emplace_back(std::move(group.feats));
    }
    for (size_t i = 0; i < hashes_removed.size(); i++) { ec.feature_space.remove(index_removed[i], hashes_removed[i]); }
  };

  remove_index(in.n1);
  remove_index(in.n2);

  ec.feature_space.get_or_create(interact_reduction_namespace, interact_reduction_namespace) = std::move(in.feat_store);

  base.predict(ec);
  if (is_learn) base.learn(ec);

  in.feat_store = std::move(ec.feature_space.get(interact_reduction_namespace, interact_reduction_namespace));
  ec.feature_space.remove(interact_reduction_namespace, interact_reduction_namespace);

  // re-insert namespace into the right position
  for (size_t i = 0; i < hashes_removed.size(); i++)
  { ec.feature_space.get_or_create(index_removed[i], hashes_removed[i]) = std::move(removed_features[i]); }

  ec.num_features = in.num_features;
}

VW::LEARNER::base_learner* interact_setup(options_i& options, vw& all)
{
  std::string s;
  option_group_definition new_options("Interact via elementwise multiplication");
  new_options.add(make_option("interact", s)
                      .keep()
                      .necessary()
                      .help("Put weights on feature products from namespaces <n1> and <n2>"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (s.length() != 2)
  {
    logger::errlog_error("Need two namespace arguments to interact: {} won't do EXITING", s);
    return nullptr;
  }

  auto data = scoped_calloc_or_throw<interact>();

  data->n1 = static_cast<unsigned char>(s[0]);
  data->n2 = static_cast<unsigned char>(s[1]);
  logger::errlog_info("Interacting namespaces {0:c} and {1:c}", data->n1, data->n2);
  data->all = &all;

  VW::LEARNER::learner<interact, example>* l;
  l = &VW::LEARNER::init_learner(data, as_singleline(setup_base(options, all)), predict_or_learn<true, true>,
      predict_or_learn<false, true>, 1, all.get_setupfn_name(interact_setup));

  return make_base(*l);
}
