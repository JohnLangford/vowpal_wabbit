// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cstring>
#include <cfloat>
#include "reductions.h"
#include "rand48.h"
#include "vw_exception.h"
#include "parse_args.h"  // for spoof_hex_encoded_namespaces

using namespace VW::LEARNER;
using namespace VW::config;

struct LRQstate
{
  vw* all;  // feature creation, audit, hash_inv
  bool lrindices[256];
  std::unordered_map<uint64_t, size_t> orig_size;
  std::set<std::string> lrpairs;
  bool dropout;
  uint64_t seed;
  uint64_t initial_seed;
};

bool valid_int(const char* s)
{
  char* endptr;

  int v = strtoul(s, &endptr, 0);
  (void)v;

  return (*s != '\0' && *endptr == '\0');
}

inline bool cheesyrbit(uint64_t& seed) { return merand48(seed) > 0.5; }

inline float cheesyrand(uint64_t x)
{
  uint64_t seed = x;

  return merand48(seed);
}

constexpr inline bool example_is_test(example& ec) { return ec.l.simple.label == FLT_MAX; }

void reset_seed(LRQstate& lrq)
{
  if (lrq.all->bfgs) lrq.seed = lrq.initial_seed;
}

template <bool is_learn>
void predict_or_learn(LRQstate& lrq, single_learner& base, example& ec)
{
  vw& all = *lrq.all;

  // Remember original features

  lrq.orig_size.clear();
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket)
    {
      if (lrq.lrindices[fs.index]) { lrq.orig_size[fs.hash] = fs.feats.size(); }
    }
  }

  size_t which = ec.example_counter;
  float first_prediction = 0;
  float first_loss = 0;
  float first_uncertainty = 0;
  unsigned int maxiter = (is_learn && !example_is_test(ec)) ? 2 : 1;

  bool do_dropout = lrq.dropout && is_learn && !example_is_test(ec);
  float scale = (!lrq.dropout || do_dropout) ? 1.f : 0.5f;

  uint32_t stride_shift = lrq.all->weights.stride_shift();
  for (unsigned int iter = 0; iter < maxiter; ++iter, ++which)
  {
    // Add left LRQ features, holding right LRQ features fixed
    //     and vice versa
    // TODO: what happens with --lrq ab2 --lrq ac2
    //       i.e. namespace occurs multiple times (?)

    for (std::string const& i : lrq.lrpairs)
    {
      unsigned char left = i[which % 2];
      unsigned char right = i[(which + 1) % 2];
      unsigned int k = atoi(i.c_str() + 2);

      for (const auto& left_ns_fs : ec.feature_space.get_list(left))
      {
        const auto& left_fs = left_ns_fs.feats;
        for (unsigned int lfn = 0; lfn < lrq.orig_size[left_ns_fs.hash]; ++lfn)
        {
          float lfx = left_fs.values[lfn];
          uint64_t lindex = left_fs.indicies[lfn] + ec.ft_offset;
          for (unsigned int n = 1; n <= k; ++n)
          {
            if (!do_dropout || cheesyrbit(lrq.seed))
            {
              uint64_t lwindex = (lindex + (static_cast<uint64_t>(n) << stride_shift));
              weight* lw = &lrq.all->weights[lwindex];

              // perturb away from saddle point at (0, 0)
              if (is_learn)
              {
                if (!example_is_test(ec) && *lw == 0)
                {
                  *lw = cheesyrand(lwindex);  // not sure if lw needs a weight mask?
                }
              }

              for (auto& right_ns_fs : ec.feature_space.get_list(right))
              {
                auto& right_fs = right_ns_fs.feats;
                for (unsigned int rfn = 0; rfn < lrq.orig_size[right_ns_fs.hash]; ++rfn)
                {
                  // NB: ec.ft_offset added by base learner
                  float rfx = right_fs.values[rfn];
                  uint64_t rindex = right_fs.indicies[rfn];
                  uint64_t rwindex = (rindex + (static_cast<uint64_t>(n) << stride_shift));

                  right_fs.push_back(scale * *lw * lfx * rfx, rwindex);

                  if (all.audit || all.hash_inv)
                  {
                    std::stringstream new_feature_buffer;
                    new_feature_buffer << right << '^' << right_fs.space_names[rfn].second << '^' << n;
#ifdef _WIN32
                    char* new_space = _strdup("lrq");
                    char* new_feature = _strdup(new_feature_buffer.str().c_str());
#else
                    char* new_space = strdup("lrq");
                    char* new_feature = strdup(new_feature_buffer.str().c_str());
#endif
                    right_fs.space_names.push_back(audit_strings(new_space, new_feature));
                  }
                }
              }
            }
          }
        }
      }
    }

    if (is_learn)
      base.learn(ec);
    else
      base.predict(ec);

    // Restore example
    if (iter == 0)
    {
      first_prediction = ec.pred.scalar;
      first_loss = ec.loss;
      first_uncertainty = ec.confidence;
    }
    else
    {
      ec.pred.scalar = first_prediction;
      ec.loss = first_loss;
      ec.confidence = first_uncertainty;
    }

    for (const std::string& i : lrq.lrpairs)
    {
      unsigned char right = i[(which + 1) % 2];

      for (auto& right_ns_fs : ec.feature_space.get_list(right))
      { right_ns_fs.feats.truncate_to(lrq.orig_size[right_ns_fs.hash]); }
    }
  }  // end for(max_iter)
}

base_learner* lrq_setup(options_i& options, vw& all)
{
  auto lrq = scoped_calloc_or_throw<LRQstate>();
  std::vector<std::string> lrq_names;
  option_group_definition new_options("Low Rank Quadratics");
  new_options.add(make_option("lrq", lrq_names).keep().necessary().help("use low rank quadratic features"))
      .add(make_option("lrqdropout", lrq->dropout).keep().help("use dropout training for low rank quadratic features"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  uint32_t maxk = 0;
  lrq->all = &all;

  for (auto& lrq_name : lrq_names) lrq_name = spoof_hex_encoded_namespaces(lrq_name);

  new (&lrq->lrpairs) std::set<std::string>(lrq_names.begin(), lrq_names.end());

  lrq->initial_seed = lrq->seed = all.random_seed | 8675309;

  if (!all.logger.quiet)
  {
    *(all.trace_message) << "creating low rank quadratic features for pairs: ";
    if (lrq->dropout) *(all.trace_message) << "(using dropout) ";
  }

  for (std::string const& i : lrq->lrpairs)
  {
    if (!all.logger.quiet)
    {
      if ((i.length() < 3) || !valid_int(i.c_str() + 2))
        THROW("error, low-rank quadratic features must involve two sets and a rank.");

      *(all.trace_message) << i << " ";
    }
    // TODO: colon-syntax

    unsigned int k = atoi(i.c_str() + 2);

    lrq->lrindices[static_cast<int>(i[0])] = true;
    lrq->lrindices[static_cast<int>(i[1])] = true;

    maxk = std::max(k, k);
  }

  if (!all.logger.quiet) *(all.trace_message) << std::endl;

  all.wpp = all.wpp * static_cast<uint64_t>(1 + maxk);
  auto base = setup_base(options, all);
  learner<LRQstate, example>& l = init_learner(lrq, as_singleline(base), predict_or_learn<true>,
      predict_or_learn<false>, 1 + maxk, all.get_setupfn_name(lrq_setup), base->learn_returns_prediction);
  l.set_end_pass(reset_seed);

  // TODO: leaks memory ?
  return make_base(l);
}
