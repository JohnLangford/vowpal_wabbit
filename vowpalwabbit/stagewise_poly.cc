// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cfloat>
#include <cassert>
#include <cmath>

#include "gd.h"
#include "accumulate.h"
#include "reductions.h"
#include "vw.h"
#include "vw_allreduce.h"

//#define MAGIC_ARGUMENT //MAY IT NEVER DIE //LIVE LONG AND PROSPER
// TODO: This file makes extensive use of #ifdef DEBUG for printing
//       leave this alone for now

using namespace VW::LEARNER;
using namespace VW::config;

static constexpr uint32_t parent_bit = 1;
static constexpr uint32_t cycle_bit = 2;
static constexpr uint32_t tree_atomics = 134;
static constexpr float tolerance = 1e-9f;
static constexpr uint32_t indicator_bit = 128;
static constexpr uint32_t default_depth = 127;

struct sort_data
{
  float weightsal;
  uint64_t wid;
};

struct stagewise_poly
{
  vw *all;  // many uses, unmodular reduction

  float sched_exponent;
  uint32_t batch_sz;
  bool batch_sz_double;

  sort_data *sd;
  size_t sd_len;
  uint8_t *depthsbits;  // interleaved array storing depth information and parent/cycle bits

  uint64_t sum_sparsity;        // of synthetic example
  uint64_t sum_input_sparsity;  // of input example
  uint64_t num_examples;
  // following three are for parallel (see end_pass())
  uint64_t sum_sparsity_sync;
  uint64_t sum_input_sparsity_sync;
  uint64_t num_examples_sync;

  example synth_ec;
  // following is bookkeeping in synth_ec creation (dfs)
  feature synth_rec_f;
  example *original_ec;
  uint32_t cur_depth;
  bool training;
  uint64_t last_example_counter;
  size_t numpasses;
  uint32_t next_batch_sz;
  bool update_support;

#ifdef DEBUG
  uint32_t max_depth;
  uint32_t depths[100000];
#endif  // DEBUG

#ifdef MAGIC_ARGUMENT
  float magic_argument;
#endif  // MAGIC_ARGUMENT

  ~stagewise_poly()
  {
#ifdef DEBUG
    std::cout << "total feature number (after poly expansion!) = " << sum_sparsity << std::endl;
#endif  // DEBUG

    free(sd);
    free(depthsbits);
  }
};

inline uint64_t stride_shift(const stagewise_poly &poly, uint64_t idx)
{
  return idx << poly.all->weights.stride_shift();
}

inline uint64_t stride_un_shift(const stagewise_poly &poly, uint64_t idx)
{
  return idx >> poly.all->weights.stride_shift();
}

inline uint64_t do_ft_offset(const stagewise_poly &poly, uint64_t idx)
{
  assert(!poly.original_ec || poly.synth_ec.ft_offset == poly.original_ec->ft_offset);
  return idx + poly.synth_ec.ft_offset;
}

inline uint64_t un_ft_offset(const stagewise_poly &poly, uint64_t idx)
{
  assert(!poly.original_ec || poly.synth_ec.ft_offset == poly.original_ec->ft_offset);
  if (poly.synth_ec.ft_offset == 0)
    return idx;
  else
  {
    while (idx < poly.synth_ec.ft_offset) { idx += poly.all->length() << poly.all->weights.stride_shift(); }
    return idx - poly.synth_ec.ft_offset;
  }
}

inline uint64_t wid_mask(const stagewise_poly &poly, uint64_t wid) { return wid & poly.all->weights.mask(); }

inline uint64_t wid_mask_un_shifted(const stagewise_poly &poly, uint64_t wid)
{
  return stride_un_shift(poly, wid & poly.all->weights.mask());
}

inline uint64_t constant_feat(const stagewise_poly &poly) { return stride_shift(poly, constant * poly.all->wpp); }

inline uint64_t constant_feat_masked(const stagewise_poly &poly) { return wid_mask(poly, constant_feat(poly)); }

inline size_t depthsbits_sizeof(const stagewise_poly &poly) { return (2 * poly.all->length() * sizeof(uint8_t)); }

void depthsbits_create(stagewise_poly &poly)
{
  poly.depthsbits = calloc_or_throw<uint8_t>(2 * poly.all->length());
  for (uint64_t i = 0; i < poly.all->length() * 2; i += 2)
  {
    poly.depthsbits[i] = default_depth;
    poly.depthsbits[i + 1] = indicator_bit;
  }
}

inline bool parent_get(const stagewise_poly &poly, uint64_t wid)
{
  assert(wid % stride_shift(poly, 1) == 0);
  assert(do_ft_offset(poly, wid) % stride_shift(poly, 1) == 0);
  return poly.depthsbits[wid_mask_un_shifted(poly, do_ft_offset(poly, wid)) * 2 + 1] & parent_bit;
}

inline void parent_toggle(stagewise_poly &poly, uint64_t wid)
{
  assert(wid % stride_shift(poly, 1) == 0);
  assert(do_ft_offset(poly, wid) % stride_shift(poly, 1) == 0);
  poly.depthsbits[wid_mask_un_shifted(poly, do_ft_offset(poly, wid)) * 2 + 1] ^= parent_bit;
}

inline bool cycle_get(const stagewise_poly &poly, uint64_t wid)
{
  // note: intentionally leaving out ft_offset.
  assert(wid % stride_shift(poly, 1) == 0);
  if ((poly.depthsbits[wid_mask_un_shifted(poly, wid) * 2 + 1] & cycle_bit) > 0)
    return true;
  else
    return false;
}

inline void cycle_toggle(stagewise_poly &poly, uint64_t wid)
{
  // note: intentionally leaving out ft_offset.
  assert(wid % stride_shift(poly, 1) == 0);
  poly.depthsbits[wid_mask_un_shifted(poly, wid) * 2 + 1] ^= cycle_bit;
}

inline uint8_t min_depths_get(const stagewise_poly &poly, uint64_t wid)
{
  assert(wid % stride_shift(poly, 1) == 0);
  assert(do_ft_offset(poly, wid) % stride_shift(poly, 1) == 0);
  return poly.depthsbits[stride_un_shift(poly, do_ft_offset(poly, wid)) * 2];
}

inline void min_depths_set(stagewise_poly &poly, uint64_t wid, uint8_t depth)
{
  assert(wid % stride_shift(poly, 1) == 0);
  assert(do_ft_offset(poly, wid) % stride_shift(poly, 1) == 0);
  poly.depthsbits[stride_un_shift(poly, do_ft_offset(poly, wid)) * 2] = depth;
}

#ifndef NDEBUG
void sanity_check_state(stagewise_poly &poly)
{
  for (uint64_t i = 0; i != poly.all->length(); ++i)
  {
    uint64_t wid = stride_shift(poly, i);

    assert(!cycle_get(poly, wid));

    assert(!(min_depths_get(poly, wid) == default_depth && parent_get(poly, wid)));

    if (poly.all->weights.sparse)
      assert(!(min_depths_get(poly, wid) == default_depth && fabsf(poly.all->weights.sparse_weights[wid]) > 0));
    else
      assert(!(min_depths_get(poly, wid) == default_depth && fabsf(poly.all->weights.dense_weights[wid]) > 0));
    // assert( min_depths_get(poly, wid) != default_depth && fabsf(poly.all->weights[wid]) < tolerance );

    assert(!(poly.depthsbits[wid_mask_un_shifted(poly, wid) * 2 + 1] & ~(parent_bit + cycle_bit + indicator_bit)));
  }
}
#endif  // NDEBUG

// Note.  OUTPUT & INPUT masked.
// It is very important that this function is invariant to stride.
inline uint64_t child_wid(const stagewise_poly &poly, uint64_t wi_atomic, uint64_t wi_general)
{
  assert(wi_atomic == wid_mask(poly, wi_atomic));
  assert(wi_general == wid_mask(poly, wi_general));
  assert((wi_atomic & (stride_shift(poly, 1) - 1)) == 0);
  assert((wi_general & (stride_shift(poly, 1) - 1)) == 0);

  if (wi_atomic == constant_feat_masked(poly))
    return wi_general;
  else if (wi_general == constant_feat_masked(poly))
    return wi_atomic;
  else
  {
    // This is basically the "Fowler–Noll–Vo" hash.  Ideally, the hash would be invariant
    // to the monomial, whereas this here is sensitive to the path followed, but whatever.
    // the two main big differences with FNV are: (1) the "*constant" case should also have
    // a big prime (so the default hash shouldn't be identity on small things, and (2) the
    // size should not just be a power of 2, but some big prime.
    return wid_mask(
        poly, stride_shift(poly, stride_un_shift(poly, wi_atomic) ^ (16777619 * stride_un_shift(poly, wi_general))));
  }
}

void sort_data_create(stagewise_poly &poly)
{
  poly.sd = nullptr;
  poly.sd_len = 0;
}

void sort_data_ensure_sz(stagewise_poly &poly, size_t len)
{
  if (poly.sd_len < len)
  {
    size_t len_candidate = 2 * len;
#ifdef DEBUG
    std::cout << "resizing sort buffer; current size " << poly.sd_len;
#endif  // DEBUG
    poly.sd_len = (len_candidate > poly.all->length()) ? poly.all->length() : len_candidate;
#ifdef DEBUG
    std::cout << ", new size " << poly.sd_len << std::endl;
#endif              // DEBUG
    free(poly.sd);  // okay for null.
    poly.sd = calloc_or_throw<sort_data>(poly.sd_len);
  }
  assert(len <= poly.sd_len);
}

#ifdef DEBUG
int sort_data_compar(const void *a_v, const void *b_v)
{
  return 2 * (((sort_data *)a_v)->weightsal < ((sort_data *)b_v)->weightsal) - 1;
}
#endif  // DEBUG

int sort_data_compar_heap(sort_data &a_v, sort_data &b_v) { return (a_v.weightsal > b_v.weightsal); }

/*
 * Performance note.
 *
 * On my laptop (Intel(R) Core(TM) i7-3520M CPU @ 2.90GHz), with compiler
 * optimizations enabled, this routine takes ~0.001 seconds with -b 18 and
 * ~0.06 seconds with -b 24.  Since it is intended to run ~8 times in 1-pass
 * mode and otherwise once per pass, it is considered adequate.
 *
 * Another choice (implemented in another version) is to never process the
 * whole weight vector (e.g., by updating a hash set of nonzero weights after
 * creating the synthetic example, and only processing that here).  This
 * choice was implemented in a similar algorithm and performed well.
 */
void sort_data_update_support(stagewise_poly &poly)
{
  assert(poly.num_examples);

  // ft_offset affects parent_set / parent_get.  This state must be reset at end.
  uint64_t pop_ft_offset = poly.original_ec->ft_offset;
  poly.synth_ec.ft_offset = 0;
  assert(poly.original_ec);
  poly.original_ec->ft_offset = 0;

  size_t num_new_features =
      static_cast<size_t>(std::pow(poly.sum_input_sparsity * 1.0f / poly.num_examples, poly.sched_exponent));
  num_new_features =
      (num_new_features > poly.all->length()) ? static_cast<uint64_t>(poly.all->length()) : num_new_features;
  sort_data_ensure_sz(poly, num_new_features);

  sort_data *heap_end = poly.sd;
  std::make_heap(poly.sd, heap_end, sort_data_compar_heap);  // redundant
  for (uint64_t i = 0; i != poly.all->length(); ++i)
  {
    uint64_t wid = stride_shift(poly, i);
    if (!parent_get(poly, wid) && wid != constant_feat_masked(poly))
    {
      float weightsal = (fabsf(poly.all->weights[wid]) * poly.all->weights[poly.all->normalized_idx + (wid)]);
      /*
       * here's some depth penalization code.  It was found to not improve
       * statistical performance, and meanwhile it is verified as giving
       * a nontrivial computational hit, thus commented out.
       *
       * - poly.magic_argument
       * sqrtf(min_depths_get(poly, stride_shift(poly, i)) * 1.0 / poly.num_examples)
       */
      ;
      if (weightsal > tolerance)
      {
        assert(heap_end >= poly.sd);
        assert(heap_end <= poly.sd + num_new_features);

        if (heap_end - poly.sd == static_cast<int>(num_new_features) && poly.sd->weightsal < weightsal)
        {
          std::pop_heap(poly.sd, heap_end, sort_data_compar_heap);
          --heap_end;
        }

        assert(heap_end >= poly.sd);
        assert(heap_end < poly.sd + poly.sd_len);

        if (heap_end - poly.sd < static_cast<int>(num_new_features))
        {
          heap_end->weightsal = weightsal;
          heap_end->wid = wid;
          ++heap_end;
          std::push_heap(poly.sd, heap_end, sort_data_compar_heap);
        }
      }
    }
  }
  num_new_features = static_cast<uint64_t>(heap_end - poly.sd);

#ifdef DEBUG
  // eyeballing weights a pain if unsorted.
  qsort(poly.sd, num_new_features, sizeof(sort_data), sort_data_compar);
#endif  // DEBUG

  for (uint64_t pos = 0; pos < num_new_features && pos < poly.sd_len; ++pos)
  {
    assert(!parent_get(poly, poly.sd[pos].wid) && poly.sd[pos].weightsal > tolerance &&
        poly.sd[pos].wid != constant_feat_masked(poly));
    parent_toggle(poly, poly.sd[pos].wid);
#ifdef DEBUG
    std::cout << "Adding feature " << pos << "/" << num_new_features << " || wid " << poly.sd[pos].wid
              << " || sort value " << poly.sd[pos].weightsal << std::endl;
#endif  // DEBUG
  }

#ifdef DEBUG
  std::cout << "depths:";
  for (uint64_t depth = 0; depth <= poly.max_depth && depth < sizeof(poly.depths) / sizeof(*poly.depths); ++depth)
    std::cout << "  [" << depth << "] = " << poly.depths[depth];
  std::cout << std::endl;

  std::cout << "Sanity check after sort... " << std::flush;
  sanity_check_state(poly);
  std::cout << "done" << std::endl;
#endif  // DEBUG

  // it's okay that these may have been initially unequal; synth_ec value irrelevant so far.
  poly.original_ec->ft_offset = pop_ft_offset;
  poly.synth_ec.ft_offset = pop_ft_offset;
}

void synthetic_reset(stagewise_poly &poly, example &ec)
{
  poly.synth_ec.l = ec.l;
  poly.synth_ec.weight = ec.weight;
  poly.synth_ec.tag = ec.tag;
  poly.synth_ec.example_counter = ec.example_counter;
  poly.synth_ec.interactions = &poly.all->interactions;

  /**
   * Some comments on ft_offset.
   *
   * The plan is to do the feature mapping dfs with weight indices ignoring
   * the ft_offset.  This is because ft_offset is then added at the end,
   * guaranteeing local/strided access on synth_ec.  This might not matter
   * too much in this implementation (where, e.g., --oaa runs one after the
   * other, not interleaved), but who knows.
   *
   * (The other choice is to basically ignore adjusting for ft_offset when
   * doing the traversal, which means synth_ec.ft_offset is 0 here...)
   *
   * Anyway, so here is how ft_offset matters:
   *   - synthetic_create_rec must "normalize it out" of the fed weight value
   *   - parent and min_depths set/get are adjusted for it.
   *   - cycle set/get are not adjusted for it, since it doesn't matter for them.
   *   - operations on the whole weight vector (sorting, save_load, all_reduce)
   *     ignore ft_offset, just treat the thing as a flat vector.
   **/
  poly.synth_ec.ft_offset = ec.ft_offset;

  poly.synth_ec.test_only = ec.test_only;
  poly.synth_ec.end_pass = ec.end_pass;
  poly.synth_ec.sorted = ec.sorted;

  poly.synth_ec.num_features = 0;

  // Ensure the feature group exists and it is empty.
  auto& tree_atomics_feature_group = poly.synth_ec.feature_space.get_or_create(tree_atomics, tree_atomics);
  tree_atomics_feature_group.clear();
}

void synthetic_decycle(stagewise_poly &poly)
{
  features& fs = poly.synth_ec.feature_space.get(tree_atomics, tree_atomics);
  for (size_t i = 0; i < fs.size(); ++i)
  {
    assert(cycle_get(poly, fs.indicies[i]));
    cycle_toggle(poly, fs.indicies[i]);
  }
}

void synthetic_create_rec(stagewise_poly &poly, float v, uint64_t findex)
{
  // Note: need to un_ft_shift since gd::foreach_feature bakes in the offset.
  uint64_t wid_atomic = wid_mask(poly, un_ft_offset(poly, findex));
  uint64_t wid_cur = child_wid(poly, wid_atomic, poly.synth_rec_f.weight_index);
  assert(wid_atomic % stride_shift(poly, 1) == 0);

  // Note: only mutate learner state when in training mode.  This is because
  // the average test errors across multiple data sets should be equal to
  // the test error on the merged dataset (which is violated if the code
  // below is run at training time).
  if (poly.cur_depth < min_depths_get(poly, wid_cur) && poly.training)
  {
    if (parent_get(poly, wid_cur))
    {
#ifdef DEBUG
      std::cout << "FOUND A TRANSPLANT!!! moving [" << wid_cur << "] from depth "
                << (uint64_t)min_depths_get(poly, wid_cur) << " to depth " << poly.cur_depth << std::endl;
#endif  // DEBUG
      // XXX arguably, should also fear transplants that occured with
      // a different ft_offset ; e.g., need to look out for cross-reduction
      // collisions.  Have not played with this issue yet...
      parent_toggle(poly, wid_cur);
    }
    min_depths_set(poly, wid_cur, static_cast<uint8_t>(poly.cur_depth));
  }

  if (!cycle_get(poly, wid_cur) &&
      ((poly.cur_depth > default_depth ? default_depth : poly.cur_depth) == min_depths_get(poly, wid_cur)))
  {
    cycle_toggle(poly, wid_cur);

#ifdef DEBUG
    ++poly.depths[poly.cur_depth];
#endif  // DEBUG

    feature temp = {v * poly.synth_rec_f.x, wid_cur};
    poly.synth_ec.feature_space.get(tree_atomics, tree_atomics).push_back(temp.x, temp.weight_index);
    poly.synth_ec.num_features++;

    if (parent_get(poly, temp.weight_index))
    {
      feature parent_f = poly.synth_rec_f;
      poly.synth_rec_f = temp;
      ++poly.cur_depth;
#ifdef DEBUG
      poly.max_depth = (poly.max_depth > poly.cur_depth) ? poly.max_depth : poly.cur_depth;
#endif  // DEBUG
      GD::foreach_feature<stagewise_poly, uint64_t, synthetic_create_rec>(*(poly.all), *(poly.original_ec), poly);
      --poly.cur_depth;
      poly.synth_rec_f = parent_f;
    }
  }
}

void synthetic_create(stagewise_poly &poly, example &ec, bool training)
{
  synthetic_reset(poly, ec);

  poly.cur_depth = 0;

  poly.synth_rec_f.x = 1.0;
  poly.synth_rec_f.weight_index = constant_feat_masked(poly);  // note: not ft_offset'd
  poly.training = training;
  /*
   * Another choice is to mark the constant feature as the single initial
   * parent, and recurse just on that feature (which arguably correctly interprets poly.cur_depth).
   * Problem with this is if there is a collision with the root...
   */
  GD::foreach_feature<stagewise_poly, uint64_t, synthetic_create_rec>(*poly.all, *poly.original_ec, poly);
  synthetic_decycle(poly);

  if (training)
  {
    poly.sum_sparsity += poly.synth_ec.get_num_features();
    poly.sum_input_sparsity += ec.get_num_features();
    poly.num_examples += 1;
  }
}

void predict(stagewise_poly &poly, single_learner &base, example &ec)
{
  poly.original_ec = &ec;
  synthetic_create(poly, ec, false);
  base.predict(poly.synth_ec);
  ec.partial_prediction = poly.synth_ec.partial_prediction;
  ec.updated_prediction = poly.synth_ec.updated_prediction;
  ec.pred.scalar = poly.synth_ec.pred.scalar;
}

void learn(stagewise_poly &poly, single_learner &base, example &ec)
{
  bool training = poly.all->training && ec.l.simple.label != FLT_MAX;
  poly.original_ec = &ec;

  if (training)
  {
    if (poly.update_support)
    {
      sort_data_update_support(poly);
      poly.update_support = false;
    }

    synthetic_create(poly, ec, training);
    base.learn(poly.synth_ec);
    ec.partial_prediction = poly.synth_ec.partial_prediction;
    ec.updated_prediction = poly.synth_ec.updated_prediction;
    ec.pred.scalar = poly.synth_ec.pred.scalar;

    if (ec.example_counter
        // following line is to avoid repeats when multiple reductions on same example.
        // XXX ideally, would get all "copies" of an example before scheduling the support
        // update, but how do we know?
        && poly.last_example_counter != ec.example_counter && poly.batch_sz &&
        ((poly.batch_sz_double && !(ec.example_counter % poly.next_batch_sz)) ||
            (!poly.batch_sz_double && !(ec.example_counter % poly.batch_sz))))
    {
      poly.next_batch_sz *= 2;  // no effect when !poly.batch_sz_double
      poly.update_support = (poly.all->all_reduce == nullptr || poly.numpasses == 1);
    }
    poly.last_example_counter = ec.example_counter;
  }
  else
    predict(poly, base, ec);
}

void reduce_min(uint8_t &v1, const uint8_t &v2)
{
  if (v1 == default_depth)
    v1 = v2;
  else if (v2 != default_depth)
    v1 = (v1 <= v2) ? v1 : v2;
}

void reduce_min_max(uint8_t &v1, const uint8_t &v2)
{
  bool parent_or_depth;
  if (v1 & indicator_bit)
    parent_or_depth = true;
  else
    parent_or_depth = false;
  bool p_or_d2;
  if (v2 & indicator_bit)
    p_or_d2 = true;
  else
    p_or_d2 = false;
  if (parent_or_depth != p_or_d2)
  {
#ifdef DEBUG
    std::cout << "Reducing parent with depth!!!!!";
#endif  // DEBUG
    return;
  }

  if (parent_or_depth)
    v1 = (v1 >= v2) ? v1 : v2;
  else
  {
    if (v1 == default_depth)
      v1 = v2;
    else if (v2 != default_depth)
      v1 = (v1 <= v2) ? v1 : v2;
  }
}

void end_pass(stagewise_poly &poly)
{
  if (!!poly.batch_sz || (poly.all->all_reduce != nullptr && poly.numpasses > 1)) return;

  uint64_t sum_sparsity_inc = poly.sum_sparsity - poly.sum_sparsity_sync;
  uint64_t sum_input_sparsity_inc = poly.sum_input_sparsity - poly.sum_input_sparsity_sync;
  uint64_t num_examples_inc = poly.num_examples - poly.num_examples_sync;

#ifdef DEBUG
  std::cout << "Sanity before allreduce\n";
  sanity_check_state(poly);
#endif  // DEBUG

  vw &all = *poly.all;
  if (all.all_reduce != nullptr)
  {
    /*
     * The following is inconsistent with the transplant code in
     * synthetic_create_rec(), which clears parent bits on depth mismatches.
     * But it's unclear what the right behavior is in general for either
     * case...
     */
    all_reduce<uint8_t, reduce_min_max>(all, poly.depthsbits, depthsbits_sizeof(poly));

    sum_input_sparsity_inc = static_cast<uint64_t>(accumulate_scalar(all, static_cast<float>(sum_input_sparsity_inc)));
    sum_sparsity_inc = static_cast<uint64_t>(accumulate_scalar(all, static_cast<float>(sum_sparsity_inc)));
    num_examples_inc = static_cast<uint64_t>(accumulate_scalar(all, static_cast<float>(num_examples_inc)));
  }

  poly.sum_input_sparsity_sync = poly.sum_input_sparsity_sync + sum_input_sparsity_inc;
  poly.sum_input_sparsity = poly.sum_input_sparsity_sync;
  poly.sum_sparsity_sync = poly.sum_sparsity_sync + sum_sparsity_inc;
  poly.sum_sparsity = poly.sum_sparsity_sync;
  poly.num_examples_sync = poly.num_examples_sync + num_examples_inc;
  poly.num_examples = poly.num_examples_sync;

#ifdef DEBUG
  std::cout << "Sanity after allreduce\n";
  sanity_check_state(poly);
#endif  // DEBUG

  if (poly.numpasses != poly.all->numpasses)
  {
    poly.update_support = true;
    poly.numpasses++;
  }
}

void finish_example(vw &all, stagewise_poly &poly, example &ec)
{
  size_t temp_num_features = ec.num_features;
  ec.num_features = poly.synth_ec.get_num_features();
  output_and_account_example(all, ec);
  ec.num_features = temp_num_features;
  VW::finish_example(all, ec);
}

void save_load(stagewise_poly &poly, io_buf &model_file, bool read, bool text)
{
  if (model_file.num_files() > 0)
  {
    std::stringstream msg;
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(poly.depthsbits),
        static_cast<uint32_t>(depthsbits_sizeof(poly)), "", read, msg, text);
  }
  // unfortunately, following can't go here since save_load called before gd::save_load and thus
  // weight vector state uninitialiazed.
  //#ifdef DEBUG
  //      std::cout << "Sanity check after save_load... " << flush;
  //      sanity_check_state(poly);
  //      std::cout << "done" << std::endl;
  //#endif //DEBUG
}

base_learner *stagewise_poly_setup(options_i &options, vw &all)
{
  auto poly = scoped_calloc_or_throw<stagewise_poly>();
  bool stage_poly = false;
  option_group_definition new_options("Stagewise polynomial options");
  new_options
      .add(make_option("stage_poly", stage_poly).keep().necessary().help("use stagewise polynomial feature learning"))
      .add(make_option("sched_exponent", poly->sched_exponent)
               .default_value(1.f)
               .help("exponent controlling quantity of included features"))
      .add(make_option("batch_sz", poly->batch_sz)
               .default_value(1000)
               .help("multiplier on batch size before including more features"))
      .add(make_option("batch_sz_no_doubling", poly->batch_sz_double).help("batch_sz does not double"));
#ifdef MAGIC_ARGUMENT
  new_options.add(
      make_typed_option("magic_argument", poly->magic_argument).default_value(0.).help("magical feature flag"));
#endif  // MAGIC_ARGUMENT

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  poly->all = &all;
  depthsbits_create(*poly.get());
  sort_data_create(*poly.get());

  poly->batch_sz_double = !poly->batch_sz_double;

  poly->sum_sparsity = 0;
  poly->sum_input_sparsity = 0;
  poly->num_examples = 0;
  poly->sum_sparsity_sync = 0;
  poly->sum_input_sparsity_sync = 0;
  poly->num_examples_sync = 0;
  poly->last_example_counter = std::numeric_limits<uint64_t>::max();
  poly->numpasses = 1;
  poly->update_support = false;
  poly->original_ec = nullptr;
  poly->next_batch_sz = poly->batch_sz;

  learner<stagewise_poly, example> &l = init_learner(
      poly, as_singleline(setup_base(options, all)), learn, predict, all.get_setupfn_name(stagewise_poly_setup));

  l.set_save_load(save_load);
  l.set_finish_example(finish_example);
  l.set_end_pass(end_pass);

  return make_base(l);
}
