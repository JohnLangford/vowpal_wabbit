// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <fstream>
#include <cfloat>
#include <cstdio>
#ifdef _WIN32
#  define NOMINMAX
#  include <winsock2.h>
#else
#  include <netdb.h>
#endif

#include "gd.h"
#include "rand48.h"
#include "reductions.h"
#include "vw_exception.h"
#include "array_parameters.h"
#include "shared_data.h"

using namespace VW::LEARNER;
using namespace VW::config;

struct gdmf
{
  vw* all;  // regressor, printing
  v_array<float> scalars;
  uint32_t rank;
  size_t no_win_counter;
  uint64_t early_stop_thres;
};

void mf_print_offset_features(gdmf& d, example& ec, size_t offset)
{
  // TODO: Where should audit stuff output to?
  vw& all = *d.all;
  parameters& weights = all.weights;
  uint64_t mask = weights.mask();
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket)
    {
      bool audit = !fs.feats.space_names.empty();
      for (const auto& f : fs.feats.audit_range())
      {
        std::cout << '\t';
        if (audit) std::cout << f.audit()->first << '^' << f.audit()->second << ':';
        std::cout << f.index() << "(" << ((f.index() + offset) & mask) << ")" << ':' << f.value();
        std::cout << ':' << (&weights[f.index()])[offset];
      }
    }
  }
  for (const auto& i : all.interactions)
  {
    if (i.size() != 2) THROW("can only use pairs in matrix factorization");

    /* print out nsk^feature:hash:value:weight:nsk^feature^:hash:value:weight:prod_weights */
    for (size_t k = 1; k <= d.rank; k++)
    {
      for (const auto ns0_feature_groups : ec.feature_space.get_list(i[0]))
      {
        for (const auto ns1_feature_groups : ec.feature_space.get_list(i[1]))
        {
          for (const auto& f1 : ns0_feature_groups.feats.audit_range())
          {
            for (const auto& f2 : ns1_feature_groups.feats.audit_range())
            {
              std::cout << '\t' << f1.audit()->first << k << '^' << f1.audit()->second << ':'
                        << ((f1.index() + k) & mask) << "(" << ((f1.index() + offset + k) & mask) << ")" << ':'
                        << f1.value();
              std::cout << ':' << (&weights[f1.index()])[offset + k];

              std::cout << ':' << f2.audit()->first << k << '^' << f2.audit()->second << ':'
                        << ((f2.index() + k + d.rank) & mask) << "(" << ((f2.index() + offset + k + d.rank) & mask)
                        << ")" << ':' << f2.value();
              std::cout << ':' << (&weights[f2.index()])[offset + k + d.rank];

              std::cout << ':' << (&weights[f1.index()])[offset + k] * (&weights[f2.index()])[offset + k + d.rank];
            }
          }
        }
      }
    }
  }
  std::cout << std::endl;
}

void mf_print_audit_features(gdmf& d, example& ec, size_t offset)
{
  print_result_by_ref(d.all->stdout_adapter.get(), ec.pred.scalar, -1, ec.tag);
  mf_print_offset_features(d, ec, offset);
}

struct pred_offset
{
  float p;
  uint64_t offset;
};

void offset_add(pred_offset& res, const float fx, float& fw) { res.p += (&fw)[res.offset] * fx; }

template <typename ContainerT>
size_t count_features(const ContainerT& list)
{
  size_t count = 0;
  for (const auto& item : list) { count += item.feats.size(); }
  return count;
}

template <class T>
float mf_predict(gdmf& d, example& ec, T& weights)
{
  vw& all = *d.all;
  const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  float prediction = simple_red_features.initial;

  ec.num_features_from_interactions = 0;
  for (const auto& i : d.all->interactions)
  {
    if (i.size() != 2) THROW("can only use pairs in matrix factorization");

    for (const auto& first_ns_fs : ec.feature_space.get_list(i[0]))
    {
      for (const auto& second_ns_fs : ec.feature_space.get_list(i[1]))
      {
        const auto interacted_count = first_ns_fs.feats.size() * second_ns_fs.feats.size();
        ec.num_features -= interacted_count;
        ec.num_features += first_ns_fs.feats.size() * d.rank;
        ec.num_features += second_ns_fs.feats.size() * d.rank;
        ec.num_features_from_interactions += interacted_count;
      }
    }
  }

  // clear stored predictions
  d.scalars.clear();

  float linear_prediction = 0.;
  // linear terms
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket) { GD::foreach_feature<float, GD::vec_add, T>(weights, fs.feats, linear_prediction); }
  }

  // store constant + linear prediction
  // note: constant is now automatically added
  d.scalars.push_back(linear_prediction);

  prediction += linear_prediction;
  // interaction terms
  for (const auto& i : d.all->interactions)
  {
    // The check for non-pair interactions is done in the previous loop
    const auto left_size = count_features(ec.feature_space.get_list(i[0]));
    const auto right_size = count_features(ec.feature_space.get_list(i[1]));
    if (left_size > 0 && right_size > 0)
    {
      for (uint64_t k = 1; k <= d.rank; k++)
      {
        // x_l * l^k
        // l^k is from index+1 to index+d.rank
        // float x_dot_l = sd_offset_add(weights, ec.atomics[(int)(*i)[0]].begin(), ec.atomics[(int)(*i)[0]].end(), k);
        pred_offset x_dot_l = {0., k};
        for (auto& first_ns_features : ec.feature_space.get_list(i[0]))
        { GD::foreach_feature<pred_offset, offset_add, T>(weights, first_ns_features.feats, x_dot_l); }
        // x_r * r^k
        // r^k is from index+d.rank+1 to index+2*d.rank
        // float x_dot_r = sd_offset_add(weights, ec.atomics[(int)(*i)[1]].begin(), ec.atomics[(int)(*i)[1]].end(),
        // k+d.rank);
        pred_offset x_dot_r = {0., k + d.rank};
        for (auto& second_ns_features : ec.feature_space.get_list(i[1]))
        { GD::foreach_feature<pred_offset, offset_add, T>(weights, second_ns_features.feats, x_dot_r); }

        prediction += x_dot_l.p * x_dot_r.p;

        // store prediction from interaction terms
        d.scalars.push_back(x_dot_l.p);
        d.scalars.push_back(x_dot_r.p);
      }
    }
  }

  // d.scalars has linear, x_dot_l_1, x_dot_r_1, x_dot_l_2, x_dot_r_2, ...

  ec.partial_prediction = prediction;

  all.set_minmax(all.sd, ec.l.simple.label);

  ec.pred.scalar = GD::finalize_prediction(all.sd, all.logger, ec.partial_prediction);

  if (ec.l.simple.label != FLT_MAX) ec.loss = all.loss->getLoss(all.sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;

  if (all.audit) mf_print_audit_features(d, ec, 0);

  return ec.pred.scalar;
}

float mf_predict(gdmf& d, example& ec)
{
  vw& all = *d.all;
  if (all.weights.sparse)
    return mf_predict(d, ec, all.weights.sparse_weights);
  else
    return mf_predict(d, ec, all.weights.dense_weights);
}

template <class T>
void sd_offset_update(T& weights, features& fs, uint64_t offset, float update, float regularization)
{
  for (size_t i = 0; i < fs.size(); i++)
    (&weights[fs.indicies[i]])[offset] += update * fs.values[i] - regularization * (&weights[fs.indicies[i]])[offset];
}

template <class T>
void mf_train(gdmf& d, example& ec, T& weights)
{
  vw& all = *d.all;
  label_data& ld = ec.l.simple;

  // use final prediction to get update size
  // update = eta_t*(y-y_hat) where eta_t = eta/(3*t^p) * importance weight
  float eta_t = all.eta / powf(static_cast<float>(all.sd->t) + ec.weight, all.power_t) / 3.f * ec.weight;
  float update = all.loss->getUpdate(ec.pred.scalar, ld.label, eta_t, 1.);  // ec.total_sum_feat_sq);

  float regularization = eta_t * all.l2_lambda;

  // linear update
  for (auto& bucket : ec)
  {
    for (auto& fs : bucket) { sd_offset_update<T>(weights, fs.feats, 0, update, regularization); }
  }

  // quadratic update
  for (const auto& i : all.interactions)
  {
    if (i.size() != 2) THROW("can only use pairs in matrix factorization");

    const auto left_size = count_features(ec.feature_space.get_list(i[0]));
    const auto right_size = count_features(ec.feature_space.get_list(i[1]));
    if (left_size > 0 && right_size > 0)
    {
      // update l^k weights
      for (size_t k = 1; k <= d.rank; k++)
      {
        // r^k \cdot x_r
        float r_dot_x = d.scalars[2 * k];
        // l^k <- l^k + update * (r^k \cdot x_r) * x_l
        for (auto& first_ns_features : ec.feature_space.get_list(i[0]))
        { sd_offset_update<T>(weights, first_ns_features.feats, k, update * r_dot_x, regularization); }
      }
      // update r^k weights
      for (size_t k = 1; k <= d.rank; k++)
      {
        // l^k \cdot x_l
        float l_dot_x = d.scalars[2 * k - 1];
        // r^k <- r^k + update * (l^k \cdot x_l) * x_r
        for (auto& second_ns_features : ec.feature_space.get_list(i[0]))
        { sd_offset_update<T>(weights, second_ns_features.feats, k + d.rank, update * l_dot_x, regularization); }
      }
    }
  }
}

void mf_train(gdmf& d, example& ec)
{
  if (d.all->weights.sparse)
    mf_train(d, ec, d.all->weights.sparse_weights);
  else
    mf_train(d, ec, d.all->weights.dense_weights);
}

void initialize_weights(weight* weights, uint64_t index, uint32_t stride)
{
  for (size_t i = 0; i != stride; ++i, ++index)
  {
    float initial_value = 0.1f * merand48(index);
    weights[i] = initial_value;
  }
}

void save_load(gdmf& d, io_buf& model_file, bool read, bool text)
{
  vw& all = *d.all;
  uint64_t length = static_cast<uint64_t>(1) << all.num_bits;
  if (read)
  {
    initialize_regressor(all);
    if (all.random_weights)
    {
      uint32_t stride = all.weights.stride();
      auto weight_initializer = [stride](
                                    weight* weights, uint64_t index) { initialize_weights(weights, index, stride); };

      all.weights.set_default(weight_initializer);
    }
  }

  if (model_file.num_files() > 0)
  {
    uint64_t i = 0;
    size_t brw = 1;
    do
    {
      brw = 0;
      size_t K = d.rank * 2 + 1;
      std::stringstream msg;
      msg << i << " ";
      brw += bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&i), sizeof(i), "", read, msg, text);
      if (brw != 0)
      {
        weight* w_i = &(all.weights.strided_index(i));
        for (uint64_t k = 0; k < K; k++)
        {
          weight* v = w_i + k;
          msg << v << " ";
          brw += bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(v), sizeof(*v), "", read, msg, text);
        }
      }
      if (text)
      {
        msg << "\n";
        brw += bin_text_read_write_fixed(model_file, nullptr, 0, "", read, msg, text);
      }

      if (!read) ++i;
    } while ((!read && i < length) || (read && brw > 0));
  }
}

void end_pass(gdmf& d)
{
  vw* all = d.all;

  all->eta *= all->eta_decay_rate;
  if (all->save_per_pass) save_predictor(*all, all->final_regressor_name, all->current_pass);

  if (!all->holdout_set_off)
  {
    if (summarize_holdout_set(*all, d.no_win_counter)) finalize_regressor(*all, all->final_regressor_name);
    if ((d.early_stop_thres == d.no_win_counter) &&
        ((all->check_holdout_every_n_passes <= 1) || ((all->current_pass % all->check_holdout_every_n_passes) == 0)))
      set_done(*all);
  }
}

void predict(gdmf& d, single_learner&, example& ec) { mf_predict(d, ec); }

void learn(gdmf& d, single_learner&, example& ec)
{
  vw& all = *d.all;

  mf_predict(d, ec);
  if (all.training && ec.l.simple.label != FLT_MAX) mf_train(d, ec);
}

base_learner* gd_mf_setup(options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<gdmf>();

  bool bfgs = false;
  bool conjugate_gradient = false;
  option_group_definition gf_md_options("Gradient Descent Matrix Factorization");
  gf_md_options.add(make_option("rank", data->rank).keep().necessary().help("rank for matrix factorization."));

  // Not supported, need to be checked to be false.
  gf_md_options.add(make_option("bfgs", bfgs).help("Option not supported by this reduction"));
  gf_md_options.add(
      make_option("conjugate_gradient", conjugate_gradient).help("Option not supported by this reduction"));

  if (!options.add_parse_and_check_necessary(gf_md_options)) return nullptr;

  if (options.was_supplied("adaptive")) THROW("adaptive is not implemented for matrix factorization");
  if (options.was_supplied("normalized")) THROW("normalized is not implemented for matrix factorization");
  if (options.was_supplied("exact_adaptive_norm"))
    THROW("normalized adaptive updates is not implemented for matrix factorization");

  if (bfgs || conjugate_gradient) THROW("bfgs is not implemented for matrix factorization");

  data->all = &all;
  data->no_win_counter = 0;

  // store linear + 2*rank weights per index, round up to power of two
  float temp = ceilf(logf(static_cast<float>(data->rank * 2 + 1)) / logf(2.f));
  all.weights.stride_shift(static_cast<size_t>(temp));
  all.random_weights = true;

  if (!all.holdout_set_off)
  {
    all.sd->holdout_best_loss = FLT_MAX;
    data->early_stop_thres = options.get_typed_option<size_t>("early_terminate").value();
  }

  if (!options.was_supplied("learning_rate") && !options.was_supplied("l"))
    all.eta = 10;  // default learning rate to 10 for non default update rule

  // default initial_t to 1 instead of 0
  if (!options.was_supplied("initial_t"))
  {
    all.sd->t = 1.f;
    all.initial_t = 1.f;
  }
  all.eta *= powf(static_cast<float>(all.sd->t), all.power_t);

  learner<gdmf, example>& l = init_learner(
      data, learn, predict, (UINT64_ONE << all.weights.stride_shift()), all.get_setupfn_name(gd_mf_setup), true);
  l.set_save_load(save_load);
  l.set_end_pass(end_pass);

  return make_base(l);
}
