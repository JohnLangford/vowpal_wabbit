// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cmath>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <memory>

#include "parse_example.h"
#include "constant.h"
#include "gd.h"
#include "cache.h"
#include "accumulate.h"
#include "learner.h"
#include "vw.h"
#include <map>
#include "memory.h"
#include "vw_allreduce.h"
#include "rand48.h"
#include "reductions.h"

#include "io/logger.h"

namespace logger = VW::io::logger;

#define SVM_KER_LIN 0
#define SVM_KER_RBF 1
#define SVM_KER_POLY 2

using namespace VW::LEARNER;
using namespace VW::config;

using std::endl;

struct svm_params;

static size_t num_kernel_evals = 0;
static size_t num_cache_evals = 0;

struct svm_example
{
  v_array<float> krow;
  flat_example ex;

  ~svm_example();
  void init_svm_example(flat_example* fec);
  int compute_kernels(svm_params& params);
  int clear_kernels();
};

struct svm_model
{
  size_t num_support;
  v_array<svm_example*> support_vec;
  v_array<float> alpha;
  v_array<float> delta;
};

void free_svm_model(svm_model* model)
{
  for (size_t i = 0; i < model->num_support; i++)
  {
    model->support_vec[i]->~svm_example();
    // Warning C6001 is triggered by the following:
    // example is allocated using (a) '&calloc_or_throw<svm_example>()' and freed using (b)
    // 'free(model->support_vec[i])'
    //
    // When the call to allocation is replaced by (a) 'new svm_example()' and deallocated using (b) 'operator delete
    // (model->support_vect[i])', the warning goes away. Disable SDL warning.
    //    #pragma warning(disable:6001)
    free(model->support_vec[i]);
    //  #pragma warning(default:6001)

    model->support_vec[i] = nullptr;
  }

  model->~svm_model();
  free(model);
}

struct svm_params
{
  size_t current_pass;
  bool active;
  bool active_pool_greedy;
  bool para_active;
  double active_c;

  size_t pool_size;
  size_t pool_pos;
  size_t subsample;  // NOTE: Eliminating subsample to only support 1/pool_size
  size_t reprocess;

  svm_model* model;
  size_t maxcache;
  // size_t curcache;

  svm_example** pool;
  float lambda;

  void* kernel_params;
  size_t kernel_type;

  size_t local_begin, local_end;
  size_t current_t;

  float loss_sum;

  vw* all;  // flatten, parallel
  std::shared_ptr<rand_state> _random_state;

  ~svm_params()
  {
    free(pool);
    if (all)
    {
      *(all->trace_message) << "Num support = " << model->num_support << endl;
      *(all->trace_message) << "Number of kernel evaluations = " << num_kernel_evals << " "
                            << "Number of cache queries = " << num_cache_evals << endl;
      *(all->trace_message) << "Total loss = " << loss_sum << endl;
    }
    if (model) { free_svm_model(model); }
    if (all) { *(all->trace_message) << "Done freeing model" << endl; }

    free(kernel_params);
    if (all)
    {
      *(all->trace_message) << "Done freeing kernel params" << endl;
      *(all->trace_message) << "Done with finish " << endl;
    }
  }
};

void svm_example::init_svm_example(flat_example* fec)
{
  ex = std::move(*fec);
  free(fec);
}

svm_example::~svm_example()
{
  // free flatten example contents
  // flat_example* fec = &calloc_or_throw<flat_example>();
  //*fec = ex;
  // free_flatten_example(fec);  // free contents of flat example and frees fec.
  if (ex.tag_len > 0) free(ex.tag);
}

float kernel_function(const flat_example* fec1, const flat_example* fec2, void* params, size_t kernel_type);

int svm_example::compute_kernels(svm_params& params)
{
  int alloc = 0;
  svm_model* model = params.model;
  size_t n = model->num_support;

  if (krow.size() < n)
  {
    // computing new kernel values and caching them
    // if(params->curcache + n > params->maxcache)
    // trim_cache(params);
    num_kernel_evals += krow.size();
    for (size_t i = krow.size(); i < n; i++)
    {
      svm_example* sec = model->support_vec[i];
      float kv = kernel_function(&ex, &(sec->ex), params.kernel_params, params.kernel_type);
      krow.push_back(kv);
      alloc += 1;
    }
  }
  else
    num_cache_evals += n;
  return alloc;
}

int svm_example::clear_kernels()
{
  int rowsize = static_cast<int>(krow.size());
  krow.clear();
  return -rowsize;
}

static int make_hot_sv(svm_params& params, size_t svi)
{
  svm_model* model = params.model;
  size_t n = model->num_support;
  if (svi >= model->num_support)
    *params.all->trace_message << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
  // rotate params fields
  svm_example* svi_e = model->support_vec[svi];
  int alloc = svi_e->compute_kernels(params);
  float svi_alpha = model->alpha[svi];
  float svi_delta = model->delta[svi];
  for (size_t i = svi; i > 0; --i)
  {
    model->support_vec[i] = model->support_vec[i - 1];
    model->alpha[i] = model->alpha[i - 1];
    model->delta[i] = model->delta[i - 1];
  }
  model->support_vec[0] = svi_e;
  model->alpha[0] = svi_alpha;
  model->delta[0] = svi_delta;
  // rotate cache
  for (size_t j = 0; j < n; j++)
  {
    svm_example* e = model->support_vec[j];
    size_t rowsize = e->krow.size();
    if (svi < rowsize)
    {
      float kv = e->krow[svi];
      for (size_t i = svi; i > 0; --i) e->krow[i] = e->krow[i - 1];
      e->krow[0] = kv;
    }
    else
    {
      float kv = svi_e->krow[j];
      e->krow.push_back(0);
      alloc += 1;
      for (size_t i = e->krow.size() - 1; i > 0; --i) e->krow[i] = e->krow[i - 1];
      e->krow[0] = kv;
    }
  }
  return alloc;
}

static int trim_cache(svm_params& params)
{
  int sz = static_cast<int>(params.maxcache);
  svm_model* model = params.model;
  size_t n = model->num_support;
  int alloc = 0;
  for (size_t i = 0; i < n; i++)
  {
    svm_example* e = model->support_vec[i];
    sz -= static_cast<int>(e->krow.size());
    if (sz < 0) alloc += e->clear_kernels();
  }
  return alloc;
}

int save_load_flat_example(io_buf& model_file, bool read, flat_example*& fec)
{
  size_t brw = 1;
  if (read)
  {
    fec = &calloc_or_throw<flat_example>();
    brw = model_file.bin_read_fixed(reinterpret_cast<char*>(fec), sizeof(flat_example), "");

    if (brw > 0)
    {
      if (fec->tag_len > 0)
      {
        fec->tag = calloc_or_throw<char>(fec->tag_len);
        brw = model_file.bin_read_fixed(fec->tag, fec->tag_len * sizeof(char), "");
        if (!brw) return 2;
      }
      if (fec->fs.size() > 0)
      {
        features& fs = fec->fs;
        size_t len = fs.size();
        fs.values.clear();
        fs.values.resize_but_with_stl_behavior(len);
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(fs.values.begin()), len * sizeof(feature_value), "");
        if (!brw) return 3;

        len = fs.indicies.size();
        fs.indicies.clear();
        fs.indicies.resize_but_with_stl_behavior(len);
        brw = model_file.bin_read_fixed(reinterpret_cast<char*>(fs.indicies.begin()), len * sizeof(feature_index), "");
        if (!brw) return 3;
      }
    }
    else
      return 1;
  }
  else
  {
    brw = model_file.bin_write_fixed(reinterpret_cast<char*>(fec), sizeof(flat_example));

    if (brw > 0)
    {
      if (fec->tag_len > 0)
      {
        brw = model_file.bin_write_fixed(fec->tag, static_cast<uint32_t>(fec->tag_len) * sizeof(char));
        if (!brw)
        {
	  // I'm assuming this is an error condition?
          logger::errlog_error("{0} {1}", fec->tag_len, fec->tag);
          return 2;
        }
      }
      if (fec->fs.size() > 0)
      {
        brw = model_file.bin_write_fixed(reinterpret_cast<char*>(fec->fs.values.begin()),
            static_cast<uint32_t>(fec->fs.size()) * sizeof(feature_value));
        if (!brw) return 3;
        brw = model_file.bin_write_fixed(reinterpret_cast<char*>(fec->fs.indicies.begin()),
            static_cast<uint32_t>(fec->fs.indicies.size()) * sizeof(feature_index));
        if (!brw) return 3;
      }
    }
    else
      return 1;
  }
  return 0;
}

void save_load_svm_model(svm_params& params, io_buf& model_file, bool read, bool text)
{
  svm_model* model = params.model;
  // TODO: check about initialization

  // params.all->opts_n_args.trace_message<<"Save load svm "<<read<<" "<<text<< endl;
  if (model_file.num_files() == 0) return;
  std::stringstream msg;
  bin_text_read_write_fixed(
      model_file, reinterpret_cast<char*>(&(model->num_support)), sizeof(model->num_support), "", read, msg, text);
  // params.all->opts_n_args.trace_message<<"Read num support "<<model->num_support<< endl;

  flat_example* fec = nullptr;
  if (read) { model->support_vec.reserve(model->num_support); }

  for (uint32_t i = 0; i < model->num_support; i++)
  {
    if (read)
    {
      save_load_flat_example(model_file, read, fec);
      svm_example* tmp = &calloc_or_throw<svm_example>();
      tmp->init_svm_example(fec);
      model->support_vec.push_back(tmp);
    }
    else
    {
      fec = &(model->support_vec[i]->ex);
      save_load_flat_example(model_file, read, fec);
    }
  }

  if (read) { model->alpha.resize_but_with_stl_behavior(model->num_support); }
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(model->alpha.data()),
      static_cast<uint32_t>(model->num_support) * sizeof(float), "", read, msg, text);
  if (read) { model->delta.resize_but_with_stl_behavior(model->num_support); }
  bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(model->delta.data()),
      static_cast<uint32_t>(model->num_support) * sizeof(float), "", read, msg, text);
}

void save_load(svm_params& params, io_buf& model_file, bool read, bool text)
{
  if (text)
  {
    *params.all->trace_message << "Not supporting readable model for kernel svm currently" << endl;
    return;
  }

  save_load_svm_model(params, model_file, read, text);
}

float linear_kernel(const flat_example* fec1, const flat_example* fec2)
{
  float dotprod = 0;

  features& fs_1 = const_cast<features&>(fec1->fs);
  features& fs_2 = const_cast<features&>(fec2->fs);
  if (fs_2.indicies.size() == 0) return 0.f;

  int numint = 0;
  for (size_t idx1 = 0, idx2 = 0; idx1 < fs_1.size() && idx2 < fs_2.size(); idx1++)
  {
    uint64_t ec1pos = fs_1.indicies[idx1];
    uint64_t ec2pos = fs_2.indicies[idx2];
    // params.all->opts_n_args.trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<" "<<ec2f->x<< endl;
    if (ec1pos < ec2pos) continue;

    while (ec1pos > ec2pos && ++idx2 < fs_2.size()) ec2pos = fs_2.indicies[idx2];

    if (ec1pos == ec2pos)
    {
      // params.all->opts_n_args.trace_message<<ec1pos<<" "<<ec2pos<<" "<<idx1<<" "<<idx2<<" "<<f->x<<"
      // "<<ec2f->x<< endl;
      numint++;
      dotprod += fs_1.values[idx1] * fs_2.values[idx2];
      ++idx2;
    }
  }
  // params.all->opts_n_args.trace_message<< endl;
  // params.all->opts_n_args.trace_message<<"numint = "<<numint<<" dotprod = "<<dotprod<< endl;
  return dotprod;
}

float poly_kernel(const flat_example* fec1, const flat_example* fec2, int power)
{
  float dotprod = linear_kernel(fec1, fec2);
  return static_cast<float>(std::pow(1 + dotprod, power));
}

float rbf_kernel(const flat_example* fec1, const flat_example* fec2, float bandwidth)
{
  float dotprod = linear_kernel(fec1, fec2);
  return expf(-(fec1->total_sum_feat_sq + fec2->total_sum_feat_sq - 2 * dotprod) * bandwidth);
}

float kernel_function(const flat_example* fec1, const flat_example* fec2, void* params, size_t kernel_type)
{
  switch (kernel_type)
  {
    case SVM_KER_RBF:
      return rbf_kernel(fec1, fec2, *(static_cast<float*>(params)));
    case SVM_KER_POLY:
      return poly_kernel(fec1, fec2, *(static_cast<int*>(params)));
    case SVM_KER_LIN:
      return linear_kernel(fec1, fec2);
  }
  return 0;
}

float dense_dot(float* v1, const v_array<float>& v2, size_t n)
{
  float dot_prod = 0.;
  for (size_t i = 0; i < n; i++) dot_prod += v1[i] * v2[i];
  return dot_prod;
}

void predict(svm_params& params, svm_example** ec_arr, float* scores, size_t n)
{
  svm_model* model = params.model;
  for (size_t i = 0; i < n; i++)
  {
    ec_arr[i]->compute_kernels(params);
    // std::cout<<"size of krow = "<<ec_arr[i]->krow.size()<< endl;
    if (ec_arr[i]->krow.size() > 0)
      scores[i] = dense_dot(ec_arr[i]->krow.begin(), model->alpha, model->num_support) / params.lambda;
    else
      scores[i] = 0;
  }
}

void predict(svm_params& params, single_learner&, example& ec)
{
  flat_example* fec = flatten_sort_example(*(params.all), &ec);
  if (fec)
  {
    svm_example* sec = &calloc_or_throw<svm_example>();
    sec->init_svm_example(fec);
    float score;
    predict(params, &sec, &score, 1);
    ec.pred.scalar = score;
    sec->~svm_example();
    free(sec);
  }
}

size_t suboptimality(svm_model* model, double* subopt)
{
  size_t max_pos = 0;
  double max_val = 0;
  for (size_t i = 0; i < model->num_support; i++)
  {
    float tmp = model->alpha[i] * model->support_vec[i]->ex.l.simple.label;
    const auto& simple_red_features =
        model->support_vec[i]->ex._reduction_features.template get<simple_label_reduction_features>();
    if ((tmp < simple_red_features.weight && model->delta[i] < 0) || (tmp > 0 && model->delta[i] > 0))
      subopt[i] = fabs(model->delta[i]);
    else
      subopt[i] = 0;

    if (subopt[i] > max_val)
    {
      max_val = subopt[i];
      max_pos = i;
    }
  }
  return max_pos;
}

int remove(svm_params& params, size_t svi)
{
  svm_model* model = params.model;
  if (svi >= model->num_support)
    *params.all->trace_message << "Internal error at " << __FILE__ << ":" << __LINE__ << endl;
  // shift params fields
  svm_example* svi_e = model->support_vec[svi];
  for (size_t i = svi; i < model->num_support - 1; ++i)
  {
    model->support_vec[i] = model->support_vec[i + 1];
    model->alpha[i] = model->alpha[i + 1];
    model->delta[i] = model->delta[i + 1];
  }
  svi_e->~svm_example();
  free(svi_e);
  model->support_vec.pop_back();
  model->alpha.pop_back();
  model->delta.pop_back();
  model->num_support--;
  // shift cache
  int alloc = 0;
  for (size_t j = 0; j < model->num_support; j++)
  {
    svm_example* e = model->support_vec[j];
    size_t rowsize = e->krow.size();
    if (svi < rowsize)
    {
      for (size_t i = svi; i < rowsize - 1; i++) e->krow[i] = e->krow[i + 1];
      e->krow.pop_back();
      alloc -= 1;
    }
  }
  return alloc;
}

int add(svm_params& params, svm_example* fec)
{
  svm_model* model = params.model;
  model->num_support++;
  model->support_vec.push_back(fec);
  model->alpha.push_back(0.);
  model->delta.push_back(0.);
  return static_cast<int>(model->support_vec.size() - 1);
}

bool update(svm_params& params, size_t pos)
{
  // params.all->opts_n_args.trace_message<<"Update\n";
  svm_model* model = params.model;
  bool overshoot = false;
  // params.all->opts_n_args.trace_message<<"Updating model "<<pos<<" "<<model->num_support<<" ";
  svm_example* fec = model->support_vec[pos];
  label_data& ld = fec->ex.l.simple;
  fec->compute_kernels(params);
  float* inprods = fec->krow.begin();
  float alphaKi = dense_dot(inprods, model->alpha, model->num_support);
  model->delta[pos] = alphaKi * ld.label / params.lambda - 1;
  float alpha_old = model->alpha[pos];
  alphaKi -= model->alpha[pos] * inprods[pos];
  model->alpha[pos] = 0.;

  float proj = alphaKi * ld.label;
  float ai = (params.lambda - proj) / inprods[pos];

  const auto& simple_red_features = fec->ex._reduction_features.template get<simple_label_reduction_features>();
  if (ai > simple_red_features.weight)
    ai = simple_red_features.weight;
  else if (ai < 0)
    ai = 0;

  ai *= ld.label;
  float diff = ai - alpha_old;

  if (std::fabs(diff) > 1.0e-06) overshoot = true;

  if (std::fabs(diff) > 1.)
  {
    // params.all->opts_n_args.trace_message<<"Here\n";
    diff = static_cast<float>(diff > 0) - (diff < 0);
    ai = alpha_old + diff;
  }

  for (size_t i = 0; i < model->num_support; i++)
  {
    label_data& ldi = model->support_vec[i]->ex.l.simple;
    model->delta[i] += diff * inprods[i] * ldi.label / params.lambda;
  }

  if (std::fabs(ai) <= 1.0e-10)
    remove(params, pos);
  else
    model->alpha[pos] = ai;

  return overshoot;
}

void copy_char(char& c1, const char& c2) noexcept
{
  if (c2 != '\0') c1 = c2;
}

void add_size_t(size_t& t1, const size_t& t2) noexcept { t1 += t2; }

void add_double(double& t1, const double& t2) noexcept { t1 += t2; }

void sync_queries(vw& all, svm_params& params, bool* train_pool)
{
  io_buf* b = new io_buf();

  char* queries;
  flat_example* fec = nullptr;

  for (size_t i = 0; i < params.pool_pos; i++)
  {
    if (!train_pool[i]) continue;

    fec = &(params.pool[i]->ex);
    save_load_flat_example(*b, false, fec);
    delete params.pool[i];
  }

  size_t* sizes = calloc_or_throw<size_t>(all.all_reduce->total);
  sizes[all.all_reduce->node] = b->unflushed_bytes_count();
  // params.all->opts_n_args.trace_message<<"Sizes = "<<sizes[all.node]<<" ";
  all_reduce<size_t, add_size_t>(all, sizes, all.all_reduce->total);

  size_t prev_sum = 0, total_sum = 0;
  for (size_t i = 0; i < all.all_reduce->total; i++)
  {
    if (i <= (all.all_reduce->node - 1)) prev_sum += sizes[i];
    total_sum += sizes[i];
  }

  // params.all->opts_n_args.trace_message<<total_sum<<" "<<prev_sum<< endl;
  if (total_sum > 0)
  {
    queries = calloc_or_throw<char>(total_sum);
    size_t bytes_copied = b->copy_to(queries + prev_sum, total_sum - prev_sum);
    if (bytes_copied < b->unflushed_bytes_count()) THROW("kernel_svm: Failed to alloc enough space.");

    all_reduce<char, copy_char>(all, queries, total_sum);
    b->replace_buffer(queries, total_sum);

    size_t num_read = 0;
    params.pool_pos = 0;

    for (size_t i = 0; i < params.pool_size; i++)
    {
      if (!save_load_flat_example(*b, true, fec))
      {
        params.pool[i] = &calloc_or_throw<svm_example>();
        params.pool[i]->init_svm_example(fec);
        train_pool[i] = true;
        params.pool_pos++;
      }
      else
        break;

      num_read += b->unflushed_bytes_count();
      if (num_read == prev_sum) params.local_begin = i + 1;
      if (num_read == prev_sum + sizes[all.all_reduce->node]) params.local_end = i;
    }
  }
  if (fec) free(fec);
  free(sizes);
  delete b;
}

void train(svm_params& params)
{
  // params.all->opts_n_args.trace_message<<"In train "<<params.all->training<< endl;

  bool* train_pool = calloc_or_throw<bool>(params.pool_size);
  for (size_t i = 0; i < params.pool_size; i++) train_pool[i] = false;

  float* scores = calloc_or_throw<float>(params.pool_pos);
  predict(params, params.pool, scores, params.pool_pos);

  if (params.active)
  {
    if (params.active_pool_greedy)
    {
      std::multimap<double, size_t> scoremap;
      for (size_t i = 0; i < params.pool_pos; i++)
        scoremap.insert(std::pair<const double, const size_t>(std::fabs(scores[i]), i));

      std::multimap<double, size_t>::iterator iter = scoremap.begin();
      // params.all->opts_n_args.trace_message<<params.pool_size<<" "<<"Scoremap: ";
      // for(;iter != scoremap.end();iter++)
      // params.all->opts_n_args.trace_message<<iter->first<<" "<<iter->second<<"
      // "<<((label_data*)params.pool[iter->second]->ld)->label<<"\t"; params.all->opts_n_args.trace_message<< endl;
      iter = scoremap.begin();

      for (size_t train_size = 1; iter != scoremap.end() && train_size <= params.subsample; train_size++)
      {
        // params.all->opts_n_args.trace_message<<train_size<<" "<<iter->second<<" "<<iter->first<< endl;
        train_pool[iter->second] = 1;
        iter++;
      }
    }
    else
    {
      for (size_t i = 0; i < params.pool_pos; i++)
      {
        float queryp = 2.0f /
            (1.0f +
                expf(static_cast<float>(params.active_c * std::fabs(scores[i])) *
                    static_cast<float>(pow(params.pool[i]->ex.example_counter, 0.5f))));
        if (params._random_state->get_and_update_random() < queryp)
        {
          svm_example* fec = params.pool[i];
          auto& simple_red_features = fec->ex._reduction_features.template get<simple_label_reduction_features>();
          simple_red_features.weight *= 1 / queryp;
          train_pool[i] = 1;
        }
      }
    }
    // free(scores);
  }

  if (params.para_active)
  {
    for (size_t i = 0; i < params.pool_pos; i++)
      if (!train_pool[i]) delete params.pool[i];
    sync_queries(*(params.all), params, train_pool);
  }

  if (params.all->training)
  {
    svm_model* model = params.model;

    for (size_t i = 0; i < params.pool_pos; i++)
    {
      // params.all->opts_n_args.trace_message<<"process: "<<i<<" "<<train_pool[i]<< endl;
      int model_pos = -1;
      if (params.active)
      {
        if (train_pool[i])
        {
          // params.all->opts_n_args.trace_message<<"i = "<<i<<"train_pool[i] = "<<train_pool[i]<<"
          // "<<params.pool[i]->example_counter<< endl;
          model_pos = add(params, params.pool[i]);
        }
      }
      else
        model_pos = add(params, params.pool[i]);

      if (model_pos >= 0)
      {
        bool overshoot = update(params, model_pos);

        double* subopt = calloc_or_throw<double>(model->num_support);
        for (size_t j = 0; j < params.reprocess; j++)
        {
          if (model->num_support == 0) break;
          int randi = 1;
          if (params._random_state->get_and_update_random() < 0.5) randi = 0;
          if (randi)
          {
            size_t max_pos = suboptimality(model, subopt);
            if (subopt[max_pos] > 0)
            {
              if (!overshoot && max_pos == static_cast<size_t>(model_pos) && max_pos > 0 && j == 0)
                *params.all->trace_message << "Shouldn't reprocess right after process!!!" << endl;
              if (max_pos * model->num_support <= params.maxcache) make_hot_sv(params, max_pos);
              update(params, max_pos);
            }
          }
          else
          {
            size_t rand_pos =
                static_cast<size_t>(floorf(params._random_state->get_and_update_random() * model->num_support));
            update(params, rand_pos);
          }
        }
        free(subopt);
      }
    }
  }
  else
    for (size_t i = 0; i < params.pool_pos; i++) delete params.pool[i];

  free(scores);
  free(train_pool);
}

void learn(svm_params& params, single_learner&, example& ec)
{
  flat_example* fec = flatten_sort_example(*(params.all), &ec);
  if (fec)
  {
    svm_example* sec = &calloc_or_throw<svm_example>();
    sec->init_svm_example(fec);
    float score = 0;
    predict(params, &sec, &score, 1);
    ec.pred.scalar = score;
    ec.loss = std::max(0.f, 1.f - score * ec.l.simple.label);
    params.loss_sum += ec.loss;
    if (params.all->training && ec.example_counter % 100 == 0) trim_cache(params);
    if (params.all->training && ec.example_counter % 1000 == 0 && ec.example_counter >= 2)
    {
      *params.all->trace_message << "Number of support vectors = " << params.model->num_support << endl;
      *params.all->trace_message << "Number of kernel evaluations = " << num_kernel_evals << " "
                                 << "Number of cache queries = " << num_cache_evals << " loss sum = " << params.loss_sum
                                 << " " << params.model->alpha[params.model->num_support - 1] << " "
                                 << params.model->alpha[params.model->num_support - 2] << endl;
    }
    params.pool[params.pool_pos] = sec;
    params.pool_pos++;

    if (params.pool_pos == params.pool_size)
    {
      train(params);
      params.pool_pos = 0;
    }
  }
}

VW::LEARNER::base_learner* kernel_svm_setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  auto params = scoped_calloc_or_throw<svm_params>();
  std::string kernel_type;
  float bandwidth = 1.f;
  int degree = 2;

  bool ksvm = false;

  option_group_definition new_options("Kernel SVM");
  new_options.add(make_option("ksvm", ksvm).keep().necessary().help("kernel svm"))
      .add(make_option("reprocess", params->reprocess).default_value(1).help("number of reprocess steps for LASVM"))
      .add(make_option("pool_greedy", params->active_pool_greedy).help("use greedy selection on mini pools"))
      .add(make_option("para_active", params->para_active).help("do parallel active learning"))
      .add(make_option("pool_size", params->pool_size).default_value(1).help("size of pools for active learning"))
      .add(make_option("subsample", params->subsample)
               .default_value(1)
               .help("number of items to subsample from the pool"))
      .add(make_option("kernel", kernel_type)
               .keep()
               .default_value("linear")
               .help("type of kernel (rbf or linear (default))"))
      .add(make_option("bandwidth", bandwidth).keep().default_value(1.f).help("bandwidth of rbf kernel"))
      .add(make_option("degree", degree).keep().default_value(2).help("degree of poly kernel"));

  if (!options.add_parse_and_check_necessary(new_options)) { return nullptr; }

  std::string loss_function = "hinge";
  float loss_parameter = 0.0;
  all.loss = getLossFunction(all, loss_function, loss_parameter);

  params->model = &calloc_or_throw<svm_model>();
  new (params->model) svm_model();
  params->model->num_support = 0;
  params->maxcache = 1024 * 1024 * 1024;
  params->loss_sum = 0.;
  params->all = &all;
  params->_random_state = all.get_random_state();

  // This param comes from the active reduction.
  // During options refactor: this changes the semantics a bit - now this will only be true if --active was supplied and
  // NOT --simulation
  if (all.active) params->active = true;
  if (params->active) params->active_c = 1.;

  params->pool = calloc_or_throw<svm_example*>(params->pool_size);
  params->pool_pos = 0;

  if (!options.was_supplied("subsample") && params->para_active)
    params->subsample = static_cast<size_t>(ceil(params->pool_size / all.all_reduce->total));

  params->lambda = all.l2_lambda;
  if (params->lambda == 0.) params->lambda = 1.;
  *params->all->trace_message << "Lambda = " << params->lambda << endl;
  *params->all->trace_message << "Kernel = " << kernel_type << endl;

  if (kernel_type.compare("rbf") == 0)
  {
    params->kernel_type = SVM_KER_RBF;
    *params->all->trace_message << "bandwidth = " << bandwidth << endl;
    params->kernel_params = &calloc_or_throw<double>();
    *(static_cast<float*>(params->kernel_params)) = bandwidth;
  }
  else if (kernel_type.compare("poly") == 0)
  {
    params->kernel_type = SVM_KER_POLY;
    *params->all->trace_message << "degree = " << degree << endl;
    params->kernel_params = &calloc_or_throw<int>();
    *(static_cast<int*>(params->kernel_params)) = degree;
  }
  else
    params->kernel_type = SVM_KER_LIN;

  params->all->weights.stride_shift(0);

  learner<svm_params, example>& l = init_learner(params, learn, predict, 1, all.get_setupfn_name(kernel_svm_setup));
  l.set_save_load(save_load);
  return make_base(l);
}
