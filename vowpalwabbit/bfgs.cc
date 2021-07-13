// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

/*
The algorithm here is generally based on Nocedal 1980, Liu and Nocedal 1989.
Implementation by Miro Dudik.
 */
#include <cmath>
#include <fstream>
#include <float.h>
#ifndef _WIN32
#  include <netdb.h>
#endif
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <sys/timeb.h>
#include "accumulate.h"
#include "reductions.h"
#include "gd.h"
#include "vw_exception.h"
#include <exception>
#include <chrono>
#include "shared_data.h"

using namespace VW::LEARNER;
using namespace VW::config;

#define CG_EXTRA 1

#define MEM_GT 0
#define MEM_XT 1
#define MEM_YT 0
#define MEM_ST 1

#define W_XT 0
#define W_GT 1
#define W_DIR 2
#define W_COND 3

#define LEARN_OK 0
#define LEARN_CURV 1
#define LEARN_CONV 2

class curv_exception : public std::exception
{
} curv_ex;

/********************************************************************/
/* mem & w definition ***********************************************/
/********************************************************************/
// mem[2*i] = y_t
// mem[2*i+1] = s_t
//
// w[0] = weight
// w[1] = accumulated first derivative
// w[2] = step direction
// w[3] = preconditioner

constexpr float max_precond_ratio = 10000.f;

struct bfgs
{
  vw* all = nullptr;  // prediction, regressor
  int m = 0;
  float rel_threshold = 0.f;  // termination threshold

  double wolfe1_bound = 0.0;

  size_t final_pass = 0;

  std::chrono::time_point<std::chrono::system_clock> t_start_global;
  std::chrono::time_point<std::chrono::system_clock> t_end_global;
  double net_time = 0.0;

  v_array<float> predictions;
  size_t example_number = 0;
  size_t current_pass = 0;
  size_t no_win_counter = 0;
  size_t early_stop_thres = 0;

  // default transition behavior
  bool first_hessian_on = false;
  bool backstep_on = false;

  // set by initializer
  int mem_stride = 0;
  bool output_regularizer = false;
  float* mem = nullptr;
  double* rho = nullptr;
  double* alpha = nullptr;

  weight* regularizers = nullptr;
  // the below needs to be included when resetting, in addition to preconditioner and derivative
  int lastj = 0;
  int origin = 0;
  double loss_sum = 0.0;
  double previous_loss_sum = 0.0;
  float step_size = 0.f;
  double importance_weight_sum = 0.0;
  double curvature = 0.0;

  // first pass specification
  bool first_pass = false;
  bool gradient_pass = false;
  bool preconditioner_pass = false;

  ~bfgs()
  {
    free(mem);
    free(rho);
    free(alpha);
  }
};

constexpr const char* curv_message =
    "Zero or negative curvature detected.\n"
    "To increase curvature you can increase regularization or rescale features.\n"
    "It is also possible that you have reached numerical accuracy\n"
    "and further decrease in the objective cannot be reliably detected.\n";

void zero_derivative(vw& all) { all.weights.set_zero(W_GT); }

void zero_preconditioner(vw& all) { all.weights.set_zero(W_COND); }

void reset_state(vw& all, bfgs& b, bool zero)
{
  b.lastj = b.origin = 0;
  b.loss_sum = b.previous_loss_sum = 0.;
  b.importance_weight_sum = 0.;
  b.curvature = 0.;
  b.first_pass = true;
  b.gradient_pass = true;
  b.preconditioner_pass = true;
  if (zero)
  {
    zero_derivative(all);
    zero_preconditioner(all);
  }
}

// w[0] = weight
// w[1] = accumulated first derivative
// w[2] = step direction
// w[3] = preconditioner

constexpr bool test_example(example& ec) noexcept { return ec.l.simple.label == FLT_MAX; }

float bfgs_predict(vw& all, example& ec)
{
  ec.partial_prediction = GD::inline_predict(all, ec);
  return GD::finalize_prediction(all.sd, all.logger, ec.partial_prediction);
}

inline void add_grad(float& d, float f, float& fw) { (&fw)[W_GT] += d * f; }

float predict_and_gradient(vw& all, example& ec)
{
  float fp = bfgs_predict(all, ec);
  label_data& ld = ec.l.simple;
  all.set_minmax(all.sd, ld.label);

  float loss_grad = all.loss->first_derivative(all.sd, fp, ld.label) * ec.weight;
  GD::foreach_feature<float, add_grad>(all, ec, loss_grad);

  return fp;
}

inline void add_precond(float& d, float f, float& fw) { (&fw)[W_COND] += d * f * f; }

void update_preconditioner(vw& all, example& ec)
{
  float curvature = all.loss->second_derivative(all.sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
  GD::foreach_feature<float, add_precond>(all, ec, curvature);
}

inline void add_DIR(float& p, const float fx, float& fw) { p += (&fw)[W_DIR] * fx; }

float dot_with_direction(vw& all, example& ec)
{
  const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
  float temp = simple_red_features.initial;
  GD::foreach_feature<float, add_DIR>(all, ec, temp);
  return temp;
}

template <class T>
double regularizer_direction_magnitude(vw& /* all */, bfgs& b, double regularizer, T& weights)
{
  double ret = 0.;
  if (b.regularizers == nullptr)
    for (typename T::iterator iter = weights.begin(); iter != weights.end(); ++iter)
      ret += regularizer * (&(*iter))[W_DIR] * (&(*iter))[W_DIR];

  else
  {
    for (typename T::iterator iter = weights.begin(); iter != weights.end(); ++iter)
      ret += ((double)b.regularizers[2 * (iter.index() >> weights.stride_shift())]) * (&(*iter))[W_DIR] *
          (&(*iter))[W_DIR];
  }
  return ret;
}

double regularizer_direction_magnitude(vw& all, bfgs& b, float regularizer)
{
  // compute direction magnitude
  double ret = 0.;

  if (regularizer == 0.) return ret;

  if (all.weights.sparse)
    return regularizer_direction_magnitude(all, b, regularizer, all.weights.sparse_weights);
  else
    return regularizer_direction_magnitude(all, b, regularizer, all.weights.dense_weights);
}

template <class T>
float direction_magnitude(vw& /* all */, T& weights)
{
  // compute direction magnitude
  double ret = 0.;
  for (typename T::iterator iter = weights.begin(); iter != weights.end(); ++iter)
    ret += ((double)(&(*iter))[W_DIR]) * (&(*iter))[W_DIR];

  return static_cast<float>(ret);
}

float direction_magnitude(vw& all)
{
  // compute direction magnitude
  if (all.weights.sparse)
    return direction_magnitude(all, all.weights.sparse_weights);
  else
    return direction_magnitude(all, all.weights.dense_weights);
}

template <class T>
void bfgs_iter_start(vw& all, bfgs& b, float* mem, int& lastj, double importance_weight_sum, int& origin, T& weights)
{
  double g1_Hg1 = 0.;
  double g1_g1 = 0.;

  origin = 0;
  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    float* mem1 = mem + (w.index() >> weights.stride_shift()) * b.mem_stride;
    if (b.m > 0) mem1[(MEM_XT + origin) % b.mem_stride] = (&(*w))[W_XT];
    mem1[(MEM_GT + origin) % b.mem_stride] = (&(*w))[W_GT];
    g1_Hg1 += ((double)(&(*w))[W_GT]) * ((&(*w))[W_GT]) * ((&(*w))[W_COND]);
    g1_g1 += ((double)((&(*w))[W_GT])) * ((&(*w))[W_GT]);
    (&(*w))[W_DIR] = -(&(*w))[W_COND] * ((&(*w))[W_GT]);
    ((&(*w))[W_GT]) = 0;
  }
  lastj = 0;
  if (!all.logger.quiet)
    fprintf(stderr, "%-10.5f\t%-10.5f\t%-10s\t%-10s\t%-10s\t", g1_g1 / (importance_weight_sum * importance_weight_sum),
        g1_Hg1 / importance_weight_sum, "", "", "");
}

void bfgs_iter_start(vw& all, bfgs& b, float* mem, int& lastj, double importance_weight_sum, int& origin)
{
  if (all.weights.sparse)
    bfgs_iter_start(all, b, mem, lastj, importance_weight_sum, origin, all.weights.sparse_weights);
  else
    bfgs_iter_start(all, b, mem, lastj, importance_weight_sum, origin, all.weights.dense_weights);
}

template <class T>
void bfgs_iter_middle(vw& all, bfgs& b, float* mem, double* rho, double* alpha, int& lastj, int& origin, T& weights)
{
  float* mem0 = mem;
  // implement conjugate gradient
  if (b.m == 0)
  {
    double g_Hy = 0.;
    double g_Hg = 0.;
    double y = 0.;

    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
      y = (&(*w))[W_GT] - mem[(MEM_GT + origin) % b.mem_stride];
      g_Hy += ((double)(&(*w))[W_GT]) * ((&(*w))[W_COND]) * y;
      g_Hg += (static_cast<double>(mem[(MEM_GT + origin) % b.mem_stride])) * ((&(*w))[W_COND]) *
          mem[(MEM_GT + origin) % b.mem_stride];
    }

    float beta = static_cast<float>(g_Hy / g_Hg);

    if (beta < 0.f || std::isnan(beta)) beta = 0.f;

    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
      mem[(MEM_GT + origin) % b.mem_stride] = (&(*w))[W_GT];

      (&(*w))[W_DIR] *= beta;
      (&(*w))[W_DIR] -= ((&(*w))[W_COND]) * ((&(*w))[W_GT]);
      (&(*w))[W_GT] = 0;
    }
    // TODO: spdlog can't print partial log lines. Figure out how to handle this..
    if (!all.logger.quiet) fprintf(stderr, "%f\t", beta);
    return;
  }
  else
  {
    if (!all.logger.quiet) fprintf(stderr, "%-10s\t", "");
  }

  // implement bfgs
  double y_s = 0.;
  double y_Hy = 0.;
  double s_q = 0.;

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    float* mem1 = mem + (w.index() >> weights.stride_shift()) * b.mem_stride;
    mem1[(MEM_YT + origin) % b.mem_stride] = (&(*w))[W_GT] - mem1[(MEM_GT + origin) % b.mem_stride];
    mem1[(MEM_ST + origin) % b.mem_stride] = (&(*w))[W_XT] - mem1[(MEM_XT + origin) % b.mem_stride];
    (&(*w))[W_DIR] = (&(*w))[W_GT];
    y_s += (static_cast<double>(mem1[(MEM_YT + origin) % b.mem_stride])) * mem1[(MEM_ST + origin) % b.mem_stride];
    y_Hy += (static_cast<double>(mem1[(MEM_YT + origin) % b.mem_stride])) * mem1[(MEM_YT + origin) % b.mem_stride] *
        ((&(*w))[W_COND]);
    s_q += (static_cast<double>(mem1[(MEM_ST + origin) % b.mem_stride])) * ((&(*w))[W_GT]);
  }

  if (y_s <= 0. || y_Hy <= 0.) throw curv_ex;
  rho[0] = 1 / y_s;

  float gamma = static_cast<float>(y_s / y_Hy);

  for (int j = 0; j < lastj; j++)
  {
    alpha[j] = rho[j] * s_q;
    s_q = 0.;
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
      (&(*w))[W_DIR] -= static_cast<float>(alpha[j]) * mem[(2 * j + MEM_YT + origin) % b.mem_stride];
      s_q += (static_cast<double>(mem[(2 * j + 2 + MEM_ST + origin) % b.mem_stride])) * ((&(*w))[W_DIR]);
    }
  }

  alpha[lastj] = rho[lastj] * s_q;
  double y_r = 0.;

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
    (&(*w))[W_DIR] -= static_cast<float>(alpha[lastj]) * mem[(2 * lastj + MEM_YT + origin) % b.mem_stride];
    (&(*w))[W_DIR] *= gamma * ((&(*w))[W_COND]);
    y_r += (static_cast<double>(mem[(2 * lastj + MEM_YT + origin) % b.mem_stride])) * ((&(*w))[W_DIR]);
  }

  double coef_j;

  for (int j = lastj; j > 0; j--)
  {
    coef_j = alpha[j] - rho[j] * y_r;
    y_r = 0.;
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
      (&(*w))[W_DIR] += static_cast<float>(coef_j) * mem[(2 * j + MEM_ST + origin) % b.mem_stride];
      y_r += (static_cast<double>(mem[(2 * j - 2 + MEM_YT + origin) % b.mem_stride])) * ((&(*w))[W_DIR]);
    }
  }

  coef_j = alpha[0] - rho[0] * y_r;
  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
    (&(*w))[W_DIR] = -(&(*w))[W_DIR] - static_cast<float>(coef_j) * mem[(MEM_ST + origin) % b.mem_stride];
  }

  /*********************
  ** shift
  ********************/

  lastj = (lastj < b.m - 1) ? lastj + 1 : b.m - 1;
  origin = (origin + b.mem_stride - 2) % b.mem_stride;

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    mem = mem0 + (w.index() >> weights.stride_shift()) * b.mem_stride;
    mem[(MEM_GT + origin) % b.mem_stride] = (&(*w))[W_GT];
    mem[(MEM_XT + origin) % b.mem_stride] = (&(*w))[W_XT];
    (&(*w))[W_GT] = 0;
  }
  for (int j = lastj; j > 0; j--) rho[j] = rho[j - 1];
}

void bfgs_iter_middle(vw& all, bfgs& b, float* mem, double* rho, double* alpha, int& lastj, int& origin)
{
  if (all.weights.sparse)
    bfgs_iter_middle(all, b, mem, rho, alpha, lastj, origin, all.weights.sparse_weights);
  else
    bfgs_iter_middle(all, b, mem, rho, alpha, lastj, origin, all.weights.dense_weights);
}

template <class T>
double wolfe_eval(vw& all, bfgs& b, float* mem, double loss_sum, double previous_loss_sum, double step_size,
    double importance_weight_sum, int& origin, double& wolfe1, T& weights)
{
  double g0_d = 0.;
  double g1_d = 0.;
  double g1_Hg1 = 0.;
  double g1_g1 = 0.;

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    float* mem1 = mem + (w.index() >> weights.stride_shift()) * b.mem_stride;
    g0_d += (static_cast<double>(mem1[(MEM_GT + origin) % b.mem_stride])) * ((&(*w))[W_DIR]);
    g1_d += ((double)(&(*w))[W_GT]) * (&(*w))[W_DIR];
    g1_Hg1 += ((double)(&(*w))[W_GT]) * (&(*w))[W_GT] * ((&(*w))[W_COND]);
    g1_g1 += ((double)(&(*w))[W_GT]) * (&(*w))[W_GT];
  }

  wolfe1 = (loss_sum - previous_loss_sum) / (step_size * g0_d);
  double wolfe2 = g1_d / g0_d;
  // double new_step_cross = (loss_sum-previous_loss_sum-g1_d*step)/(g0_d-g1_d);

  if (!all.logger.quiet)
    fprintf(stderr, "%-10.5f\t%-10.5f\t%s%-10f\t%-10f\t", g1_g1 / (importance_weight_sum * importance_weight_sum),
        g1_Hg1 / importance_weight_sum, " ", wolfe1, wolfe2);
  return 0.5 * step_size;
}

double wolfe_eval(vw& all, bfgs& b, float* mem, double loss_sum, double previous_loss_sum, double step_size,
    double importance_weight_sum, int& origin, double& wolfe1)
{
  if (all.weights.sparse)
    return wolfe_eval(all, b, mem, loss_sum, previous_loss_sum, step_size, importance_weight_sum, origin, wolfe1,
        all.weights.sparse_weights);
  else
    return wolfe_eval(all, b, mem, loss_sum, previous_loss_sum, step_size, importance_weight_sum, origin, wolfe1,
        all.weights.dense_weights);
}

template <class T>
double add_regularization(vw& all, bfgs& b, float regularization, T& weights)
{
  // compute the derivative difference
  double ret = 0.;

  if (b.regularizers == nullptr)
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      (&(*w))[W_GT] += regularization * (*w);
      ret += 0.5 * regularization * (*w) * (*w);
    }
  else
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      uint64_t i = w.index() >> weights.stride_shift();
      weight delta_weight = *w - b.regularizers[2 * i + 1];
      (&(*w))[W_GT] += b.regularizers[2 * i] * delta_weight;
      ret += 0.5 * b.regularizers[2 * i] * delta_weight * delta_weight;
    }

  // if we're not regularizing the intercept term, then subtract it off from the result above
  // when accessing weights[constant], always use weights.strided_index(constant)
  if (all.no_bias)
  {
    if (b.regularizers == nullptr)
    {
      (&weights.strided_index(constant))[W_GT] -= regularization * (weights.strided_index(constant));
      ret -= 0.5 * regularization * (weights.strided_index(constant)) * (weights.strided_index(constant));
    }
    else
    {
      uint64_t i = constant >> weights.stride_shift();
      weight delta_weight = (weights.strided_index(constant)) - b.regularizers[2 * i + 1];
      (&weights.strided_index(constant))[W_GT] -= b.regularizers[2 * i] * delta_weight;
      ret -= 0.5 * b.regularizers[2 * i] * delta_weight * delta_weight;
    }
  }

  return ret;
}

double add_regularization(vw& all, bfgs& b, float regularization)
{
  if (all.weights.sparse)
    return add_regularization(all, b, regularization, all.weights.sparse_weights);
  else
    return add_regularization(all, b, regularization, all.weights.dense_weights);
}

template <class T>
void finalize_preconditioner(vw& /* all */, bfgs& b, float regularization, T& weights)
{
  float max_hessian = 0.f;

  if (b.regularizers == nullptr)
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      (&(*w))[W_COND] += regularization;
      if ((&(*w))[W_COND] > max_hessian) max_hessian = (&(*w))[W_COND];
      if ((&(*w))[W_COND] > 0) (&(*w))[W_COND] = 1.f / (&(*w))[W_COND];
    }
  else
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      (&(*w))[W_COND] += b.regularizers[2 * (w.index() >> weights.stride_shift())];
      if ((&(*w))[W_COND] > max_hessian) max_hessian = (&(*w))[W_COND];
      if ((&(*w))[W_COND] > 0) (&(*w))[W_COND] = 1.f / (&(*w))[W_COND];
    }

  float max_precond = (max_hessian == 0.f) ? 0.f : max_precond_ratio / max_hessian;

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    if (std::isinf((&(*w))[W_COND]) || (&(*w))[W_COND] > max_precond) (&(*w))[W_COND] = max_precond;
  }
}
void finalize_preconditioner(vw& all, bfgs& b, float regularization)
{
  if (all.weights.sparse)
    finalize_preconditioner(all, b, regularization, all.weights.sparse_weights);
  else
    finalize_preconditioner(all, b, regularization, all.weights.dense_weights);
}

template <class T>
void preconditioner_to_regularizer(vw& all, bfgs& b, float regularization, T& weights)
{
  uint32_t length = 1 << all.num_bits;

  if (b.regularizers == nullptr)
  {
    b.regularizers = calloc_or_throw<weight>(2 * length);

    if (b.regularizers == nullptr) THROW("Failed to allocate weight array: try decreasing -b <bits>");

    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      uint64_t i = w.index() >> weights.stride_shift();
      b.regularizers[2 * i] = regularization;
      if ((&(*w))[W_COND] > 0.f) b.regularizers[2 * i] += 1.f / (&(*w))[W_COND];
    }
  }
  else
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      if ((&(*w))[W_COND] > 0.f) b.regularizers[2 * (w.index() >> weights.stride_shift())] += 1.f / (&(*w))[W_COND];
    }

  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    b.regularizers[2 * (w.index() >> weights.stride_shift()) + 1] = *w;
}
void preconditioner_to_regularizer(vw& all, bfgs& b, float regularization)
{
  if (all.weights.sparse)
    preconditioner_to_regularizer(all, b, regularization, all.weights.sparse_weights);
  else
    preconditioner_to_regularizer(all, b, regularization, all.weights.dense_weights);
}

template <class T>
void regularizer_to_weight(vw& /* all */, bfgs& b, T& weights)
{
  if (b.regularizers != nullptr)
  {
    for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
    {
      uint64_t i = w.index() >> weights.stride_shift();
      (&(*w))[W_COND] = b.regularizers[2 * i];
      *w = b.regularizers[2 * i + 1];
    }
  }
}

void regularizer_to_weight(vw& all, bfgs& b)
{
  if (all.weights.sparse)
    regularizer_to_weight(all, b, all.weights.sparse_weights);
  else
    regularizer_to_weight(all, b, all.weights.dense_weights);
}

void zero_state(vw& all)
{
  all.weights.set_zero(W_GT);
  all.weights.set_zero(W_DIR);
  all.weights.set_zero(W_COND);
}

template <class T>
double derivative_in_direction(vw& /* all */, bfgs& b, float* mem, int& origin, T& weights)
{
  double ret = 0.;
  for (typename T::iterator w = weights.begin(); w != weights.end(); ++w)
  {
    float* mem1 = mem + (w.index() >> weights.stride_shift()) * b.mem_stride;
    ret += (static_cast<double>(mem1[(MEM_GT + origin) % b.mem_stride])) * (&(*w))[W_DIR];
  }
  return ret;
}

double derivative_in_direction(vw& all, bfgs& b, float* mem, int& origin)
{
  if (all.weights.sparse)
    return derivative_in_direction(all, b, mem, origin, all.weights.sparse_weights);
  else
    return derivative_in_direction(all, b, mem, origin, all.weights.dense_weights);
}

template <class T>
void update_weight(vw& /* all */, float step_size, T& w)
{
  for (typename T::iterator iter = w.begin(); iter != w.end(); ++iter)
    (&(*iter))[W_XT] += step_size * (&(*iter))[W_DIR];
}

void update_weight(vw& all, float step_size)
{
  if (all.weights.sparse)
    update_weight(all, step_size, all.weights.sparse_weights);
  else
    update_weight(all, step_size, all.weights.dense_weights);
}

int process_pass(vw& all, bfgs& b)
{
  int status = LEARN_OK;

  finalize_preconditioner(all, b, all.l2_lambda);
  /********************************************************************/
  /* A) FIRST PASS FINISHED: INITIALIZE FIRST LINE SEARCH *************/
  /********************************************************************/
  if (b.first_pass)
  {
    if (all.all_reduce != nullptr)
    {
      accumulate(all, all.weights, W_COND);  // Accumulate preconditioner
      float temp = static_cast<float>(b.importance_weight_sum);
      b.importance_weight_sum = accumulate_scalar(all, temp);
    }
    // finalize_preconditioner(all, b, all.l2_lambda);
    if (all.all_reduce != nullptr)
    {
      float temp = static_cast<float>(b.loss_sum);
      b.loss_sum = accumulate_scalar(all, temp);  // Accumulate loss_sums
      accumulate(all, all.weights, 1);            // Accumulate gradients from all nodes
    }
    if (all.l2_lambda > 0.) b.loss_sum += add_regularization(all, b, all.l2_lambda);
    if (!all.logger.quiet)
      fprintf(stderr, "%2lu %-10.5f\t", static_cast<long unsigned int>(b.current_pass) + 1,
          b.loss_sum / b.importance_weight_sum);

    b.previous_loss_sum = b.loss_sum;
    b.loss_sum = 0.;
    b.example_number = 0;
    b.curvature = 0;
    bfgs_iter_start(all, b, b.mem, b.lastj, b.importance_weight_sum, b.origin);
    if (b.first_hessian_on)
    {
      b.gradient_pass = false;  // now start computing curvature
    }
    else
    {
      b.step_size = 0.5;
      float d_mag = direction_magnitude(all);
      b.t_end_global = std::chrono::system_clock::now();
      b.net_time = static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(b.t_end_global - b.t_start_global).count());
      if (!all.logger.quiet) fprintf(stderr, "%-10s\t%-10.5f\t%-.5f\n", "", d_mag, b.step_size);
      b.predictions.clear();
      update_weight(all, b.step_size);
    }
  }
  else
      /********************************************************************/
      /* B) GRADIENT CALCULATED *******************************************/
      /********************************************************************/
      if (b.gradient_pass)  // We just finished computing all gradients
  {
    if (all.all_reduce != nullptr)
    {
      float t = static_cast<float>(b.loss_sum);
      b.loss_sum = accumulate_scalar(all, t);  // Accumulate loss_sums
      accumulate(all, all.weights, 1);         // Accumulate gradients from all nodes
    }
    if (all.l2_lambda > 0.) b.loss_sum += add_regularization(all, b, all.l2_lambda);
    if (!all.logger.quiet)
    {
      if (!all.holdout_set_off && b.current_pass >= 1)
      {
        if (all.sd->holdout_sum_loss_since_last_pass == 0. && all.sd->weighted_holdout_examples_since_last_pass == 0.)
        {
          fprintf(stderr, "%2lu ", static_cast<long unsigned int>(b.current_pass) + 1);
          fprintf(stderr, "h unknown    ");
        }
        else
          fprintf(stderr, "%2lu h%-10.5f\t", static_cast<long unsigned int>(b.current_pass) + 1,
              all.sd->holdout_sum_loss_since_last_pass / all.sd->weighted_holdout_examples_since_last_pass);
      }
      else
        fprintf(stderr, "%2lu %-10.5f\t", static_cast<long unsigned int>(b.current_pass) + 1,
            b.loss_sum / b.importance_weight_sum);
    }
    double wolfe1;
    double new_step = wolfe_eval(
        all, b, b.mem, b.loss_sum, b.previous_loss_sum, b.step_size, b.importance_weight_sum, b.origin, wolfe1);

    /********************************************************************/
    /* B0) DERIVATIVE ZERO: MINIMUM FOUND *******************************/
    /********************************************************************/
    if (std::isnan(static_cast<float>(wolfe1)))
    {
      fprintf(stderr, "\n");
      fprintf(stdout, "Derivative 0 detected.\n");
      b.step_size = 0.0;
      status = LEARN_CONV;
    }
    /********************************************************************/
    /* B1) LINE SEARCH FAILED *******************************************/
    /********************************************************************/
    else if (b.backstep_on && (wolfe1 < b.wolfe1_bound || b.loss_sum > b.previous_loss_sum))
    {
      // curvature violated, or we stepped too far last time: step back
      b.t_end_global = std::chrono::system_clock::now();
      b.net_time = static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(b.t_end_global - b.t_start_global).count());
      float ratio = (b.step_size == 0.f) ? 0.f : static_cast<float>(new_step) / b.step_size;
      if (!all.logger.quiet) fprintf(stderr, "%-10s\t%-10s\t(revise x %.1f)\t%-.5f\n", "", "", ratio, new_step);
      b.predictions.clear();
      update_weight(all, static_cast<float>(-b.step_size + new_step));
      b.step_size = static_cast<float>(new_step);
      zero_derivative(all);
      b.loss_sum = 0.;
    }

    /********************************************************************/
    /* B2) LINE SEARCH SUCCESSFUL OR DISABLED          ******************/
    /*     DETERMINE NEXT SEARCH DIRECTION             ******************/
    /********************************************************************/
    else
    {
      double rel_decrease = (b.previous_loss_sum - b.loss_sum) / b.previous_loss_sum;
      if (!std::isnan(static_cast<float>(rel_decrease)) && b.backstep_on && fabs(rel_decrease) < b.rel_threshold)
      {
        fprintf(stdout,
            "\nTermination condition reached in pass %ld: decrease in loss less than %.3f%%.\n"
            "If you want to optimize further, decrease termination threshold.\n",
            static_cast<long int>(b.current_pass) + 1, b.rel_threshold * 100.0);
        status = LEARN_CONV;
      }
      b.previous_loss_sum = b.loss_sum;
      b.loss_sum = 0.;
      b.example_number = 0;
      b.curvature = 0;
      b.step_size = 1.0;

      try
      {
        bfgs_iter_middle(all, b, b.mem, b.rho, b.alpha, b.lastj, b.origin);
      }
      catch (const curv_exception&)
      {
        fprintf(stdout, "In bfgs_iter_middle: %s", curv_message);
        b.step_size = 0.0;
        status = LEARN_CURV;
      }

      if (all.hessian_on)
      {
        b.gradient_pass = false;  // now start computing curvature
      }
      else
      {
        float d_mag = direction_magnitude(all);
        b.t_end_global = std::chrono::system_clock::now();
        b.net_time = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(b.t_end_global - b.t_start_global).count());
        if (!all.logger.quiet) fprintf(stderr, "%-10s\t%-10.5f\t%-.5f\n", "", d_mag, b.step_size);
        b.predictions.clear();
        update_weight(all, b.step_size);
      }
    }
  }

  /********************************************************************/
  /* C) NOT FIRST PASS, CURVATURE CALCULATED **************************/
  /********************************************************************/
  else  // just finished all second gradients
  {
    if (all.all_reduce != nullptr)
    {
      float t = static_cast<float>(b.curvature);
      b.curvature = accumulate_scalar(all, t);  // Accumulate curvatures
    }
    if (all.l2_lambda > 0.) b.curvature += regularizer_direction_magnitude(all, b, all.l2_lambda);
    float dd = static_cast<float>(derivative_in_direction(all, b, b.mem, b.origin));
    if (b.curvature == 0. && dd != 0.)
    {
      fprintf(stdout, "%s", curv_message);
      b.step_size = 0.0;
      status = LEARN_CURV;
    }
    else if (dd == 0.)
    {
      fprintf(stdout, "Derivative 0 detected.\n");
      b.step_size = 0.0;
      status = LEARN_CONV;
    }
    else
      b.step_size = -dd / static_cast<float>(b.curvature);

    float d_mag = direction_magnitude(all);

    b.predictions.clear();
    update_weight(all, b.step_size);
    b.t_end_global = std::chrono::system_clock::now();
    b.net_time = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(b.t_end_global - b.t_start_global).count());

    if (!all.logger.quiet)
      fprintf(stderr, "%-10.5f\t%-10.5f\t%-.5f\n", b.curvature / b.importance_weight_sum, d_mag, b.step_size);
    b.gradient_pass = true;
  }  // now start computing derivatives.
  b.current_pass++;
  b.first_pass = false;
  b.preconditioner_pass = false;

  if (b.output_regularizer)  // need to accumulate and place the regularizer.
  {
    if (all.all_reduce != nullptr) accumulate(all, all.weights, W_COND);  // Accumulate preconditioner
    // preconditioner_to_regularizer(all, b, all.l2_lambda);
  }
  b.t_end_global = std::chrono::system_clock::now();
  b.net_time = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(b.t_end_global - b.t_start_global).count());

  if (all.save_per_pass) save_predictor(all, all.final_regressor_name, b.current_pass);
  return status;
}

void process_example(vw& all, bfgs& b, example& ec)
{
  label_data& ld = ec.l.simple;
  if (b.first_pass) b.importance_weight_sum += ec.weight;

  /********************************************************************/
  /* I) GRADIENT CALCULATION ******************************************/
  /********************************************************************/
  if (b.gradient_pass)
  {
    ec.pred.scalar = predict_and_gradient(all, ec);  // w[0] & w[1]
    ec.loss = all.loss->getLoss(all.sd, ec.pred.scalar, ld.label) * ec.weight;
    b.loss_sum += ec.loss;
    b.predictions.push_back(ec.pred.scalar);
  }
  /********************************************************************/
  /* II) CURVATURE CALCULATION ****************************************/
  /********************************************************************/
  else  // computing curvature
  {
    float d_dot_x = dot_with_direction(all, ec);   // w[2]
    if (b.example_number >= b.predictions.size())  // Make things safe in case example source is strange.
      b.example_number = b.predictions.size() - 1;
    ec.pred.scalar = b.predictions[b.example_number];
    ec.partial_prediction = b.predictions[b.example_number];
    ec.loss = all.loss->getLoss(all.sd, ec.pred.scalar, ld.label) * ec.weight;
    float sd = all.loss->second_derivative(all.sd, b.predictions[b.example_number++], ld.label);
    b.curvature += (static_cast<double>(d_dot_x)) * d_dot_x * sd * ec.weight;
  }
  ec.updated_prediction = ec.pred.scalar;

  if (b.preconditioner_pass) update_preconditioner(all, ec);  // w[3]
}

void end_pass(bfgs& b)
{
  vw* all = b.all;

  if (b.current_pass <= b.final_pass)
  {
    if (b.current_pass < b.final_pass)
    {
      int status = process_pass(*all, b);

      // reaching the max number of passes regardless of convergence
      if (b.final_pass == b.current_pass)
      {
        *(b.all->trace_message) << "Maximum number of passes reached. ";
        if (!b.output_regularizer)
          *(b.all->trace_message) << "If you want to optimize further, increase the number of passes\n";
        if (b.output_regularizer)
        {
          *(b.all->trace_message) << "\nRegular model file has been created. ";
          *(b.all->trace_message) << "Output feature regularizer file is created only when the convergence is reached. "
                                     "Try increasing the number of passes for convergence\n";
          b.output_regularizer = false;
        }
      }

      // attain convergence before reaching max iterations
      if (status != LEARN_OK && b.final_pass > b.current_pass) { b.final_pass = b.current_pass; }
      else
      {
        // Not converged yet.
        // Reset preconditioner to zero so that it is correctly recomputed in the next pass
        zero_preconditioner(*all);
      }
      if (!all->holdout_set_off)
      {
        if (summarize_holdout_set(*all, b.no_win_counter)) finalize_regressor(*all, all->final_regressor_name);
        if (b.early_stop_thres == b.no_win_counter)
        {
          set_done(*all);
          *(b.all->trace_message) << "Early termination reached w.r.t. holdout set error";
        }
      }
      if (b.final_pass == b.current_pass)
      {
        finalize_regressor(*all, all->final_regressor_name);
        set_done(*all);
      }
    }
    else  // reaching convergence in the previous pass
      b.current_pass++;
  }
}

// placeholder
template <bool audit>
void predict(bfgs& b, base_learner&, example& ec)
{
  vw* all = b.all;
  ec.pred.scalar = bfgs_predict(*all, ec);
  if (audit) GD::print_audit_features(*(b.all), ec);
}

template <bool audit>
void learn(bfgs& b, base_learner& base, example& ec)
{
  vw* all = b.all;

  if (b.current_pass <= b.final_pass)
  {
    if (test_example(ec))
      predict<audit>(b, base, ec);
    else
      process_example(*all, b, ec);
  }
}

void save_load_regularizer(vw& all, bfgs& b, io_buf& model_file, bool read, bool text)
{
  int c = 0;
  uint32_t length = 2 * (1 << all.num_bits);
  uint32_t i = 0;
  size_t brw = 1;

  if (b.output_regularizer && !read) preconditioner_to_regularizer(*(b.all), b, b.all->l2_lambda);

  do
  {
    brw = 1;
    weight* v;
    if (read)
    {
      c++;
      brw = model_file.bin_read_fixed(reinterpret_cast<char*>(&i), sizeof(i), "");
      if (brw > 0)
      {
        assert(i < length);
        v = &(b.regularizers[i]);
        brw += model_file.bin_read_fixed(reinterpret_cast<char*>(v), sizeof(*v), "");
      }
    }
    else  // write binary or text
    {
      v = &(b.regularizers[i]);
      if (*v != 0.)
      {
        c++;
        std::stringstream msg;
        msg << i;
        brw = bin_text_write_fixed(model_file, reinterpret_cast<char*>(&i), sizeof(i), msg, text);

        msg << ":" << *v << "\n";
        brw += bin_text_write_fixed(model_file, reinterpret_cast<char*>(v), sizeof(*v), msg, text);
      }
    }
    if (!read) i++;
  } while ((!read && i < length) || (read && brw > 0));

  if (read) regularizer_to_weight(all, b);
}

void save_load(bfgs& b, io_buf& model_file, bool read, bool text)
{
  vw* all = b.all;

  uint32_t length = 1 << all->num_bits;

  if (read)
  {
    initialize_regressor(*all);
    if (all->per_feature_regularizer_input != "")
    {
      b.regularizers = calloc_or_throw<weight>(2 * length);
      if (b.regularizers == nullptr) THROW("Failed to allocate regularizers array: try decreasing -b <bits>");
    }
    int m = b.m;

    b.mem_stride = (m == 0) ? CG_EXTRA : 2 * m;
    b.mem = calloc_or_throw<float>(all->length() * b.mem_stride);
    b.rho = calloc_or_throw<double>(m);
    b.alpha = calloc_or_throw<double>(m);

    uint32_t stride_shift = all->weights.stride_shift();

    if (!all->logger.quiet)
      std::cerr << "m = " << m << std::endl
                << "Allocated "
                << (static_cast<long unsigned int>(all->length()) *
                           (sizeof(float) * (b.mem_stride) + (sizeof(weight) << stride_shift)) >>
                       20)
                << "M for weights and mem" << std::endl;

    b.net_time = 0.0;
    b.t_start_global = std::chrono::system_clock::now();

    if (!all->logger.quiet)
    {
      const char* header_fmt = "%2s %-10s\t%-10s\t%-10s\t %-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-s\n";
      fprintf(stderr, header_fmt, "##", "avg. loss", "der. mag.", "d. m. cond.", "wolfe1", "wolfe2", "mix fraction",
          "curvature", "dir. magnitude", "step size");
      std::cerr.precision(5);
    }

    if (b.regularizers != nullptr) all->l2_lambda = 1;  // To make sure we are adding the regularization
    b.output_regularizer = (all->per_feature_regularizer_output != "" || all->per_feature_regularizer_text != "");
    reset_state(*all, b, false);
  }

  // bool reg_vector = b.output_regularizer || all->per_feature_regularizer_input.length() > 0;
  bool reg_vector = (b.output_regularizer && !read) || (all->per_feature_regularizer_input.length() > 0 && read);

  if (model_file.num_files() > 0)
  {
    std::stringstream msg;
    msg << ":" << reg_vector << "\n";
    bin_text_read_write_fixed(
        model_file, reinterpret_cast<char*>(&reg_vector), sizeof(reg_vector), "", read, msg, text);

    if (reg_vector)
      save_load_regularizer(*all, b, model_file, read, text);
    else
      GD::save_load_regressor(*all, model_file, read, text);
  }
}

void init_driver(bfgs& b) { b.backstep_on = true; }

base_learner* bfgs_setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  auto b = VW::make_unique<bfgs>();
  bool conjugate_gradient = false;
  bool bfgs_option = false;
  option_group_definition bfgs_outer_options("Conjugate Gradient options");
  bfgs_outer_options.add(make_option("conjugate_gradient", conjugate_gradient)
                             .keep()
                             .necessary()
                             .help("use conjugate gradient based optimization"));

  option_group_definition bfgs_inner_options("LBFGS and Conjugate Gradient options");
  bfgs_inner_options.add(
      make_option("bfgs", bfgs_option).keep().necessary().help("use conjugate gradient based optimization"));
  bfgs_inner_options.add(make_option("hessian_on", all.hessian_on).help("use second derivative in line search"));
  bfgs_inner_options.add(make_option("mem", b->m).default_value(15).help("memory in bfgs"));
  bfgs_inner_options.add(
      make_option("termination", b->rel_threshold).default_value(0.001f).help("Termination threshold"));

  if (!options.add_parse_and_check_necessary(bfgs_outer_options))
    if (!options.add_parse_and_check_necessary(bfgs_inner_options)) return nullptr;

  b->all = &all;
  b->wolfe1_bound = 0.01;
  b->first_hessian_on = true;
  b->first_pass = true;
  b->gradient_pass = true;
  b->preconditioner_pass = true;
  b->backstep_on = false;
  b->final_pass = all.numpasses;
  b->no_win_counter = 0;

  if (!all.holdout_set_off)
  {
    all.sd->holdout_best_loss = FLT_MAX;
    b->early_stop_thres = options.get_typed_option<size_t>("early_terminate").value();
  }

  if (b->m == 0) all.hessian_on = true;

  if (!all.logger.quiet)
  {
    if (b->m > 0)
      *(b->all->trace_message) << "enabling BFGS based optimization ";
    else
      *(b->all->trace_message) << "enabling conjugate gradient optimization via BFGS ";
    if (all.hessian_on)
      *(b->all->trace_message) << "with curvature calculation" << std::endl;
    else
      *(b->all->trace_message) << "**without** curvature calculation" << std::endl;
  }

  if (all.numpasses < 2 && all.training) THROW("you must make at least 2 passes to use BFGS");

  all.bfgs = true;
  all.weights.stride_shift(2);

  void (*learn_ptr)(bfgs&, base_learner&, example&) = nullptr;
  void (*predict_ptr)(bfgs&, base_learner&, example&) = nullptr;
  std::string learner_name;
  if (all.audit || all.hash_inv)
  {
    learn_ptr = learn<true>;
    predict_ptr = predict<true>;
    learner_name = all.get_setupfn_name(bfgs_setup) + "-audit";
  }
  else
  {
    learn_ptr = learn<false>;
    predict_ptr = predict<false>;
    learner_name = all.get_setupfn_name(bfgs_setup);
  }

  return make_base(*make_base_learner(
      std::move(b), learn_ptr, predict_ptr, learner_name, prediction_type_t::scalar, label_type_t::simple)
                        .set_params_per_weight(all.weights.stride())
                        .set_save_load(save_load)
                        .set_init_driver(init_driver)
                        .set_end_pass(end_pass)
                        .build());
}
