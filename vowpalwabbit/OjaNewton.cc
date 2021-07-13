// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cmath>
#include <string>
#include "gd.h"
#include "vw.h"
#include "rand48.h"
#include "reductions.h"
#include <cmath>
#include <memory>

using namespace VW::LEARNER;
using namespace VW::config;

#define NORM2 (m + 1)

struct oja_n_update_data
{
  struct OjaNewton* ON;
  float g;
  float sketch_cnt;
  float norm2_x;
  float* Zx;
  float* AZx;
  float* delta;
  float bdelta;
  float prediction;
};

struct OjaNewton
{
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  int m;
  int epoch_size;
  float alpha;
  int cnt;
  int t;

  float* ev;
  float* b;
  float* D;
  float** A;
  float** K;

  float* zv;
  float* vv;
  float* tmp;

  example** buffer;
  float* weight_buffer;
  struct oja_n_update_data data;

  float learning_rate_cnt;
  bool normalize;
  bool random_init;

  void initialize_Z(parameters& weights)
  {
    uint32_t length = 1 << all->num_bits;
    if (normalize)  // initialize normalization part
    {
      for (uint32_t i = 0; i < length; i++) (&(weights.strided_index(i)))[NORM2] = 0.1f;
    }
    if (!random_init)
    {
      // simple initialization
      for (int i = 1; i <= m; i++) (&(weights.strided_index(i)))[i] = 1.f;
    }
    else
    {
      // more complicated initialization: orthgonal basis of a random matrix

      const double PI2 = 2.f * 3.1415927f;

      for (uint32_t i = 0; i < length; i++)
      {
        weight& w = weights.strided_index(i);
        float r1, r2;
        for (int j = 1; j <= m; j++)
        {
          // box-muller tranform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
          // redraw until r1 should be strictly positive
          do
          {
            r1 = _random_state->get_and_update_random();
            r2 = _random_state->get_and_update_random();
          } while (r1 == 0.f);

          (&w)[j] = std::sqrt(-2.f * std::log(r1)) * static_cast<float>(cos(PI2 * r2));
        }
      }
    }

    // Gram-Schmidt
    for (int j = 1; j <= m; j++)
    {
      for (int k = 1; k <= j - 1; k++)
      {
        double temp = 0;

        for (uint32_t i = 0; i < length; i++)
          temp += (static_cast<double>((&(weights.strided_index(i)))[j])) * (&(weights.strided_index(i)))[k];
        for (uint32_t i = 0; i < length; i++)
          (&(weights.strided_index(i)))[j] -= static_cast<float>(temp) * (&(weights.strided_index(i)))[k];
      }
      double norm = 0;
      for (uint32_t i = 0; i < length; i++)
        norm += (static_cast<double>((&(weights.strided_index(i)))[j])) * (&(weights.strided_index(i)))[j];
      norm = std::sqrt(norm);
      for (uint32_t i = 0; i < length; i++) (&(weights.strided_index(i)))[j] /= static_cast<float>(norm);
    }
  }

  void compute_AZx()
  {
    for (int i = 1; i <= m; i++)
    {
      data.AZx[i] = 0;
      for (int j = 1; j <= i; j++) { data.AZx[i] += A[i][j] * data.Zx[j]; }
    }
  }

  void update_eigenvalues()
  {
    for (int i = 1; i <= m; i++)
    {
      float gamma = std::fmin(learning_rate_cnt / t, 1.f);
      float temp = data.AZx[i] * data.sketch_cnt;

      if (t == 1) { ev[i] = gamma * temp * temp; }
      else
      {
        ev[i] = (1 - gamma) * t * ev[i] / (t - 1) + gamma * t * temp * temp;
      }
    }
  }

  void compute_delta()
  {
    data.bdelta = 0;
    for (int i = 1; i <= m; i++)
    {
      float gamma = std::fmin(learning_rate_cnt / t, 1.f);

      // if different learning rates are used
      /*data.delta[i] = gamma * data.AZx[i] * data.sketch_cnt;
      for (int j = 1; j < i; j++) {
          data.delta[i] -= A[i][j] * data.delta[j];
      }
      data.delta[i] /= A[i][i];*/

      // if a same learning rate is used
      data.delta[i] = gamma * data.Zx[i] * data.sketch_cnt;

      data.bdelta += data.delta[i] * b[i];
    }
  }

  void update_K()
  {
    float temp = data.norm2_x * data.sketch_cnt * data.sketch_cnt;
    for (int i = 1; i <= m; i++)
    {
      for (int j = 1; j <= m; j++)
      {
        K[i][j] += data.delta[i] * data.Zx[j] * data.sketch_cnt;
        K[i][j] += data.delta[j] * data.Zx[i] * data.sketch_cnt;
        K[i][j] += data.delta[i] * data.delta[j] * temp;
      }
    }
  }

  void update_A()
  {
    for (int i = 1; i <= m; i++)
    {
      for (int j = 1; j < i; j++)
      {
        zv[j] = 0;
        for (int k = 1; k <= i; k++) { zv[j] += A[i][k] * K[k][j]; }
      }

      for (int j = 1; j < i; j++)
      {
        vv[j] = 0;
        for (int k = 1; k <= j; k++) { vv[j] += A[j][k] * zv[k]; }
      }

      for (int j = 1; j < i; j++)
      {
        for (int k = j; k < i; k++) { A[i][j] -= vv[k] * A[k][j]; }
      }

      float norm = 0;
      for (int j = 1; j <= i; j++)
      {
        float temp = 0;
        for (int k = 1; k <= i; k++) { temp += K[j][k] * A[i][k]; }
        norm += A[i][j] * temp;
      }
      norm = sqrtf(norm);

      for (int j = 1; j <= i; j++) { A[i][j] /= norm; }
    }
  }

  void update_b()
  {
    for (int j = 1; j <= m; j++)
    {
      float temp = 0;
      for (int i = j; i <= m; i++) { temp += ev[i] * data.AZx[i] * A[i][j] / (alpha * (alpha + ev[i])); }
      b[j] += temp * data.g;
    }
  }

  void update_D()
  {
    for (int j = 1; j <= m; j++)
    {
      float scale = std::fabs(A[j][j]);
      for (int i = j + 1; i <= m; i++) scale = std::fmin(std::fabs(A[i][j]), scale);
      if (scale < 1e-10) continue;
      for (int i = 1; i <= m; i++)
      {
        A[i][j] /= scale;
        K[j][i] *= scale;
        K[i][j] *= scale;
      }
      b[j] /= scale;
      D[j] *= scale;
      // printf("D[%d] = %f\n", j, D[j]);
    }
  }

  void check()
  {
    double max_norm = 0;
    for (int i = 1; i <= m; i++)
      for (int j = i; j <= m; j++) max_norm = fmax(max_norm, std::fabs(K[i][j]));
    // printf("|K| = %f\n", max_norm);
    if (max_norm < 1e7) return;

    // implicit -> explicit representation
    // printf("begin conversion: t = %d, norm(K) = %f\n", t, max_norm);

    // first step: K <- AKA'

    // K <- AK
    for (int j = 1; j <= m; j++)
    {
      memset(tmp, 0, sizeof(double) * (m + 1));

      for (int i = 1; i <= m; i++)
      {
        for (int h = 1; h <= m; h++) { tmp[i] += A[i][h] * K[h][j]; }
      }

      for (int i = 1; i <= m; i++) K[i][j] = tmp[i];
    }
    // K <- KA'
    for (int i = 1; i <= m; i++)
    {
      memset(tmp, 0, sizeof(double) * (m + 1));

      for (int j = 1; j <= m; j++)
        for (int h = 1; h <= m; h++) tmp[j] += K[i][h] * A[j][h];

      for (int j = 1; j <= m; j++) { K[i][j] = tmp[j]; }
    }

    // second step: w[0] <- w[0] + (DZ)'b, b <- 0.

    uint32_t length = 1 << all->num_bits;
    for (uint32_t i = 0; i < length; i++)
    {
      weight& w = all->weights.strided_index(i);
      for (int j = 1; j <= m; j++) w += (&w)[j] * b[j] * D[j];
    }

    memset(b, 0, sizeof(double) * (m + 1));

    // third step: Z <- ADZ, A, D <- Identity

    // double norm = 0;
    for (uint32_t i = 0; i < length; ++i)
    {
      memset(tmp, 0, sizeof(float) * (m + 1));
      weight& w = all->weights.strided_index(i);
      for (int j = 1; j <= m; j++)
      {
        for (int h = 1; h <= m; ++h) tmp[j] += A[j][h] * D[h] * (&w)[h];
      }
      for (int j = 1; j <= m; ++j)
      {
        // norm = std::max(norm, fabs(tmp[j]));
        (&w)[j] = tmp[j];
      }
    }
    // printf("|Z| = %f\n", norm);

    for (int i = 1; i <= m; i++)
    {
      memset(A[i], 0, sizeof(double) * (m + 1));
      D[i] = 1;
      A[i][i] = 1;
    }
  }

  ~OjaNewton()
  {
    free(ev);
    free(b);
    free(D);
    free(buffer);
    free(weight_buffer);
    free(zv);
    free(vv);
    free(tmp);
    if (A)
    {
      for (int i = 1; i <= m; i++)
      {
        free(A[i]);
        free(K[i]);
      }
    }

    free(A);
    free(K);

    free(data.Zx);
    free(data.AZx);
    free(data.delta);
  }
};

void keep_example(vw& all, OjaNewton& /* ON */, example& ec) { output_and_account_example(all, ec); }

void make_pred(oja_n_update_data& data, float x, float& wref)
{
  int m = data.ON->m;
  float* w = &wref;

  if (data.ON->normalize) { x /= std::sqrt(w[NORM2]); }

  data.prediction += w[0] * x;
  for (int i = 1; i <= m; i++) { data.prediction += w[i] * x * data.ON->D[i] * data.ON->b[i]; }
}

void predict(OjaNewton& ON, base_learner&, example& ec)
{
  ON.data.prediction = 0;
  GD::foreach_feature<oja_n_update_data, make_pred>(*ON.all, ec, ON.data);
  ec.partial_prediction = ON.data.prediction;
  ec.pred.scalar = GD::finalize_prediction(ON.all->sd, ON.all->logger, ec.partial_prediction);
}

void update_Z_and_wbar(oja_n_update_data& data, float x, float& wref)
{
  float* w = &wref;
  int m = data.ON->m;
  if (data.ON->normalize) x /= std::sqrt(w[NORM2]);
  float s = data.sketch_cnt * x;

  for (int i = 1; i <= m; i++) { w[i] += data.delta[i] * s / data.ON->D[i]; }
  w[0] -= s * data.bdelta;
}

void compute_Zx_and_norm(oja_n_update_data& data, float x, float& wref)
{
  float* w = &wref;
  int m = data.ON->m;
  if (data.ON->normalize) x /= std::sqrt(w[NORM2]);

  for (int i = 1; i <= m; i++) { data.Zx[i] += w[i] * x * data.ON->D[i]; }
  data.norm2_x += x * x;
}

void update_wbar_and_Zx(oja_n_update_data& data, float x, float& wref)
{
  float* w = &wref;
  int m = data.ON->m;
  if (data.ON->normalize) x /= std::sqrt(w[NORM2]);

  float g = data.g * x;

  for (int i = 1; i <= m; i++) { data.Zx[i] += w[i] * x * data.ON->D[i]; }
  w[0] -= g / data.ON->alpha;
}

void update_normalization(oja_n_update_data& data, float x, float& wref)
{
  float* w = &wref;
  int m = data.ON->m;

  w[NORM2] += x * x * data.g * data.g;
}

void learn(OjaNewton& ON, base_learner& base, example& ec)
{
  // predict
  predict(ON, base, ec);

  oja_n_update_data& data = ON.data;
  data.g = ON.all->loss->first_derivative(ON.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
  data.g /= 2;  // for half square loss

  if (ON.normalize) GD::foreach_feature<oja_n_update_data, update_normalization>(*ON.all, ec, data);

  ON.buffer[ON.cnt] = &ec;
  ON.weight_buffer[ON.cnt++] = data.g / 2;

  if (ON.cnt == ON.epoch_size)
  {
    for (int k = 0; k < ON.epoch_size; k++, ON.t++)
    {
      example& ex = *(ON.buffer[k]);
      data.sketch_cnt = ON.weight_buffer[k];

      data.norm2_x = 0;
      memset(data.Zx, 0, sizeof(float) * (ON.m + 1));
      GD::foreach_feature<oja_n_update_data, compute_Zx_and_norm>(*ON.all, ex, data);
      ON.compute_AZx();

      ON.update_eigenvalues();
      ON.compute_delta();

      ON.update_K();

      GD::foreach_feature<oja_n_update_data, update_Z_and_wbar>(*ON.all, ex, data);
    }

    ON.update_A();
    // ON.update_D();
  }

  memset(data.Zx, 0, sizeof(float) * (ON.m + 1));
  GD::foreach_feature<oja_n_update_data, update_wbar_and_Zx>(*ON.all, ec, data);
  ON.compute_AZx();

  ON.update_b();
  ON.check();

  if (ON.cnt == ON.epoch_size)
  {
    ON.cnt = 0;
    for (int k = 0; k < ON.epoch_size; k++) { VW::finish_example(*ON.all, *ON.buffer[k]); }
  }
}

void save_load(OjaNewton& ON, io_buf& model_file, bool read, bool text)
{
  vw& all = *ON.all;
  if (read)
  {
    initialize_regressor(all);
    ON.initialize_Z(all.weights);
  }

  if (model_file.num_files() > 0)
  {
    bool resume = all.save_resume;
    std::stringstream msg;
    msg << ":" << resume << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&resume), sizeof(resume), "", read, msg, text);

    double temp = 0.;
    if (resume)
      GD::save_load_online_state(all, model_file, read, text, temp);
    else
      GD::save_load_regressor(all, model_file, read, text);
  }
}

base_learner* OjaNewton_setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  auto ON = scoped_calloc_or_throw<OjaNewton>();

  bool oja_newton;
  float alpha_inverse;

  // These two are the only two boolean options that default to true. For now going to do this hack
  // as the infrastructure doesn't easily support this possibility at the same time providing the
  // ease of bool switches elsewhere. It seems that the switch behavior is more critical because
  // of the positional data argument.
  std::string normalize = "true";
  std::string random_init = "true";
  option_group_definition new_options("OjaNewton options");
  new_options.add(make_option("OjaNewton", oja_newton).keep().necessary().help("Online Newton with Oja's Sketch"))
      .add(make_option("sketch_size", ON->m).default_value(10).help("size of sketch"))
      .add(make_option("epoch_size", ON->epoch_size).default_value(1).help("size of epoch"))
      .add(make_option("alpha", ON->alpha).default_value(1.f).help("mutiplicative constant for indentiy"))
      .add(make_option("alpha_inverse", alpha_inverse).help("one over alpha, similar to learning rate"))
      .add(make_option("learning_rate_cnt", ON->learning_rate_cnt)
               .default_value(2.f)
               .help("constant for the learning rate 1/t"))
      .add(make_option("normalize", normalize).help("normalize the features or not"))
      .add(make_option("random_init", random_init).help("randomize initialization of Oja or not"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  ON->all = &all;
  ON->_random_state = all.get_random_state();

  ON->normalize = normalize == "true";
  ON->random_init = random_init == "true";

  if (options.was_supplied("alpha_inverse")) ON->alpha = 1.f / alpha_inverse;

  ON->cnt = 0;
  ON->t = 1;
  ON->ev = calloc_or_throw<float>(ON->m + 1);
  ON->b = calloc_or_throw<float>(ON->m + 1);
  ON->D = calloc_or_throw<float>(ON->m + 1);
  ON->A = calloc_or_throw<float*>(ON->m + 1);
  ON->K = calloc_or_throw<float*>(ON->m + 1);
  for (int i = 1; i <= ON->m; i++)
  {
    ON->A[i] = calloc_or_throw<float>(ON->m + 1);
    ON->K[i] = calloc_or_throw<float>(ON->m + 1);
    ON->A[i][i] = 1;
    ON->K[i][i] = 1;
    ON->D[i] = 1;
  }

  ON->buffer = calloc_or_throw<example*>(ON->epoch_size);
  ON->weight_buffer = calloc_or_throw<float>(ON->epoch_size);

  ON->zv = calloc_or_throw<float>(ON->m + 1);
  ON->vv = calloc_or_throw<float>(ON->m + 1);
  ON->tmp = calloc_or_throw<float>(ON->m + 1);

  ON->data.ON = ON.get();
  ON->data.Zx = calloc_or_throw<float>(ON->m + 1);
  ON->data.AZx = calloc_or_throw<float>(ON->m + 1);
  ON->data.delta = calloc_or_throw<float>(ON->m + 1);

  all.weights.stride_shift(static_cast<uint32_t>(ceil(log2(ON->m + 2))));

  learner<OjaNewton, example>& l =
      init_learner(ON, learn, predict, all.weights.stride(), all.get_setupfn_name(OjaNewton_setup));
  l.set_save_load(save_load);
  l.set_finish_example(keep_example);
  return make_base(l);
}
