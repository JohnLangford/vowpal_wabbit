// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cmath>
#include <string>
#include <cfloat>
#include "correctedMath.h"
#include "gd.h"
#include "shared_data.h"
#include "io/logger.h"

using namespace VW::LEARNER;
using namespace VW::config;
using namespace VW::math;

namespace logger = VW::io::logger;

#define W    0  // current parameter
#define Gsum    1  // sum of gradients
#define Vsum    2  // sum of squared gradients
#define H1   3  // maximum absolute value of features
#define HT   4  // maximum gradient
#define S    5  // sum of radios \sum_s |x_s|/h_s  

struct freegrad_update_data
{
  struct freegrad* FG;
  float update;
  float ec_weight;
  float predict;
  float squared_norm_prediction; 
  float grad_dot_w;
  float squared_norm_clipped_grad;
  float sum_normalized_grad_norms;
  float maximum_clipped_gradient_norm;
};

struct freegrad
{
  vw* all;  // features, finalize, l1, l2,
  float epsilon;
  bool restart;
  bool project;
  bool adaptiveradius;
  float radius;
  struct freegrad_update_data data;
  size_t no_win_counter;
  size_t early_stop_thres;
  uint32_t freegrad_size;
  double total_weight;
};


template <bool audit>
void predict(freegrad& b, single_learner&, example& ec)
{
  size_t num_features_from_interactions = 0;
  ec.partial_prediction = GD::inline_predict(*b.all, ec, num_features_from_interactions);
  ec.num_features_from_interactions = num_features_from_interactions;
  ec.pred.scalar = GD::finalize_prediction(b.all->sd, b.all->logger, ec.partial_prediction);
  if (audit) GD::print_audit_features(*(b.all), ec);
}


void inner_freegrad_predict(freegrad_update_data& d, float x, float& wref)
{
  float* w = &wref;
  float h1 = w[H1]; // will be set to the value of the first non-zero gradient w.r.t. the scalar feature x
  float ht = w[HT]; // maximum absolute value of the gradient w.r.t. scalar feature x 
  float w_pred  = 0.0;   // weight for the feature x
  float G  = w[Gsum];  // sum of gradients w.r.t. scalar feature x
  float absG = std::fabs(G); 
  float V  = w[Vsum]; // sum of squared gradients w.r.t. scalar feature x
  float epsilon = d.FG->epsilon;
    
  // Only predict a non-zero w_pred if a non-zero gradient has been observed
  // freegrad update Equation 9 in paper http://proceedings.mlr.press/v125/mhammedi20a/mhammedi20a.pdf
  if (h1 > 0) 
    w_pred  = -G * epsilon * (2. * V + ht * absG) * pow(h1, 2.f)/(2.*pow(V + ht * absG,2.f) * sqrtf(V)) * exp(pow(absG,2.f)/(2. * V + 2. * ht * absG));

  d.squared_norm_prediction += pow(w_pred,2.f);
  // This is the unprojected predict
  d.predict +=  w_pred * x;
}


void freegrad_predict(freegrad& FG, single_learner&, example& ec)
{
  FG.data.predict = 0.;
  FG.data.squared_norm_prediction = 0.;
  size_t num_features_from_interactions = 0.;
  FG.total_weight += ec.weight;
  float norm_w_pred;
  float projection_radius;
    
  // Compute the unprojected predict
  GD::foreach_feature<freegrad_update_data, inner_freegrad_predict>(*FG.all, ec, FG.data, num_features_from_interactions);
  norm_w_pred =  sqrtf(FG.data.squared_norm_prediction);
  
  if (FG.project){
    // Set the project radius either to the user-specified value, or adaptively  
    if (FG.adaptiveradius)
      projection_radius=FG.epsilon * sqrtf(FG.data.sum_normalized_grad_norms);
    else
      projection_radius=FG.radius;
    // Compute the projected predict if applicable
    if (norm_w_pred > projection_radius)
      FG.data.predict *= projection_radius / norm_w_pred;
  }
  ec.partial_prediction = FG.data.predict;
    
  ec.num_features_from_interactions = num_features_from_interactions;
  ec.pred.scalar = GD::finalize_prediction(FG.all->sd, FG.all->logger, ec.partial_prediction);
}


void gradient_dot_w(freegrad_update_data& d, float x, float& wref) {
  float* w = &wref;
  float h1 = w[H1]; // will be set to the value of the first non-zero gradient w.r.t. the scalar feature x
  float ht = w[HT]; // maximum absolute value of the gradient w.r.t. scalar feature x 
  float w_pred  = 0.0;   // weight for the feature x
  float G  = w[Gsum];  // sum of gradients w.r.t. scalar feature x
  float absG = std::fabs(G); 
  float V  = w[Vsum]; // sum of squared gradients w.r.t. scalar feature x
  float epsilon = d.FG->epsilon;
  float gradient = d.update * x;
    
  // Only predict a non-zero w_pred if a non-zero gradient has been observed
  if (h1>0)
    w_pred =  -G * epsilon * (2. * V + ht * absG) * pow(h1,2.f)/(2.*pow(V + ht * absG,2.f) * sqrtf(V))* exp(pow(absG,2.f)/(2 * V + 2. * ht * absG));

  d.grad_dot_w += gradient * w_pred;
}


void inner_freegrad_update_after_prediction(freegrad_update_data& d, float x, float& wref) {
  float* w = &wref;
  float gradient = d.update * x;
  float tilde_gradient = gradient;
  float clipped_gradient;
  float fabs_g = std::fabs(gradient);
  float g_dot_w = d.grad_dot_w;
  float norm_w_pred =sqrtf(d.squared_norm_prediction); 
  float projection_radius;
  float fabs_tilde_g;
 
  float h1 = w[H1]; // will be set to the value of the first non-zero gradient w.r.t. the scalar feature x
  float ht = w[HT]; // maximum absolute value of the gradient w.r.t. scalar feature x 
  float w_pred = 0.0;   // weight for the feature x
  float G  = w[Gsum];  // sum of gradients w.r.t. scalar feature x
  float absG = std::fabs(G); 
  float V  = w[Vsum]; // sum of squared gradients w.r.t. scalar feature x
  float epsilon = d.FG->epsilon;
  
  // Computing the freegrad prediction again (Eq.(9) and Line 7 of Alg. 2 in paper)
  if (h1>0)
    w[W] =  -G * epsilon * (2. * V + ht * absG) * pow(h1, 2.f)/(2.*pow(V + ht * absG,2.f) * sqrtf(V))* exp(pow(absG,2.f)/(2 * V + 2. * ht * absG));

  // Compute the tilted gradient: 
  // Cutkosky's varying constrains' reduction in 
  // Alg. 1 in http://proceedings.mlr.press/v119/cutkosky20a/cutkosky20a.pdf with sphere sets
  if (d.FG->project){ 
    // Set the project radius either to the user-specified value, or adaptively  
    if (d.FG->adaptiveradius)
      projection_radius=d.FG->epsilon * sqrtf(d.sum_normalized_grad_norms); 
    else
      projection_radius=d.FG->radius;
    
    if(norm_w_pred > projection_radius && g_dot_w < 0) 
      tilde_gradient = gradient - (g_dot_w * w[W]) / pow(norm_w_pred,2.f);
  }
    
  if (tilde_gradient==0) // Only do something if a non-zero gradient has been observed 
    return;
    
  clipped_gradient = tilde_gradient;
  fabs_tilde_g = std::fabs(tilde_gradient);
    
  // Updating the hint sequence
  if (h1 == 0){
    w[H1] = fabs_tilde_g;
    w[HT] = fabs_tilde_g;
    w[Vsum] += d.ec_weight * pow(fabs_tilde_g,2.f);
  }
  else if (fabs_tilde_g > ht) {
    // Perform gradient clipping if necessary
    clipped_gradient *= ht / fabs_tilde_g;
    w[HT] = fabs_tilde_g;
  }
  d.squared_norm_clipped_grad += pow(clipped_gradient,2.f);

  // Check if restarts are enabled and whether the condition is satisfied
  if (d.FG->restart && w[HT]/w[H1]>w[S]+2) {
      // Do a restart, but keep the lastest hint info
      w[H1] = w[HT];
      w[Gsum] = clipped_gradient + (d.ec_weight - 1) * tilde_gradient;
      w[Vsum] = pow(clipped_gradient, 2.f) + (d.ec_weight - 1) *  pow(tilde_gradient, 2.f);
  }
  else {
      // Updating the gradient information
      w[Gsum] += clipped_gradient + (d.ec_weight - 1) * tilde_gradient;
      w[Vsum] += pow(clipped_gradient, 2.f) + (d.ec_weight - 1) *  pow(tilde_gradient, 2.f);
  }
  if (ht>0)
      w[S] += std::fabs(clipped_gradient)/ht + (d.ec_weight - 1) * std::fabs(tilde_gradient)/w[HT];
}


void freegrad_update_after_prediction(freegrad& FG, example& ec)
{
    float clipped_grad_norm;
    FG.data.grad_dot_w = 0.;
    FG.data.squared_norm_clipped_grad = 0.;
    FG.data.ec_weight = (float) ec.weight;
    
    // Partial derivative of loss (Note that the weight of the examples ec is not accounted for at this stage. This is done in inner_freegrad_update_after_prediction)
    FG.data.update = FG.all->loss->first_derivative(FG.all->sd, ec.pred.scalar, ec.l.simple.label);
    
    // Compute gradient norm
    GD::foreach_feature<freegrad_update_data, gradient_dot_w>(*FG.all, ec, FG.data);
    
    // Performing the update
    GD::foreach_feature<freegrad_update_data, inner_freegrad_update_after_prediction>(*FG.all, ec, FG.data);
    
    // Update the maximum gradient norm value
    clipped_grad_norm = sqrtf(FG.data.squared_norm_clipped_grad);
    if (clipped_grad_norm > FG.data.maximum_clipped_gradient_norm)
        FG.data.maximum_clipped_gradient_norm = clipped_grad_norm;
  
    if (FG.data.maximum_clipped_gradient_norm >0)
      FG.data.sum_normalized_grad_norms += FG.data.ec_weight * clipped_grad_norm/FG.data.maximum_clipped_gradient_norm;
}


template <bool audit>
void learn_freegrad(freegrad& a, single_learner& base, example& ec) {
  // update state based on the example and predict
  freegrad_predict(a, base, ec);
  if (audit) GD::print_audit_features(*(a.all), ec);
  
  // update state based on the prediction
  freegrad_update_after_prediction(a, ec);
}


void save_load(freegrad& FG, io_buf& model_file, bool read, bool text)
{
  vw* all = FG.all;
  if (read) initialize_regressor(*all);

  if (model_file.num_files() != 0)
  {
    bool resume = all->save_resume;
    std::stringstream msg;
    msg << ":" << resume << "\n";
    bin_text_read_write_fixed(model_file, reinterpret_cast<char*>(&resume), sizeof(resume), "", read, msg, text);

    if (resume)
      GD::save_load_online_state(*all, model_file, read, text, FG.total_weight, nullptr, FG.freegrad_size);
    else
      GD::save_load_regressor(*all, model_file, read, text);
  }
}

void end_pass(freegrad& g)
{
  vw& all = *g.all;

  if (!all.holdout_set_off)
  {
    if (summarize_holdout_set(all, g.no_win_counter)) finalize_regressor(all, all.final_regressor_name);
    if ((g.early_stop_thres == g.no_win_counter) &&
        ((all.check_holdout_every_n_passes <= 1) || ((all.current_pass % all.check_holdout_every_n_passes) == 0)))
      set_done(all);
  }
}

base_learner* freegrad_setup(options_i& options, vw& all)
{
  auto FG = scoped_calloc_or_throw<freegrad>();
  bool FreeGrad;
  bool restart = false;
  bool project = false;
  bool adaptiveradius = true;
  float radius; 

  option_group_definition new_options("FreeGrad options");
  new_options.add(make_option("freegrad", FreeGrad).keep().help("Diagonal FreeGrad Algorithm")).add(make_option("restart", restart).help("Use the FreeRange restarts"))
      .add(make_option("project", project).help("Project the outputs to adapt to both the lipschitz and comparator norm")).add(make_option("radius", radius).help("Radius of the l2-ball for the projection. If not supplied, an adaptive radius will be used.")).add(make_option("fepsilon", FG->epsilon).default_value(1.f).help("Initial wealth"));

  options.add_and_parse(new_options);

  if (!FreeGrad) { return nullptr; }

  if (options.was_supplied("radius")){
      FG->radius = radius;
      adaptiveradius = false;
  }
    
  // Defaults
  FG->data.sum_normalized_grad_norms = 1;
  FG->data.maximum_clipped_gradient_norm = 0.;
  FG->data.FG = FG.get();

  FG->all = &all;
  FG->restart = restart;
  FG->project = project;
  FG->adaptiveradius = adaptiveradius;
  FG->no_win_counter = 0;
  FG->all->normalized_sum_norm_x = 0;
  FG->total_weight = 0;
    
  void (*learn_ptr)(freegrad&, single_learner&, example&) = nullptr;

  std::string algorithm_name;

  algorithm_name = "FreeGrad";
  if (all.audit || all.hash_inv)
    learn_ptr = learn_freegrad<true>;
  else
    learn_ptr = learn_freegrad<false>;
    
  all.weights.stride_shift(3);  // NOTE: for more parameter storage
  FG->freegrad_size = 6;
  bool learn_returns_prediction = true;

  if (!all.logger.quiet)
  {
    *(all.trace_message) << "Enabling FreeGrad based optimization" << std::endl;
    *(all.trace_message) << "Algorithm used: " << algorithm_name << std::endl;
  }

  if (!all.holdout_set_off)
  {
    all.sd->holdout_best_loss = FLT_MAX;
    FG->early_stop_thres = options.get_typed_option<size_t>("early_terminate").value();
  }
  
  learner<freegrad, example>* l;
  if (all.audit || all.hash_inv)
    l = &init_learner(FG, learn_ptr, predict<true>, UINT64_ONE << all.weights.stride_shift(),
        all.get_setupfn_name(freegrad_setup) + "-" + algorithm_name + "-audit");
  else
    l = &init_learner(FG, learn_ptr, predict<false>, UINT64_ONE << all.weights.stride_shift(),
        all.get_setupfn_name(freegrad_setup) + "-" + algorithm_name, learn_returns_prediction);
    
  // (TODO) Check what the multipredict is about and set_and_pass
  // l->set_sensitivity(sensitivity);
  // if (all.audit || all.hash_inv)
  //   l->set_multipredict(multipredict<true>);
  // else
  //   l->set_multipredict(multipredict<false>);

  l->set_save_load(save_load);
  l->set_end_pass(end_pass);
  return make_base(*l);
}
