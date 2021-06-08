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
#define prev_S    6  // sum of radios \sum_s |x_s|/h_s of the previous epoch

struct freegrad_update_data
{
  struct freegrad* FG;
  float update;
  float predict;
  float squared_norm_prediction; 
  float squared_norm_clipped_grad;
  float sum_normalized_grad_norms;
  float maximum_gradient_norm;
  int num_features;
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


void inner_freegrad_predict(freegrad_update_data& d, float x, float& wref)
{
  float* w = &wref;
  float h1 = w[H1]; // will be set to the value of the first non-zero gradient w.r.t. the scalar feature x
  float ht = w[HT]; // maximum absolute value of the gradient w.r.t. scalar feature x 
  float w_pred  = 0.0;   // weight for the feature x
  float G  = w[Gsum];  // sum of gradients w.r.t. scalar feature x
  float absG = std::fabs(G); 
  float V  = w[Vsum]; // sum of squared gradients w.r.t. scalar feature x
  float prev_s  = w[prev_S]+1;
    
  // Only predict a non-zero w_pred if a non-zero gradient has been observed
  if (h1 > 0) {
      // freegrad update Equation 9 in paper http://proceedings.mlr.press/v125/mhammedi20a/mhammedi20a.pdf
      w_pred = - G * (2 * V + ht * absG) * pow(h1,2)/(2*pow(V + ht * absG,2) * sqrtf(V)) * exp(pow(absG,2)/(2 * V + 2 * ht * absG));
      // Scaling for linear models (Line 7 of Alg. 2 in the paper)
      w_pred /= (h1 * prev_s);     
  }
  d.squared_norm_prediction += pow(w_pred,2);
  // This is the unprojected predict
  d.predict += w_pred * x;
  d.num_features += 1;
}


void freegrad_predict(freegrad& FG, single_learner&, example& ec)
{
  FG.data.predict = 0.;
  FG.data.num_features = 0;
  FG.data.squared_norm_prediction = 0.;
  size_t num_features_from_interactions = 0.;
  FG.total_weight += ec.weight;
  float norm_w_pred = sqrtf(FG.data.squared_norm_prediction);
  float projection_radius;
    
  // Compute the unprojected predict
  GD::foreach_feature<freegrad_update_data, inner_freegrad_predict>(*FG.all, ec, FG.data, num_features_from_interactions);
    
  if (FG.project){
      // Set the project radius either to the user-specified value, or adaptively  
      if (FG.adaptiveradius)
          projection_radius=sqrtf(FG.data.sum_normalized_grad_norms);
      else
          projection_radius=FG.radius;
      // Compute the projected predict if applicable
      if (norm_w_pred > projection_radius)
           FG.data.predict *= projection_radius / norm_w_pred;
  }
    
  ec.partial_prediction = FG.data.predict * FG.epsilon / FG.data.num_features;
    
  ec.num_features_from_interactions = num_features_from_interactions;
  ec.pred.scalar = GD::finalize_prediction(FG.all->sd, FG.all->logger, ec.partial_prediction);
}


// The following function is computing the tilted gradient of lemma 2 in pap, in case of projects
void freegrad_clipped_gradient_norm(freegrad_update_data& d, float x, float& wref) {
  float* w = &wref;
  float gradient = d.update * x;
  float clipped_gradient = gradient;
  float fabs_g = std::fabs(gradient);
    
  // Perform gradient clipping if necessary
  if (fabs_g > w[HT]) 
      clipped_gradient = gradient * w[HT]/fabs_g;

  d.squared_norm_clipped_grad += pow(clipped_gradient,2);
}


void inner_freegrad_update_after_prediction(freegrad_update_data& d, float x, float& wref) {
  float* w = &wref;
  float gradient = d.update * x;
  float fabs_g = std::fabs(gradient);
  float clipped_gradient = gradient;
  float norm_clipped_grad = sqrtf(d.squared_norm_clipped_grad);
  float norm_w_pred =sqrtf(d.squared_norm_prediction); 
  float projection_radius;
  float tilde_gradient;
 
  float h1 = w[H1]; // will be set to the value of the first non-zero gradient w.r.t. the scalar feature x
  float ht = w[HT]; // maximum absolute value of the gradient w.r.t. scalar feature x 
  float w_pred  = 0.0;   // weight for the feature x
  float G  = w[Gsum];  // sum of gradients w.r.t. scalar feature x
  float absG = std::fabs(G); 
  float V  = w[Vsum]; // sum of squared gradients w.r.t. scalar feature x
  float prev_s  = w[prev_S]+1;
  
  // Computing the freegrad prediction again (Eq.(9) and Line 7 of Alg. 2 in paper)
  if (h1>0) 
      w_pred = - G * (2 * V + ht * absG) * pow(h1,2)/(2*pow(V + ht * absG,2) * sqrtf(V)) * exp(pow(absG,2)/(2 * V + 2 * ht * absG))/(h1 * prev_s);
    
  // Updating the hint sequence
  if (fabs_g > ht) {
      // Perform gradient clipping if necessary
      clipped_gradient = gradient * ht / fabs_g;
      if (h1 == 0)
          w[H1] = fabs_g; 
      w[HT] = fabs_g;
  }      
   
  if (w[H1]>0){ // Only do something if a non-zero gradient has been observed    
      // Set the project radius either to the user-specified value, or adaptively  
      if (d.FG->adaptiveradius)
          projection_radius=sqrtf(d.sum_normalized_grad_norms); 
      else
          projection_radius=d.FG->radius;
      
      // Compute the tilted gradient
      if (norm_w_pred > projection_radius)
          tilde_gradient = (clipped_gradient + norm_clipped_grad * w_pred/norm_w_pred)/2;
      else
          tilde_gradient = clipped_gradient/2;
      
      // Check if restarts are enabled and whether the condition is satisfied
      // (TODO) check if this is the correct thresholding
      if (d.FG->restart && w[HT]/w[H1]>2+w[S]) {
          // Do a restart, but keep the lastest hint info
          w[H1] = w[HT];
          w[Gsum] = tilde_gradient;
          w[Vsum] = pow(tilde_gradient,2);
          w[prev_S]= w[S] + 1; // The restart condition necessarily implies that fabs_g/w[HT]=1
      }
      else {
          // Updating the gradient information
          w[Gsum] += tilde_gradient;
          w[Vsum] += pow(tilde_gradient,2);
      }
      w[S] += fabs_g/w[HT];
  }  
}


void freegrad_update_after_prediction(freegrad& FG, example& ec)
{
    float grad_norm;
    FG.data.squared_norm_clipped_grad = 0;
    
    // Partial derivative of loss
    FG.data.update = FG.all->loss->first_derivative(FG.all->sd, ec.pred.scalar, ec.l.simple.label) * ec.weight;
    
    // Precomputations before update
    GD::foreach_feature<freegrad_update_data, freegrad_clipped_gradient_norm>(*FG.all, ec, FG.data);
    // Update the maximum gradient norm value
    grad_norm = sqrtf(FG.data.squared_norm_clipped_grad);
    if (grad_norm > FG.data.maximum_gradient_norm)
        FG.data.maximum_gradient_norm = grad_norm;
  
    FG.data.sum_normalized_grad_norms += grad_norm/FG.data.maximum_gradient_norm;
    
    // Performing the update
    GD::foreach_feature<freegrad_update_data, inner_freegrad_update_after_prediction>(*FG.all, ec, FG.data);
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

    // (TODO) there used to be a FG.total_weight here
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

// (TODO) check the options below
base_learner* freegrad_setup(options_i& options, vw& all)
{
  auto FG = scoped_calloc_or_throw<freegrad>();
  bool FreeGrad;
  bool restart = false;
  bool project = false;
  bool adaptiveradius = true;
  float radius; 

  option_group_definition new_options("FreeGrad options");
  new_options.add(make_option("FreeGrad", FreeGrad).keep().necessary().help("Diagonal FreeGrad Algorithm")).add(make_option("restart", restart).help("Use the FreeRange restarts"))
      .add(make_option("project", project).help("Project the outputs to adapt to both the lipschitz and comparator norm")).add(make_option("radius", radius).help("Radius of the l2-ball for the projection. If not supplied, an adaptive radius will be used.")).add(make_option("epsilon", FG->epsilon).default_value(1.f).help("Initial wealth"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (options.was_supplied("radius")){
      FG->radius = radius;
      adaptiveradius = false;
  }
    
  // Defaults
  FG->data.sum_normalized_grad_norms = 1.;
  FG->data.maximum_gradient_norm = 0.;
  FG->data.FG = FG.get();

  FG->all = &all;
  FG->restart = restart;
  FG->project = project;
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
  FG->freegrad_size = 7;
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
  // if (all.audit || all.hash_inv)
  //  l = &init_learner(FG, learn_ptr, predict<true>, UINT64_ONE << all.weights.stride_shift(),
  //      all.get_setupfn_name(freegrad_setup) + "-" + algorithm_name + "-audit");
  // else
  //  l = &init_learner(FG, learn_ptr, predict<false>, UINT64_ONE << all.weights.stride_shift(),
  //      all.get_setupfn_name(freegrad_setup) + "-" + algorithm_name, learn_returns_prediction);
    
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
