// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <cmath>
#include <sstream>
#include <cfloat>
#include <cmath>
#include "correctedMath.h"
#include "reductions.h"
#include "vw_exception.h"
#include "vw.h"
#include "shared_data.h"

#include "io/logger.h"

using namespace VW::config;
namespace logger = VW::io::logger;

struct oaa
{
  uint64_t k;
  vw* all;                    // for raw
  polyprediction* pred;       // for multipredict
  uint64_t num_subsample;     // for randomized subsampling, how many negatives to draw?
  uint32_t* subsample_order;  // for randomized subsampling, in what order should we touch classes
  size_t subsample_id;        // for randomized subsampling, where do we live in the list

  ~oaa()
  {
    free(pred);
    free(subsample_order);
  }
};

void learn_randomized(oaa& o, VW::LEARNER::single_learner& base, example& ec)
{
  MULTICLASS::label_t ld = ec.l.multi;
  if (ld.label == 0 || (ld.label > o.k && ld.label != static_cast<uint32_t>(-1)))
    logger::log_error("label {0} is not in {{1,{1}}} This won't work right.", ld.label, o.k);

  ec.l.simple = {1.};  // truth
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
  base.learn(ec, ld.label - 1);

  size_t prediction = ld.label;
  float best_partial_prediction = ec.partial_prediction;

  ec.l.simple.label = -1.;
  float weight_temp = ec.weight;
  ec.weight *= (static_cast<float>(o.k)) / static_cast<float>(o.num_subsample);
  size_t p = o.subsample_id;
  size_t count = 0;
  while (count < o.num_subsample)
  {
    uint32_t l = o.subsample_order[p];
    p = (p + 1) % o.k;
    if (l == ld.label - 1) continue;
    base.learn(ec, l);
    if (ec.partial_prediction > best_partial_prediction)
    {
      best_partial_prediction = ec.partial_prediction;
      prediction = l + 1;
    }
    count++;
  }
  o.subsample_id = p;

  ec.pred.multiclass = static_cast<uint32_t>(prediction);
  ec.l.multi = ld;
  ec.weight = weight_temp;
}

template <bool print_all, bool scores, bool probabilities>
void learn(oaa& o, VW::LEARNER::single_learner& base, example& ec)
{
  // Save label
  MULTICLASS::label_t mc_label_data = ec.l.multi;

  // Label validation
  if (mc_label_data.label == 0 || (mc_label_data.label > o.k && mc_label_data.label != static_cast<uint32_t>(-1)))
    logger::log_error("label {0} is not in {{1,{1}}} This won't work right.", mc_label_data.label, o.k);

  ec.l.simple = {FLT_MAX};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();

  for (uint32_t i = 1; i <= o.k; i++)
  {
    ec.l.simple = {(mc_label_data.label == i) ? 1.f : -1.f};
    // The following is an unfortunate loss of abstraction
    // Downstream reduction (gd.update) uses the prediction
    // from here
    ec.pred.scalar = o.pred[i - 1].scalar;
    base.update(ec, i - 1);
  }

  // Restore label
  ec.l.multi = mc_label_data;
}

template <bool print_all, bool scores, bool probabilities>
void predict(oaa& o, LEARNER::single_learner& base, example& ec)
{
  // The predictions are either an array of scores or a single
  // class id of a multiclass label

  // In the case we return scores, we need to save a copy of
  // the pre-allocated scores array since ec.pred will be
  // used for other predictions.
  v_array<float> scores_array;
  if (scores) scores_array = ec.pred.scalars;

  // oaa.pred - Predictions will get stored in this array
  // oaa.k    - Number of learners to call predict() on
  base.multipredict(ec, 0, o.k, o.pred, true);

  // Find the class with the largest score (index +1)
  uint32_t prediction = 1;
  for (uint32_t i = 2; i <= o.k; i++)
    if (o.pred[i - 1].scalar > o.pred[prediction - 1].scalar) prediction = i;

  if (ec.passthrough)
    for (uint32_t i = 1; i <= o.k; i++) add_passthrough_feature(ec, i, o.pred[i - 1].scalar);

  // Print predictions to a file
  if (print_all)
  {
    std::stringstream output_string_stream;
    output_string_stream << "1:" << o.pred[0].scalar;
    for (uint32_t i = 2; i <= o.k; i++) output_string_stream << ' ' << i << ':' << o.pred[i - 1].scalar;
    o.all->print_text_by_ref(o.all->raw_prediction.get(), output_string_stream.str(), ec.tag);
  }

  // The predictions are an array of scores (as opposed to a single index of a
  // class)
  if (scores)
  {
    scores_array.clear();
    for (uint32_t i = 0; i < o.k; i++) scores_array.push_back(o.pred[i].scalar);
    ec.pred.scalars = scores_array;

    // The scores should be converted to probabilities
    if (probabilities)
    {
      float sum_prob = 0;
      for (uint32_t i = 0; i < o.k; i++)
      {
        ec.pred.scalars[i] = 1.f / (1.f + correctedExp(-o.pred[i].scalar));
        sum_prob += ec.pred.scalars[i];
      }
      const float inv_sum_prob = 1.f / sum_prob;
      for (uint32_t i = 0; i < o.k; i++) ec.pred.scalars[i] *= inv_sum_prob;
    }
  }
  else
    ec.pred.multiclass = prediction;
}

// TODO: partial code duplication with multiclass.cc:finish_example
template <bool probabilities>
void finish_example_scores(vw& all, oaa& o, example& ec)
{
  // === Compute multiclass_log_loss
  // TODO:
  // What to do if the correct label is unknown, i.e. (uint32_t)-1?
  //   Suggestion: increase all.sd->weighted_unlabeled_examples???,
  //               but not sd.example_number, so the average loss is not influenced.
  // What to do if the correct_class_prob==0?
  //   Suggestion: have some maximal multiclass_log_loss limit, e.g. 999.
  float multiclass_log_loss = 999;  // -log(0) = plus infinity
  float correct_class_prob = 0;
  if (probabilities)
  {
    if (ec.l.multi.label <= o.k)  // prevent segmentation fault if labeĺ==(uint32_t)-1
      correct_class_prob = ec.pred.scalars[ec.l.multi.label - 1];
    if (correct_class_prob > 0) multiclass_log_loss = -std::log(correct_class_prob) * ec.weight;
    if (ec.test_only)
      all.sd->holdout_multiclass_log_loss += multiclass_log_loss;
    else
      all.sd->multiclass_log_loss += multiclass_log_loss;
  }
  // === Compute `prediction` and zero_one_loss
  // We have already computed `prediction` in predict_or_learn,
  // but we cannot store it in ec.pred union because we store ec.pred.probs there.
  uint32_t prediction = 0;
  for (uint32_t i = 1; i < o.k; i++)
    if (ec.pred.scalars[i] > ec.pred.scalars[prediction]) prediction = i;
  prediction++;  // prediction is 1-based index (not 0-based)
  float zero_one_loss = 0;
  if (ec.l.multi.label != prediction) zero_one_loss = ec.weight;

  // === Print probabilities for all classes
  std::ostringstream outputStringStream;
  for (uint32_t i = 0; i < o.k; i++)
  {
    if (i > 0) outputStringStream << ' ';
    if (all.sd->ldict) { outputStringStream << all.sd->ldict->get(i + 1); }
    else
      outputStringStream << i + 1;
    outputStringStream << ':' << ec.pred.scalars[i];
  }
  const auto ss_str = outputStringStream.str();
  for (auto& sink : all.final_prediction_sink) all.print_text_by_ref(sink.get(), ss_str, ec.tag);

  // === Report updates using zero-one loss
  all.sd->update(
      ec.test_only, ec.l.multi.label != static_cast<uint32_t>(-1), zero_one_loss, ec.weight, ec.get_num_features());
  // Alternatively, we could report multiclass_log_loss.
  // all.sd->update(ec.test_only, multiclass_log_loss, ec.weight, ec.num_features);
  // Even better would be to report both losses, but this would mean to increase
  // the number of columns and this would not fit narrow screens.
  // So let's report (average) multiclass_log_loss only in the final resume.

  // === Print progress report
  if (probabilities)
    MULTICLASS::print_update_with_probability(all, ec, prediction);
  else
    MULTICLASS::print_update_with_score(all, ec, prediction);
  VW::finish_example(all, ec);
}

VW::LEARNER::base_learner* oaa_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<oaa>();
  bool probabilities = false;
  bool scores = false;
  option_group_definition new_options("One Against All Options");
  new_options.add(make_option("oaa", data->k).keep().necessary().help("One-against-all multiclass with <k> labels"))
      .add(make_option("oaa_subsample", data->num_subsample)
               .help("subsample this number of negative examples when learning"))
      .add(make_option("probabilities", probabilities).help("predict probabilites of all classes"))
      .add(make_option("scores", scores).help("output raw scores per class"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (all.sd->ldict && (data->k != all.sd->ldict->getK()))
    THROW("error: you have " << all.sd->ldict->getK() << " named labels; use that as the argument to oaa")

  data->all = &all;
  data->pred = calloc_or_throw<polyprediction>(data->k);
  data->subsample_order = nullptr;
  data->subsample_id = 0;
  if (data->num_subsample > 0)
  {
    if (data->num_subsample >= data->k)
    {
      data->num_subsample = 0;
      *(all.trace_message) << "oaa is turning off subsampling because your parameter >= K" << std::endl;
    }
    else
    {
      data->subsample_order = calloc_or_throw<uint32_t>(data->k);
      for (size_t i = 0; i < data->k; i++) data->subsample_order[i] = static_cast<uint32_t>(i);
      for (size_t i = 0; i < data->k; i++)
      {
        size_t j =
            static_cast<size_t>(all.get_random_state()->get_and_update_random() * static_cast<float>(data->k - i)) + i;
        uint32_t tmp = data->subsample_order[i];
        data->subsample_order[i] = data->subsample_order[j];
        data->subsample_order[j] = tmp;
      }
    }
  }

  oaa* data_ptr = data.get();
  VW::LEARNER::learner<oaa, example>* l;
  auto base = as_singleline(setup_base(options, all));
  if (probabilities || scores)
  {
    if (probabilities)
    {
      auto loss_function_type = all.loss->getType();
      if (loss_function_type != "logistic")
        *(all.trace_message) << "WARNING: --probabilities should be used only with --loss_function=logistic"
                             << std::endl;
      // the three boolean template parameters are: is_learn, print_all and scores
      l = &LEARNER::init_multiclass_learner(data, base, learn<false, true, true>, predict<false, true, true>,
          all.example_parser, data->k, all.get_setupfn_name(oaa_setup) + "-prob", prediction_type_t::scalars);
      all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
      all.sd->report_multiclass_log_loss = true;
      l->set_finish_example(finish_example_scores<true>);
    }
    else
    {
      l = &VW::LEARNER::init_multiclass_learner(data, base, learn<false, true, false>, predict<false, true, false>,
          all.example_parser, data->k, all.get_setupfn_name(oaa_setup) + "-scores", prediction_type_t::scalars);
      all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
      l->set_finish_example(finish_example_scores<false>);
    }
  }
  else if (all.raw_prediction != nullptr)
  {
    l = &VW::LEARNER::init_multiclass_learner(data, base, learn<true, false, false>, predict<true, false, false>,
        all.example_parser, data->k, all.get_setupfn_name(oaa_setup) + "-raw", prediction_type_t::multiclass);
    all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
  }
  else
  {
    l = &VW::LEARNER::init_multiclass_learner(data, base, learn<false, false, false>, predict<false, false, false>,
        all.example_parser, data->k, all.get_setupfn_name(oaa_setup), prediction_type_t::multiclass);
    all.example_parser->lbl_parser.label_type = label_type_t::multiclass;
  }

  if (data_ptr->num_subsample > 0)
  {
    l->set_learn(learn_randomized);
    l->set_finish_example(MULTICLASS::finish_example_without_loss<oaa>);
    l->learn_returns_prediction = false;
  }

  return make_base(*l);
}
