// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cmath>
#include <cfloat>
#include <errno.h>
#include "reductions.h"
#include "rand48.h"
#include <cfloat>
#include "vw.h"
#include "vw_exception.h"
#include "csoaa.h"
#include "debug_log.h"
#include "io/logger.h"
#include "shared_data.h"

//#define B_SEARCH_MAX_ITER 50
#define B_SEARCH_MAX_ITER 20

using namespace VW::LEARNER;
using namespace COST_SENSITIVE;
using namespace VW::config;

// TODO: cs_active should have its own logger instance (since it uses its own debug flag)
namespace logger = VW::io::logger;

using std::endl;

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cs_active

struct lq_data
{
  // The following are used by cost-sensitive active learning
  float max_pred;            // The max cost for this label predicted by the current set of good regressors
  float min_pred;            // The min cost for this label predicted by the current set of good regressors
  bool is_range_large;       // Indicator of whether this label's cost range was large
  bool is_range_overlapped;  // Indicator of whether this label's cost range overlaps with the cost range that has the
                             // minimum max_pred
  bool query_needed;         // Used in reduction mode: tell upper-layer whether a query is needed for this label
  COST_SENSITIVE::wclass* cl;
};

struct cs_active
{
  // active learning algorithm parameters
  float c0;        // mellowness controlling the width of the set of good functions
  float c1;        // multiplier on the threshold for the cost range test
  float cost_max;  // max cost
  float cost_min;  // min cost

  uint32_t num_classes;
  size_t t;

  bool print_debug_stuff;
  size_t min_labels;
  size_t max_labels;

  bool is_baseline;
  bool use_domination;

  vw* all;  // statistics, loss
  VW::LEARNER::base_learner* l;

  v_array<lq_data> query_data;

  size_t num_any_queries;  // examples where at least one label is queried
  size_t overlapped_and_range_small;
  v_array<size_t> examples_by_queries;
  size_t labels_outside_range;
  float distance_to_range;
  float range;
};

float binarySearch(float fhat, float delta, float sens, float tol)
{
  float maxw = std::min(fhat / sens, FLT_MAX);

  if (maxw * fhat * fhat <= delta) return maxw;

  float l = 0, u = maxw, w, v;

  for (int iter = 0; iter < B_SEARCH_MAX_ITER; iter++)
  {
    w = (u + l) / 2.f;
    v = w * (fhat * fhat - (fhat - sens * w) * (fhat - sens * w)) - delta;
    if (v > 0)
      u = w;
    else
      l = w;
    if (std::fabs(v) <= tol || u - l <= tol) break;
  }

  return l;
}

template <bool is_learn, bool is_simulation>
inline void inner_loop(cs_active& cs_a, single_learner& base, example& ec, uint32_t i, float cost, uint32_t& prediction,
    float& score, float& partial_prediction, bool query_this_label, bool& query_needed)
{
  base.predict(ec, i - 1);
  if (is_learn)
  {
    vw& all = *cs_a.all;
    ec.weight = 1.;
    if (is_simulation)
    {
      // In simulation mode
      if (query_this_label)
      {
        ec.l.simple.label = cost;
        all.sd->queries += 1;
      }
      else
        ec.l.simple.label = FLT_MAX;
    }
    else
    {
      // In reduction mode.
      // If the cost of this label was previously queried, then it should be available for learning now.
      // If the cost of this label was not queried, then skip it.
      if (query_needed)
      {
        ec.l.simple.label = cost;
        if ((cost < cs_a.cost_min) || (cost > cs_a.cost_max))
	  logger::errlog_warn("cost {0} outside of cost range[{1}, {2}]!", cost, cs_a.cost_min, cs_a.cost_max);
      }
      else
        ec.l.simple.label = FLT_MAX;
    }

    if (ec.l.simple.label != FLT_MAX) base.learn(ec, i - 1);
  }
  else if (!is_simulation)
    // Prediction in reduction mode could be used by upper layer to ask whether this label needs to be queried.
    // So we return that.
    query_needed = query_this_label;

  partial_prediction = ec.partial_prediction;
  if (ec.partial_prediction < score || (ec.partial_prediction == score && i < prediction))
  {
    score = ec.partial_prediction;
    prediction = i;
  }
  add_passthrough_feature(ec, i, ec.partial_prediction);
}

inline void find_cost_range(cs_active& cs_a, single_learner& base, example& ec, uint32_t i, float delta, float eta,
    float& min_pred, float& max_pred, bool& is_range_large)
{
  float tol = 1e-6f;

  base.predict(ec, i - 1);
  float sens = base.sensitivity(ec, i - 1);

  if (cs_a.t <= 1 || std::isnan(sens) || std::isinf(sens))
  {
    min_pred = cs_a.cost_min;
    max_pred = cs_a.cost_max;
    is_range_large = true;
    if (cs_a.print_debug_stuff)
      logger::errlog_info(" find_cost_rangeA: i={0} pp={1} sens={2} eta={3} [{4}, {5}] = {6}",
                          i, ec.partial_prediction, sens, eta, min_pred, max_pred, (max_pred - min_pred));
  }
  else
  {
    // finding max_pred and min_pred by binary search
    max_pred =
        std::min(ec.pred.scalar + sens * binarySearch(cs_a.cost_max - ec.pred.scalar, delta, sens, tol), cs_a.cost_max);
    min_pred =
        std::max(ec.pred.scalar - sens * binarySearch(ec.pred.scalar - cs_a.cost_min, delta, sens, tol), cs_a.cost_min);
    is_range_large = (max_pred - min_pred > eta);
    if (cs_a.print_debug_stuff)
      logger::errlog_info(" find_cost_rangeB: i={0} pp={1} sens={2} eta={3} [{4}, {5}] = {6}",
                          i, ec.partial_prediction, sens, eta, min_pred, max_pred, (max_pred - min_pred));
  }
}

template <bool is_learn, bool is_simulation>
void predict_or_learn(cs_active& cs_a, single_learner& base, example& ec)
{
  COST_SENSITIVE::label ld = ec.l.cs;

  if (cs_a.all->sd->queries >= cs_a.min_labels * cs_a.num_classes)
  {
    // save regressor
    std::stringstream filename;
    filename << cs_a.all->final_regressor_name << "." << ec.example_counter << "." << cs_a.all->sd->queries << "."
             << cs_a.num_any_queries;
    VW::save_predictor(*(cs_a.all), filename.str());
    *(cs_a.all->trace_message) << endl << "Number of examples with at least one query = " << cs_a.num_any_queries;
    // Double label query budget
    cs_a.min_labels *= 2;

    for (size_t i = 0; i < cs_a.examples_by_queries.size(); i++)
    {
      *(cs_a.all->trace_message) << endl << "examples with " << i << " labels queried = " << cs_a.examples_by_queries[i];
    }

    *(cs_a.all->trace_message) << endl << "labels outside of cost range = " << cs_a.labels_outside_range;
    *(cs_a.all->trace_message) << endl
                               << "average distance to range = "
                               << cs_a.distance_to_range / (static_cast<float>(cs_a.labels_outside_range));
    *(cs_a.all->trace_message) << endl
                               << "average range = " << cs_a.range / (static_cast<float>(cs_a.labels_outside_range));
  }

  if (cs_a.all->sd->queries >= cs_a.max_labels * cs_a.num_classes) return;

  uint32_t prediction = 1;
  float score = FLT_MAX;
  ec.l.simple = {0.f};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();

  float min_max_cost = FLT_MAX;
  float t = static_cast<float>(cs_a.t);  // ec.example_t;  // current round
  float t_prev = t - 1.f;   // ec.weight; // last round

  float eta = cs_a.c1 * (cs_a.cost_max - cs_a.cost_min) / std::sqrt(t);  // threshold on cost range
  float delta = cs_a.c0 * std::log((cs_a.num_classes * std::max(t_prev, 1.f))) *
      static_cast<float>(std::pow(cs_a.cost_max - cs_a.cost_min, 2));  // threshold on empirical loss difference

  if (ld.costs.size() > 0)
  {
    // Create metadata structure
    for (COST_SENSITIVE::wclass& cl : ld.costs)
    {
      lq_data f = {0.0, 0.0, 0, 0, 0, &cl};
      cs_a.query_data.push_back(f);
    }
    uint32_t n_overlapped = 0;
    for (lq_data& lqd : cs_a.query_data)
    {
      find_cost_range(cs_a, base, ec, lqd.cl->class_index, delta, eta, lqd.min_pred, lqd.max_pred, lqd.is_range_large);
      min_max_cost = std::min(min_max_cost, lqd.max_pred);
    }
    for (lq_data& lqd : cs_a.query_data)
    {
      lqd.is_range_overlapped = (lqd.min_pred <= min_max_cost);
      n_overlapped += static_cast<uint32_t>(lqd.is_range_overlapped);
      cs_a.overlapped_and_range_small += static_cast<size_t>(lqd.is_range_overlapped && !lqd.is_range_large);
      if (lqd.cl->x > lqd.max_pred || lqd.cl->x < lqd.min_pred)
      {
        cs_a.labels_outside_range++;
        cs_a.distance_to_range += std::max(lqd.cl->x - lqd.max_pred, lqd.min_pred - lqd.cl->x);
        cs_a.range += lqd.max_pred - lqd.min_pred;
      }
    }

    bool query = (n_overlapped > 1);
    size_t queries = cs_a.all->sd->queries;
    for (lq_data& lqd : cs_a.query_data)
    {
      bool query_label = ((query && cs_a.is_baseline) || (!cs_a.use_domination && lqd.is_range_large) ||
          (query && lqd.is_range_overlapped && lqd.is_range_large));
      inner_loop<is_learn, is_simulation>(cs_a, base, ec, lqd.cl->class_index, lqd.cl->x, prediction, score,
          lqd.cl->partial_prediction, query_label, lqd.query_needed);
      if (lqd.query_needed) { ec.pred.active_multiclass.more_info_required_for_classes.push_back(lqd.cl->class_index); }
      if (cs_a.print_debug_stuff)
        logger::errlog_info("label={0} x={1} prediction={2} score={3} pp={4} ql={5} qn={6} ro={7} rl={8} "
                            "[{9}, {10}] vs delta={11} n_overlapped={12} is_baseline={13}",
                            lqd.cl->class_index/*0*/, lqd.cl->x/*1*/, prediction/*2*/, score/*3*/, lqd.cl->partial_prediction/*4*/,
                            query_label/*5*/, lqd.query_needed/*6*/, lqd.is_range_overlapped/*7*/, lqd.is_range_large/*8*/,
                            lqd.min_pred/*9*/, lqd.max_pred/*10*/, delta/*11*/, n_overlapped/*12*/, cs_a.is_baseline/*13*/);
    }

    // Need to pop metadata
    cs_a.query_data.clear();

    if (cs_a.all->sd->queries - queries > 0) cs_a.num_any_queries++;

    cs_a.examples_by_queries[cs_a.all->sd->queries - queries] += 1;

    ec.partial_prediction = score;
    if (is_learn) { cs_a.t++; }
  }
  else
  {
    float temp = 0.f;
    bool temp2 = false, temp3 = false;
    for (uint32_t i = 1; i <= cs_a.num_classes; i++)
    { inner_loop<false, is_simulation>(cs_a, base, ec, i, FLT_MAX, prediction, score, temp, temp2, temp3); }
  }

  ec.pred.active_multiclass.predicted_class = prediction;
  ec.l.cs = ld;
}

void finish_example(vw& all, cs_active&, example& ec)
{
  COST_SENSITIVE::output_example(all, ec, ec.l.cs, ec.pred.active_multiclass.predicted_class);
  VW::finish_example(all, ec);
}

base_learner* cs_active_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<cs_active>();

  bool simulation = false;
  int domination;
  option_group_definition new_options("Cost-sensitive Active Learning");
  new_options
      .add(make_option("cs_active", data->num_classes)
               .keep()
               .necessary()
               .help("Cost-sensitive active learning with <k> costs"))
      .add(make_option("simulation", simulation).help("cost-sensitive active learning simulation mode"))
      .add(make_option("baseline", data->is_baseline).help("cost-sensitive active learning baseline"))
      .add(make_option("domination", domination)
               .default_value(1)
               .help("cost-sensitive active learning use domination. Default 1"))
      .add(
          make_option("mellowness", data->c0).keep().default_value(0.1f).help("mellowness parameter c_0. Default 0.1."))
      .add(make_option("range_c", data->c1)
               .default_value(0.5f)
               .help("parameter controlling the threshold for per-label cost uncertainty. Default 0.5."))
      .add(make_option("max_labels", data->max_labels)
               .default_value(std::numeric_limits<size_t>::max())
               .help("maximum number of label queries."))
      .add(make_option("min_labels", data->min_labels)
               .default_value(std::numeric_limits<size_t>::max())
               .help("minimum number of label queries."))
      .add(make_option("cost_max", data->cost_max).default_value(1.f).help("cost upper bound. Default 1."))
      .add(make_option("cost_min", data->cost_min).default_value(0.f).help("cost lower bound. Default 0."))
      // TODO replace with trace and quiet
      .add(make_option("csa_debug", data->print_debug_stuff).help("print debug stuff for cs_active"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  data->use_domination = true;
  if (options.was_supplied("domination") && !domination) data->use_domination = false;

  data->all = &all;
  data->t = 1;

  auto loss_function_type = all.loss->getType();
  if (loss_function_type != "squared") THROW("error: you can't use non-squared loss with cs_active");

  if (options.was_supplied("lda")) THROW("error: you can't combine lda and active learning");

  if (options.was_supplied("active")) THROW("error: you can't use --cs_active and --active at the same time");

  if (options.was_supplied("active_cover"))
    THROW("error: you can't use --cs_active and --active_cover at the same time");

  if (options.was_supplied("csoaa")) THROW("error: you can't use --cs_active and --csoaa at the same time");

  if (!options.was_supplied("adax")) *(all.trace_message) << "WARNING: --cs_active should be used with --adax" << endl;

  // Label parser set to cost sensitive label parser
  all.example_parser->lbl_parser = cs_label;
  all.set_minmax(all.sd, data->cost_max);
  all.set_minmax(all.sd, data->cost_min);
  for (uint32_t i = 0; i < data->num_classes + 1; i++) data->examples_by_queries.push_back(0);

  learner<cs_active, example>& l = simulation
      ? init_learner(data, as_singleline(setup_base(options, all)), predict_or_learn<true, true>,
            predict_or_learn<false, true>, data->num_classes, prediction_type_t::active_multiclass,
            all.get_setupfn_name(cs_active_setup) + "-sim", true)
      : init_learner(data, as_singleline(setup_base(options, all)), predict_or_learn<true, false>,
            predict_or_learn<false, false>, data->num_classes, prediction_type_t::active_multiclass,
            all.get_setupfn_name(cs_active_setup), true);

  l.set_finish_example(finish_example);
  base_learner* b = make_base(l);
  all.cost_sensitive = b;
  return b;
}
