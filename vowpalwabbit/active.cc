// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cerrno>
#include <cfloat>
#include <cmath>

#include "reductions.h"
#include "rand48.h"
#include "vw.h"
#include "active.h"
#include "vw_exception.h"
#include "shared_data.h"

#include "io/logger.h"

using namespace VW::LEARNER;
using namespace VW::config;

namespace logger = VW::io::logger;

float get_active_coin_bias(float k, float avg_loss, float g, float c0)
{
  float b, sb, rs, sl;
  b = static_cast<float>(c0 * (log(k + 1.) + 0.0001) / (k + 0.0001));
  sb = std::sqrt(b);
  avg_loss = std::min(1.f, std::max(0.f, avg_loss));  // loss should be in [0,1]

  sl = std::sqrt(avg_loss) + std::sqrt(avg_loss + g);
  if (g <= sb * sl + b) return 1;
  rs = (sl + std::sqrt(sl * sl + 4 * g)) / (2 * g);
  return b * rs * rs;
}

float query_decision(active& a, float ec_revert_weight, float k)
{
  float bias, avg_loss, weighted_queries;
  if (k <= 1.)
    bias = 1.;
  else
  {
    weighted_queries = static_cast<float>(a.all->sd->weighted_labeled_examples);
    avg_loss =
        static_cast<float>(a.all->sd->sum_loss / k + std::sqrt((1. + 0.5 * std::log(k)) / (weighted_queries + 0.0001)));
    bias = get_active_coin_bias(k, avg_loss, ec_revert_weight / k, a.active_c0);
  }
  if (a._random_state->get_and_update_random() < bias)
    return 1.f / bias;
  else
    return -1.;
}

template <bool is_learn>
void predict_or_learn_simulation(active& a, single_learner& base, example& ec)
{
  base.predict(ec);

  if (is_learn)
  {
    vw& all = *a.all;

    float k = static_cast<float>(all.sd->t);
    float threshold = 0.f;

    ec.confidence = fabsf(ec.pred.scalar - threshold) / base.sensitivity(ec);
    float importance = query_decision(a, ec.confidence, k);

    if (importance > 0)
    {
      all.sd->queries += 1;
      ec.weight *= importance;
      base.learn(ec);
    }
    else
    {
      ec.l.simple.label = FLT_MAX;
      ec.weight = 0.f;
    }
  }
}

template <bool is_learn>
void predict_or_learn_active(active& a, single_learner& base, example& ec)
{
  if (is_learn)
    base.learn(ec);
  else
    base.predict(ec);

  if (ec.l.simple.label == FLT_MAX)
  {
    float threshold = (a.all->sd->max_label + a.all->sd->min_label) * 0.5f;
    ec.confidence = fabsf(ec.pred.scalar - threshold) / base.sensitivity(ec);
  }
}

void active_print_result(VW::io::writer* f, float res, float weight, v_array<char> tag)
{
  if (f == nullptr) { return; }

  std::stringstream ss;
  ss << std::fixed << res;
  if (!print_tag_by_ref(ss, tag)) { ss << ' '; }

  if (weight >= 0) { ss << " " << std::fixed << weight; }
  ss << '\n';
  const auto ss_str = ss.str();
  ssize_t len = ss_str.size();
  ssize_t t = f->write(ss_str.c_str(), static_cast<unsigned int>(len));
  if (t != len) {
    logger::errlog_error("write error: {}", VW::strerror_to_string(errno));
  }
}

void output_and_account_example(vw& all, active& a, example& ec)
{
  label_data& ld = ec.l.simple;

  all.sd->update(ec.test_only, ld.label != FLT_MAX, ec.loss, ec.weight, ec.get_num_features());
  if (ld.label != FLT_MAX && !ec.test_only) all.sd->weighted_labels += (static_cast<double>(ld.label)) * ec.weight;
  all.sd->weighted_unlabeled_examples += ld.label == FLT_MAX ? ec.weight : 0;

  float ai = -1;
  if (ld.label == FLT_MAX)
    ai = query_decision(a, ec.confidence, static_cast<float>(all.sd->weighted_unlabeled_examples));

  all.print_by_ref(all.raw_prediction.get(), ec.partial_prediction, -1, ec.tag);
  for (auto& i : all.final_prediction_sink) { active_print_result(i.get(), ec.pred.scalar, ai, ec.tag); }

  print_update(all, ec);
}

void return_active_example(vw& all, active& a, example& ec)
{
  output_and_account_example(all, a, ec);
  VW::finish_example(all, ec);
}

base_learner* active_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<active>();

  bool active_option = false;
  bool simulation = false;
  option_group_definition new_options("Active Learning");
  new_options.add(make_option("active", active_option).keep().necessary().help("enable active learning"))
      .add(make_option("simulation", simulation).help("active learning simulation mode"))
      .add(make_option("mellowness", data->active_c0)
               .keep()
               .default_value(8.f)
               .help("active learning mellowness parameter c_0. Default 8"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  data->all = &all;
  data->_random_state = all.get_random_state();

  if (options.was_supplied("lda")) THROW("error: you can't combine lda and active learning");

  auto base = as_singleline(setup_base(options, all));

  // Create new learner
  learner<active, example>* l;
  if (options.was_supplied("simulation"))
    l = &init_learner(data, base, predict_or_learn_simulation<true>, predict_or_learn_simulation<false>,
        all.get_setupfn_name(active_setup) + "-simulation", true);
  else
  {
    all.active = true;
    l = &init_learner(data, base, predict_or_learn_active<true>, predict_or_learn_active<false>,
        all.get_setupfn_name(active_setup), base->learn_returns_prediction);
    l->set_finish_example(return_active_example);
  }

  return make_base(*l);
}
