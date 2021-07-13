// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <sstream>
#include <cfloat>
#include "reductions.h"
#include "vw.h"

#include "io/logger.h"

using namespace VW::config;
namespace logger = VW::io::logger;

struct multi_oaa
{
  size_t k;
};

template <bool is_learn>
void predict_or_learn(multi_oaa& o, VW::LEARNER::single_learner& base, example& ec)
{
  MULTILABEL::labels multilabels = ec.l.multilabels;
  MULTILABEL::labels preds = ec.pred.multilabels;
  preds.label_v.clear();

  ec.l.simple = {FLT_MAX};
  ec._reduction_features.template get<simple_label_reduction_features>().reset_to_default();
  uint32_t multilabel_index = 0;
  for (uint32_t i = 0; i < o.k; i++)
  {
    if (is_learn)
    {
      ec.l.simple.label = -1.f;
      if (multilabels.label_v.size() > multilabel_index && multilabels.label_v[multilabel_index] == i)
      {
        ec.l.simple.label = 1.f;
        multilabel_index++;
      }
      base.learn(ec, i);
    }
    else
      base.predict(ec, i);
    if (ec.pred.scalar > 0.) preds.label_v.push_back(i);
  }
  if (is_learn)
  {
    if (multilabel_index < multilabels.label_v.size())
    {
      logger::log_error("label {0} is not in {{0,{1}}} This won't work right.",
                        multilabels.label_v[multilabel_index], o.k - 1);
    }
  }
  ec.pred.multilabels = preds;
  ec.l.multilabels = multilabels;
}

void finish_example(vw& all, multi_oaa&, example& ec)
{
  MULTILABEL::output_example(all, ec);
  VW::finish_example(all, ec);
}

VW::LEARNER::base_learner* multilabel_oaa_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<multi_oaa>();
  option_group_definition new_options("Multilabel One Against All");
  new_options.add(
      make_option("multilabel_oaa", data->k).keep().necessary().help("One-against-all multilabel with <k> labels"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  VW::LEARNER::learner<multi_oaa, example>& l = VW::LEARNER::init_learner(data, as_singleline(setup_base(options, all)),
      predict_or_learn<true>, predict_or_learn<false>, data->k, prediction_type_t::multilabels,
      all.get_setupfn_name(multilabel_oaa_setup), true);
  l.set_finish_example(finish_example);
  all.example_parser->lbl_parser = MULTILABEL::multilabel;

  return make_base(l);
}
