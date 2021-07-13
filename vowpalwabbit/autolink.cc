// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "autolink.h"

#include "learner.h"
#include "global_data.h"
#include "parse_args.h"

#include <cstdint>

using namespace VW::config;

namespace VW
{
struct autolink
{
  autolink(uint32_t d, uint32_t stride_shift);
  void predict(VW::LEARNER::single_learner& base, example& ec);
  void learn(VW::LEARNER::single_learner& base, example& ec);

private:
  void prepare_example(VW::LEARNER::single_learner& base, example& ec);
  void reset_example(example& ec);

  // degree of the polynomial
  const uint32_t _poly_degree;
  const uint32_t _stride_shift;
  static constexpr int AUTOCONSTANT = 524267083;
};
}  // namespace VW

VW::autolink::autolink(uint32_t poly_degree, uint32_t stride_shift)
    : _poly_degree(poly_degree), _stride_shift(stride_shift)
{
}

void VW::autolink::predict(VW::LEARNER::single_learner& base, example& ec)
{
  prepare_example(base, ec);
  base.predict(ec);
  reset_example(ec);
}

void VW::autolink::learn(VW::LEARNER::single_learner& base, example& ec)
{
  prepare_example(base, ec);
  base.learn(ec);
  reset_example(ec);
}

void VW::autolink::prepare_example(VW::LEARNER::single_learner& base, example& ec)
{
  base.predict(ec);
  float base_pred = ec.pred.scalar;

  // Add features of label.
  ec.indices.push_back(autolink_namespace);
  features& fs = ec.feature_space[autolink_namespace];
  for (size_t i = 0; i < _poly_degree; i++)
  {
    if (base_pred != 0.)
    {
      fs.push_back(base_pred, AUTOCONSTANT + (i << _stride_shift));
      base_pred *= ec.pred.scalar;
    }
  }
  ec.reset_total_sum_feat_sq();
}

void VW::autolink::reset_example(example& ec)
{
  features& fs = ec.feature_space[autolink_namespace];
  ec.reset_total_sum_feat_sq();
  fs.clear();
  ec.indices.pop_back();
}

template <bool is_learn>
void predict_or_learn(VW::autolink& b, VW::LEARNER::single_learner& base, example& ec)
{
  if (is_learn)
    b.learn(base, ec);
  else
    b.predict(base, ec);
}

VW::LEARNER::base_learner* autolink_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  uint32_t d;
  option_group_definition new_options("Autolink");
  new_options.add(make_option("autolink", d).keep().necessary().help("create link function with polynomial d"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  auto autolink_reduction = scoped_calloc_or_throw<VW::autolink>(d, all.weights.stride_shift());
  auto base = setup_base(options, all);
  return make_base(init_learner(autolink_reduction, as_singleline(base), predict_or_learn<true>,
      predict_or_learn<false>, all.get_setupfn_name(autolink_setup), base->learn_returns_prediction));
}
