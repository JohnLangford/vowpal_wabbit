// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include "gd.h"
#include <cfloat>
#include "reductions.h"

using namespace VW::config;

using std::cout;

// TODO: This file should probably(?) use trace_message
struct print
{
  print(vw* all) : all(all) {}
  vw* all;
};  // regressor, feature loop

void print_feature(vw& /* all */, float value, uint64_t index)
{
  cout << index;
  if (value != 1.) cout << ":" << value;
  cout << " ";
}

void learn(print& p, VW::LEARNER::base_learner&, example& ec)
{
  if (ec.l.simple.label != FLT_MAX)
  {
    cout << ec.l.simple.label << " ";
    const auto& simple_red_features = ec._reduction_features.template get<simple_label_reduction_features>();
    if (ec.weight != 1 || simple_red_features.initial != 0)
    {
      cout << ec.weight << " ";
      if (simple_red_features.initial != 0) cout << simple_red_features.initial << " ";
    }
  }
  if (!ec.tag.empty())
  {
    cout << '\'';
    cout.write(ec.tag.begin(), ec.tag.size());
  }
  cout << "| ";
  GD::foreach_feature<vw, uint64_t, print_feature>(*(p.all), ec, *p.all);
  cout << std::endl;
}

VW::LEARNER::base_learner* print_setup(VW::setup_base_fn&, options_i& options, vw& all)
{
  bool print_option = false;
  option_group_definition new_options("Print psuedolearner");
  new_options.add(make_option("print", print_option).keep().necessary().help("print examples"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  all.weights.stride_shift(0);
  auto* learner = VW::LEARNER::make_base_learner(VW::make_unique<print>(&all), learn, learn,
      all.get_setupfn_name(print_setup), prediction_type_t::scalar, label_type_t::simple)
                      .build();
  return VW::LEARNER::make_base(*learner);
}
