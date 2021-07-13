// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "cb_explore_adf_first.h"

#include "reductions.h"
#include "cb_adf.h"
#include "rand48.h"
#include "bs.h"
#include "gen_cs_example.h"
#include "cb_explore.h"
#include "explore.h"
#include "cb_explore_adf_common.h"
#include "vw_versions.h"
#include "version.h"
#include "label_parser.h"

#include <vector>
#include <algorithm>
#include <cmath>

using namespace VW::LEARNER;

namespace VW
{
namespace cb_explore_adf
{
namespace first
{
struct cb_explore_adf_first
{
private:
  size_t _tau;
  float _epsilon;

  VW::version_struct _model_file_version;

public:
  cb_explore_adf_first(size_t tau, float epsilon, VW::version_struct model_file_version);
  ~cb_explore_adf_first() = default;

  // Should be called through cb_explore_adf_base for pre/post-processing
  void predict(multi_learner& base, multi_ex& examples) { predict_or_learn_impl<false>(base, examples); }
  void learn(multi_learner& base, multi_ex& examples) { predict_or_learn_impl<true>(base, examples); }
  void save_load(io_buf& io, bool read, bool text);

private:
  template <bool is_learn>
  void predict_or_learn_impl(multi_learner& base, multi_ex& examples);
};

cb_explore_adf_first::cb_explore_adf_first(size_t tau, float epsilon, VW::version_struct model_file_version)
    : _tau(tau), _epsilon(epsilon), _model_file_version(std::move(model_file_version))
{
}

template <bool is_learn>
void cb_explore_adf_first::predict_or_learn_impl(multi_learner& base, multi_ex& examples)
{
  // Explore tau times, then act according to optimal.
  if (is_learn)
    multiline_learn_or_predict<true>(base, examples, examples[0]->ft_offset);
  else
    multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);

  v_array<ACTION_SCORE::action_score>& preds = examples[0]->pred.a_s;
  uint32_t num_actions = static_cast<uint32_t>(preds.size());

  if (_tau)
  {
    float prob = 1.f / static_cast<float>(num_actions);
    for (size_t i = 0; i < num_actions; i++) preds[i].score = prob;
    if (is_learn) _tau--;
  }
  else
  {
    for (size_t i = 1; i < num_actions; i++) preds[i].score = 0.;
    preds[0].score = 1.0;
  }

  exploration::enforce_minimum_probability(_epsilon, true, begin_scores(preds), end_scores(preds));
}

void cb_explore_adf_first::save_load(io_buf& io, bool read, bool text)
{
  if (io.num_files() == 0) { return; }
  if (!read || _model_file_version >= VERSION_FILE_WITH_FIRST_SAVE_RESUME)
  {
    std::stringstream msg;
    if (!read) { msg << "cb first adf storing example counter:  = " << _tau << "\n"; }
    bin_text_read_write_fixed_validated(io, reinterpret_cast<char*>(&_tau), sizeof(_tau), "", read, msg, text);
  }
}

base_learner* setup(VW::setup_base_fn& setup_base, config::options_i& options, vw& all)
{
  using config::make_option;
  bool cb_explore_adf_option = false;
  size_t tau = 0;
  float epsilon = 0.;
  config::option_group_definition new_options("Contextual Bandit Exploration with ADF (tau-first)");
  new_options
      .add(make_option("cb_explore_adf", cb_explore_adf_option)
               .keep()
               .necessary()
               .help("Online explore-exploit for a contextual bandit problem with multiline action dependent features"))
      .add(make_option("first", tau).keep().necessary().help("tau-first exploration"))
      .add(make_option("epsilon", epsilon).keep().allow_override().help("epsilon-greedy exploration"));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  // Ensure serialization of cb_adf in all cases.
  if (!options.was_supplied("cb_adf")) { options.insert("cb_adf", ""); }

  size_t problem_multiplier = 1;

  multi_learner* base = as_multiline(setup_base(options, all));
  all.example_parser->lbl_parser = CB::cb_label;

  bool with_metrics = options.was_supplied("extra_metrics");

  using explore_type = cb_explore_adf_base<cb_explore_adf_first>;
  auto data = VW::make_unique<explore_type>(with_metrics, tau, epsilon, all.model_file_ver);

  if (epsilon < 0.0 || epsilon > 1.0) { THROW("The value of epsilon must be in [0,1]"); }
  auto* l = make_reduction_learner(
      std::move(data), base, explore_type::learn, explore_type::predict, all.get_setupfn_name(setup) + "-first")
                .set_params_per_weight(problem_multiplier)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .set_finish_example(explore_type::finish_multiline_example)
                .set_print_example(explore_type::print_multiline_example)
                .set_persist_metrics(explore_type::persist_metrics)
                .set_save_load(explore_type::save_load)
                .build();
  return make_base(*l);
}

}  // namespace first
}  // namespace cb_explore_adf
}  // namespace VW