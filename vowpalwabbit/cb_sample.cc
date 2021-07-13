// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "cb_sample.h"
#include "explore.h"
#include "label_parser.h"

#include "rand48.h"
#include "vw_string_view.h"
#include "tag_utils.h"

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::cb_sample

using namespace VW::LEARNER;
using namespace VW;
using namespace VW::config;

namespace VW
{
// cb_sample is used to automatically sample and swap from a cb explore pdf.
struct cb_sample_data
{
  explicit cb_sample_data(std::shared_ptr<rand_state> &random_state) : _random_state(random_state) {}
  explicit cb_sample_data(std::shared_ptr<rand_state> &&random_state) : _random_state(random_state) {}

  template <bool is_learn>
  inline void learn_or_predict(multi_learner &base, multi_ex &examples)
  {
    // If base.learn() does not return prediction then we need to predict first
    // so that there is something to sample from
    if (is_learn && !base.learn_returns_prediction)
      multiline_learn_or_predict<false>(base, examples, examples[0]->ft_offset);

    multiline_learn_or_predict<is_learn>(base, examples, examples[0]->ft_offset);

    auto &action_scores = examples[0]->pred.a_s;

    uint32_t chosen_action = 0;
    int64_t maybe_labelled_action = -1;

    // Find that chosen action in the learning case, skip the shared example.
    auto it = std::find_if(examples.begin(), examples.end(), [](example *item) { return !item->l.cb.costs.empty(); });
    if (it != examples.end()) { maybe_labelled_action = static_cast<int64_t>(std::distance(examples.begin(), it)); }

    // If we are learning and have a label, then take that action as the chosen action. Otherwise sample the
    // distribution.
    bool learn = is_learn && maybe_labelled_action >= 0;
    if (learn)
    {
      // Find where the labelled action is in the final prediction to determine if swapping needs to occur.
      // This only matters if the prediction decided to explore, but the same output should happen for the learn case.
      for (size_t i = 0; i < action_scores.size(); i++)
      {
        auto &a_s = action_scores[i];
        if (a_s.action == static_cast<uint32_t>(maybe_labelled_action))
        {
          chosen_action = static_cast<uint32_t>(i);
          break;
        }
      }
    }
    else
    {
      uint64_t seed = _random_state->get_current_state();

      VW::string_view tag_seed;
      const bool tag_provided_seed = try_extract_random_seed(*examples[0], tag_seed);
      if (tag_provided_seed) { seed = uniform_hash(tag_seed.begin(), tag_seed.size(), 0); }

      // Sampling is done after the base learner has generated a pdf.
      auto result = exploration::sample_after_normalizing(
          seed, ACTION_SCORE::begin_scores(action_scores), ACTION_SCORE::end_scores(action_scores), chosen_action);
      assert(result == S_EXPLORATION_OK);
      _UNUSED(result);

      // Update the seed state in place if it was used for this example.
      if (!tag_provided_seed) { _random_state->get_and_update_random(); }
    }

    auto result = exploration::swap_chosen(action_scores.begin(), action_scores.end(), chosen_action);
    assert(result == S_EXPLORATION_OK);

    VW_DBG(examples) << "cb " << cb_decision_to_string(examples[0]->pred.a_s)
                     << " rnd:" << _random_state->get_current_state() << std::endl;

    _UNUSED(result);
  }

  std::string cb_decision_to_string(const ACTION_SCORE::action_scores &action_scores)
  {
    std::ostringstream ostrm;
    if (action_scores.empty()) return "";
    ostrm << "chosen" << action_scores[0] << action_scores;
    return ostrm.str();
  }

private:
  std::shared_ptr<rand_state> _random_state;
};
}  // namespace VW

template <bool is_learn>
void learn_or_predict(cb_sample_data &data, multi_learner &base, multi_ex &examples)
{
  data.learn_or_predict<is_learn>(base, examples);
}

base_learner* cb_sample_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  bool cb_sample_option = false;

  option_group_definition new_options("CB Sample");
  new_options.add(
      make_option("cb_sample", cb_sample_option).keep().necessary().help("Sample from CB pdf and swap top action."));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  auto data = VW::make_unique<cb_sample_data>(all.get_random_state());

  auto* l = make_reduction_learner(std::move(data), as_multiline(setup_base(options, all)), learn_or_predict<true>,
      learn_or_predict<false>, all.get_setupfn_name(cb_sample_setup))
                .set_learn_returns_prediction(true)
                .set_prediction_type(prediction_type_t::action_probs)
                .set_label_type(label_type_t::cb)
                .build();
  return make_base(*l);
}
