// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#include <sstream>
#include <cfloat>
#include "generate_interactions.h"
#include "interactions.h"
#include "reductions.h"
#include "v_array.h"

#include "io/logger.h"
#include "interactions.h"
#include "vw_math.h"
#include <algorithm>
#include <iterator>

using namespace VW::config;

template <bool is_learn, INTERACTIONS::generate_func_t generate_func, bool leave_duplicate_interactions>
void transform_single_ex(INTERACTIONS::interactions_generator& data, VW::LEARNER::single_learner& base, example& ec)
{
  // We pass *ec.interactions here BUT the contract is that this does not change...
  data.update_interactions_if_new_namespace_seen<generate_func, leave_duplicate_interactions>(
      *ec.interactions, ec.indices);

  auto* saved_interactions = ec.interactions;
  ec.interactions = &data.generated_interactions;

  if (is_learn) { base.learn(ec); }
  else
  {
    base.predict(ec);
  }
  ec.interactions = saved_interactions;
}

template <INTERACTIONS::generate_func_t generate_func, bool leave_duplicate_interactions>
void update(INTERACTIONS::interactions_generator& data, VW::LEARNER::single_learner& base, example& ec)
{
  // We pass *ec.interactions here BUT the contract is that this does not change...
  data.update_interactions_if_new_namespace_seen<generate_func, leave_duplicate_interactions>(
      *ec.interactions, ec.indices);

  auto* saved_interactions = ec.interactions;
  ec.interactions = &data.generated_interactions;
  base.update(ec);
  ec.interactions = saved_interactions;
}

template <INTERACTIONS::generate_func_t generate_func, bool leave_duplicate_interactions>
inline void multipredict(INTERACTIONS::interactions_generator& data, VW::LEARNER::single_learner& base, example& ec,
    size_t count, size_t, polyprediction* pred, bool finalize_predictions)
{
  // We pass *ec.interactions here BUT the contract is that this does not change...
  data.update_interactions_if_new_namespace_seen<generate_func, leave_duplicate_interactions>(
      *ec.interactions, ec.indices);

  auto* saved_interactions = ec.interactions;
  ec.interactions = &data.generated_interactions;
  base.multipredict(ec, 0, count, pred, finalize_predictions);
  ec.interactions = saved_interactions;
}

VW::LEARNER::base_learner* generate_interactions_setup(VW::setup_base_fn& setup_base, options_i& options, vw& all)
{
  bool leave_duplicate_interactions;
  option_group_definition new_options("Generate interactions");
  new_options.add(make_option("leave_duplicate_interactions", leave_duplicate_interactions)
                      .help("Don't remove interactions with duplicate combinations of namespaces. For ex. this is a "
                            "duplicate: '-q ab -q ba' and a lot more in '-q ::'."));
  options.add_and_parse(new_options);

  auto interactions_spec_contains_wildcards = false;
  for (const auto& inter : all.interactions)
  {
    if (INTERACTIONS::contains_wildcard(inter))
    {
      interactions_spec_contains_wildcards = true;
      break;
    }
  }

  // If there are no wildcards, then no expansion is required.
  // ccb_explore_adf adds a wildcard post setup and so this reduction must be turned on.
  if (!(interactions_spec_contains_wildcards || options.was_supplied("ccb_explore_adf"))) { return nullptr; }

  using learn_pred_func_t = void (*)(INTERACTIONS::interactions_generator&, VW::LEARNER::single_learner&, example&);
  using multipredict_func_t = void (*)(INTERACTIONS::interactions_generator&, VW::LEARNER::single_learner&, example&,
      size_t, size_t, polyprediction*, bool);
  learn_pred_func_t learn_func;
  learn_pred_func_t pred_func;
  learn_pred_func_t update_func;
  multipredict_func_t multipredict_func;

  if (leave_duplicate_interactions)
  {
    learn_func = transform_single_ex<true, INTERACTIONS::generate_namespace_permutations_with_repetition, true>;
    pred_func = transform_single_ex<false, INTERACTIONS::generate_namespace_permutations_with_repetition, true>;
    update_func = update<INTERACTIONS::generate_namespace_permutations_with_repetition, true>;
    multipredict_func = multipredict<INTERACTIONS::generate_namespace_permutations_with_repetition, true>;
  }
  else
  {
    learn_func = transform_single_ex<true, INTERACTIONS::generate_namespace_combinations_with_repetition, false>;
    pred_func = transform_single_ex<false, INTERACTIONS::generate_namespace_combinations_with_repetition, false>;
    update_func = update<INTERACTIONS::generate_namespace_combinations_with_repetition, false>;
    multipredict_func = multipredict<INTERACTIONS::generate_namespace_combinations_with_repetition, false>;
  }

  auto data = VW::make_unique<INTERACTIONS::interactions_generator>();
  auto* base = as_singleline(setup_base(options, all));
  auto* l = VW::LEARNER::make_reduction_learner(
      std::move(data), base, learn_func, pred_func, all.get_setupfn_name(generate_interactions_setup))
                .set_learn_returns_prediction(base->learn_returns_prediction)
                .set_update(update_func)
                .set_multipredict(multipredict_func)
                .build();
  return VW::LEARNER::make_base(*l);
}
