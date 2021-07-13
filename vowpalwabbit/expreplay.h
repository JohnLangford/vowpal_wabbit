// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include "learner.h"
#include "vw.h"
#include "parse_args.h"
#include "rand48.h"
#include "rand_state.h"

#include <memory>

namespace ExpReplay
{
template <label_parser& lp>
struct expreplay
{
  vw* all;
  std::shared_ptr<rand_state> _random_state;
  size_t N;             // how big is the buffer?
  example* buf;         // the deep copies of examples (N of them)
  bool* filled;         // which of buf[] is filled
  size_t replay_count;  // each time er.learn() is called, how many times do we call base.learn()? default=1 (in which
                        // case we're just permuting)
  VW::LEARNER::single_learner* base;

  ~expreplay()
  {
    VW::dealloc_examples(buf, N);
    free(filled);
  }
};

template <label_parser &lp>
void learn(expreplay<lp> &er, LEARNER::single_learner &base, example &ec)
{
  // Cannot learn if the example weight is 0.
  if (lp.get_weight(&ec.l, ec._reduction_features) == 0.) return;

  for (size_t replay = 1; replay < er.replay_count; replay++)
  {
    size_t n = (size_t)(er._random_state->get_and_update_random() * (float)er.N);
    if (er.filled[n]) base.learn(er.buf[n]);
  }

  size_t n = (size_t)(er._random_state->get_and_update_random() * (float)er.N);
  if (er.filled[n]) base.learn(er.buf[n]);

  er.filled[n] = true;
  VW::copy_example_data_with_label(&er.buf[n], &ec);
}

template <label_parser &lp>
void predict(expreplay<lp> &, LEARNER::single_learner &base, example &ec)
{
  base.predict(ec);
}

template <label_parser &lp>
void multipredict(expreplay<lp> &, LEARNER::single_learner &base, example &ec, size_t count, size_t step,
    polyprediction *pred, bool finalize_predictions)
{
  base.multipredict(ec, count, step, pred, finalize_predictions);
}

template <label_parser& lp>
void end_pass(expreplay<lp>& er)
{  // we need to go through and learn on everyone who remains
  // also need to clean up remaining examples
  for (size_t n = 0; n < er.N; n++)
    if (er.filled[n])
    {  // TODO: if er.replay_count > 1 do we need to play these more?
      er.base->learn(er.buf[n]);
      er.filled[n] = false;
    }
}

template <char er_level, label_parser& lp>
VW::LEARNER::base_learner* expreplay_setup(VW::setup_base_fn& setup_base, VW::config::options_i& options, vw& all)
{
  std::string replay_string = "replay_";
  replay_string += er_level;
  std::string replay_count_string = replay_string;
  replay_count_string += "_count";

  auto er = scoped_calloc_or_throw<expreplay<lp>>();
  VW::config::option_group_definition new_options("Experience Replay / " + replay_string);
  new_options
      .add(VW::config::make_option(replay_string, er->N)
               .keep()
               .necessary()
               .help("use experience replay at a specified level [b=classification/regression, m=multiclass, c=cost "
                     "sensitive] with specified buffer size"))
      .add(VW::config::make_option(replay_count_string, er->replay_count)
               .default_value(1)
               .help("how many times (in expectation) should each example be played (default: 1 = permuting)"));

  if (!options.add_parse_and_check_necessary(new_options) || er->N == 0) return nullptr;

  er->all = &all;
  er->_random_state = all.get_random_state();
  er->buf = VW::alloc_examples(er->N);
  er->buf->interactions = &all.interactions;
  er->filled = calloc_or_throw<bool>(er->N);

  if (!all.logger.quiet)
    *(all.trace_message) << "experience replay level=" << er_level << ", buffer=" << er->N << ", replay count=" << er->replay_count
              << std::endl;

  er->base = VW::LEARNER::as_singleline(setup_base(options, all));
  VW::LEARNER::learner<expreplay<lp>, example> *l = &init_learner(er, er->base, learn<lp>, predict<lp>, replay_string);
  l->set_end_pass(end_pass<lp>);

  return make_base(*l);
}
}  // namespace ExpReplay
