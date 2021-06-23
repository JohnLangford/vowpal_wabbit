// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "decision_scores.h"

#include "slates.h"
#include "conditional_contextual_bandit.h"
#include "action_score.h"
#include "io_buf.h"
#include "global_data.h"
#include "shared_data.h"
#include "io/io_adapter.h"

#include "io/logger.h"

#include <iostream>

namespace logger = VW::io::logger;

template <typename LabelPrintFunc>
void print_update(vw& all, const std::vector<example*>& slots, const VW::decision_scores_t& decision_scores,
    size_t num_features, LabelPrintFunc label_print_func)
{
  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.logger.quiet && !all.bfgs)
  {
    std::ostringstream label_buf;
    label_buf << std::setw(shared_data::col_current_label) << std::right << std::setfill(' ')
              << label_print_func(slots);

    std::stringstream pred_ss;
    std::string delim;
    size_t counter = 0;
    for (const auto& slot : decision_scores)
    {
      counter++;
      pred_ss << delim << slot[0].action;
      delim = ",";

      // Stop after 3...
      if (counter > 2)
      {
        pred_ss << delim << "...";
        break;
      }
    }
    std::ostringstream pred_buf;
    pred_buf << std::setw(shared_data::col_current_predict) << std::right << std::setfill(' ') << pred_ss.str();

    all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass, label_buf.str(), pred_buf.str(),
        num_features, all.progress_add, all.progress_arg);
  }
}

namespace VW
{
void print_decision_scores(VW::io::writer* f, const VW::decision_scores_t& decision_scores)
{
  if (f != nullptr)
  {
    std::stringstream ss;
    for (const auto& slot : decision_scores)
    {
      std::string delimiter;
      for (const auto& action_score : slot)
      {
        ss << delimiter << action_score.action << ':' << action_score.score;
        delimiter = ",";
      }
      ss << '\n';
    }
    const auto str = ss.str();
    ssize_t len = str.size();
    ssize_t t = f->write(str.c_str(), static_cast<unsigned int>(len));
    if (t != len) { logger::errlog_error("write error: {}", VW::strerror_to_string(errno)); }
  }
}

void print_update_ccb(
    vw& all, std::vector<example*>& slots, const VW::decision_scores_t& decision_scores, size_t num_features)
{
  print_update(all, slots, decision_scores, num_features, CCB::generate_ccb_label_printout);
}

void print_update_slates(
    vw& all, std::vector<example*>& slots, const VW::decision_scores_t& decision_scores, size_t num_features)
{
  print_update(all, slots, decision_scores, num_features, slates::generate_slates_label_printout);
}
}  // namespace VW
