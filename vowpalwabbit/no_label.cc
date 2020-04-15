// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cstring>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "vw_string_view.h"

#include "cache.h"
#include "accumulate.h"
#include "best_constant.h"
#include "vw_string_view.h"

namespace no_label
{
size_t read_cached_no_label(shared_data*, polylabel& label, io_buf&)
{
  if (label.get_type() != label_type_t::empty)
  {
    label.reset();
    label.init_as_empty();
  }
  return 1;
}

float get_weight(polylabel&) { return 1.; }

void cache_no_label(polylabel&, io_buf&) {}

// This is wasted work, ideally empty and unset should be the same thing.
void default_no_label(polylabel& label)
{
  if (label.get_type() != label_type_t::empty)
  {
    label.reset();
    label.init_as_empty();
  }
}

bool test_label(polylabel&) { return false; }

void parse_no_label(parser*, shared_data*, polylabel&, v_array<VW::string_view>& words)
{
  switch (words.size())
  {
    case 0:
      break;
    default:
      std::cout << "Error: " << words.size() << " is too many tokens for a simple label: ";
      for (const auto & word : words) std::cout << word;
      std::cout << std::endl;
  }
}

label_parser no_label_parser = {default_no_label, parse_no_label, cache_no_label, read_cached_no_label, polylabel_delete_label,
    get_weight, polylabel_copy_label, test_label, sizeof(nullptr)};

void print_no_label_update(vw& all, example& ec)
{
  if (all.sd->weighted_labeled_examples + all.sd->weighted_unlabeled_examples >= all.sd->dump_interval &&
      !all.logger.quiet && !all.bfgs)
  {
    all.sd->print_update(all.holdout_set_off, all.current_pass, 0.f, ec.pred.scalar(), ec.num_features, all.progress_add,
        all.progress_arg);
  }
}

void output_and_account_no_label_example(vw& all, example& ec)
{
  all.sd->update(ec.test_only, false, ec.loss, ec.weight, ec.num_features);

  all.print_by_ref(all.raw_prediction, ec.partial_prediction, -1, ec.tag);
  for (auto& sink : all.final_prediction_sink)
  {
    all.print_by_ref(sink, ec.pred.scalar(), 0, ec.tag);
  }

  print_no_label_update(all, ec);
}

void return_no_label_example(vw& all, polylabel&, example& ec)
{
  output_and_account_example(all, ec);
  VW::finish_example(all, ec);
}
}  // namespace no_label
