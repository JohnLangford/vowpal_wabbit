// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cstring>
#include <climits>
#include "global_data.h"
#include "vw.h"
#include "vw_exception.h"
#include "vw_string_view.h"
#include "example.h"
#include "parse_primitives.h"
#include "shared_data.h"

namespace MULTICLASS
{
// label_t::label_t() { reset_to_default(); }

// label_t::label_t(uint32_t label, float weight) : label(label), weight(weight) {}

// void label_t::reset_to_default()
// {
//   label = std::numeric_limits<uint32_t>::max();
//   weight = 1.f;
// }

void default_label(label_t& ld) { ld.reset_to_default(); }

void cache_label(const label_t& ld, io_buf& cache)
{
  char* c;
  cache.buf_write(c, sizeof(ld.label) + sizeof(ld.weight));
  memcpy(c, &ld.label, sizeof(ld.label));
  c += sizeof(ld.label);
  memcpy(c, &ld.weight, sizeof(ld.weight));
  c += sizeof(ld.weight);
}

size_t read_cached_label(shared_data*, label_t& ld, io_buf& cache)
{
  char* c;
  size_t total = sizeof(ld.label) + sizeof(ld.weight);
  if (cache.buf_read(c, total) < total) return 0;
  memcpy(&ld.label, c, sizeof(ld.label));
  c += sizeof(ld.label);
  memcpy(&ld.weight, c, sizeof(ld.weight));
  c += sizeof(ld.weight);
  return total;
}
float weight(label_t& ld) { return (ld.weight > 0) ? ld.weight : 0.f; }
bool test_label(const label_t& ld) { return ld.label == static_cast<uint32_t>(-1); }

void parse_label(parser*, shared_data* sd, label_t& ld, std::vector<VW::string_view>& words, reduction_features&)
{
  switch (words.size())
  {
    case 0:
      break;
    case 1:
      if (sd->ldict) { ld.label = sd->ldict->get(words[0]); }
      else
      {
        char* char_after_int = nullptr;
        ld.label = int_of_string(words[0], char_after_int);
        if (char_after_int != nullptr && *char_after_int != ' ' && *char_after_int != '\0')
        { THROW("malformed example: label has trailing character(s): " << *char_after_int); }
      }
      ld.weight = 1.0;
      break;
    case 2:
      if (sd->ldict) { ld.label = sd->ldict->get(words[0]); }
      else
      {
        char* char_after_int = nullptr;
        ld.label = int_of_string(words[0], char_after_int);
        if (char_after_int != nullptr && *char_after_int != ' ' && *char_after_int != '\0')
        { THROW("malformed example: label has trailing character(s): " << *char_after_int); }
      }
      ld.weight = float_of_string(words[1]);
      break;
    default:
      THROW("malformed example, words.size() = " << words.size());
  }
  if (ld.label == 0)
    THROW("label 0 is not allowed for multiclass.  Valid labels are {1,k}"
        << (sd->ldict ? "\nthis likely happened because you specified an invalid label with named labels" : ""));
}

// clang-format off
label_parser mc_label = {
  // default_label
  [](polylabel* v) { default_label(v->multi); },
  // parse_label
  [](parser* p, shared_data* sd, polylabel* v, std::vector<VW::string_view>& words, reduction_features& red_features) {
    parse_label(p, sd, v->multi, words, red_features);
  },
  // cache_label
  [](polylabel* v, reduction_features&, io_buf& cache) { cache_label(v->multi, cache); },
  // read_cached_label
  [](shared_data* sd, polylabel* v, reduction_features&, io_buf& cache) { return read_cached_label(sd, v->multi, cache); },
  // get_weight
  [](polylabel* v, const reduction_features&) { return weight(v->multi); },
  // test_label
  [](polylabel* v) { return test_label(v->multi); },
  label_type_t::multiclass
};
// clang-format on

void print_label_pred(vw& all, example& ec, uint32_t prediction)
{
  VW::string_view sv_label = all.sd->ldict->get(ec.l.multi.label);
  VW::string_view sv_pred = all.sd->ldict->get(prediction);
  all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass,
      sv_label.empty() ? "unknown" : sv_label.to_string(), sv_pred.empty() ? "unknown" : sv_pred.to_string(),
      ec.get_num_features(), all.progress_add, all.progress_arg);
}

void print_probability(vw& all, example& ec, uint32_t prediction)
{
  std::stringstream pred_ss;
  pred_ss << prediction << "(" << std::setw(2) << std::setprecision(0) << std::fixed
          << 100 * ec.pred.scalars[prediction - 1] << "%)";

  std::stringstream label_ss;
  label_ss << ec.l.multi.label;

  all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass, label_ss.str(), pred_ss.str(),
      ec.get_num_features(), all.progress_add, all.progress_arg);
}

void print_score(vw& all, example& ec, uint32_t prediction)
{
  std::stringstream pred_ss;
  pred_ss << prediction;

  std::stringstream label_ss;
  label_ss << ec.l.multi.label;

  all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass, label_ss.str(), pred_ss.str(),
      ec.get_num_features(), all.progress_add, all.progress_arg);
}

void direct_print_update(vw& all, example& ec, uint32_t prediction)
{
  all.sd->print_update(*all.trace_message, all.holdout_set_off, all.current_pass, ec.l.multi.label, prediction,
      ec.get_num_features(), all.progress_add, all.progress_arg);
}

template <void (*T)(vw&, example&, uint32_t)>
void print_update(vw& all, example& ec, uint32_t prediction)
{
  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.logger.quiet && !all.bfgs)
  {
    if (!all.sd->ldict)
      T(all, ec, prediction);
    else
      print_label_pred(all, ec, ec.pred.multiclass);
  }
}

void print_update_with_probability(vw& all, example& ec, uint32_t pred)
{
  print_update<print_probability>(all, ec, pred);
}
void print_update_with_score(vw& all, example& ec, uint32_t pred) { print_update<print_score>(all, ec, pred); }

void finish_example(vw& all, example& ec, bool update_loss)
{
  float loss = 0;
  if (ec.l.multi.label != ec.pred.multiclass && ec.l.multi.label != static_cast<uint32_t>(-1)) loss = ec.weight;

  all.sd->update(ec.test_only, update_loss && (ec.l.multi.label != static_cast<uint32_t>(-1)), loss, ec.weight,
      ec.get_num_features());

  for (auto& sink : all.final_prediction_sink)
    if (!all.sd->ldict)
      all.print_by_ref(sink.get(), static_cast<float>(ec.pred.multiclass), 0, ec.tag);
    else
    {
      VW::string_view sv_pred = all.sd->ldict->get(ec.pred.multiclass);
      all.print_text_by_ref(sink.get(), sv_pred.to_string(), ec.tag);
    }

  MULTICLASS::print_update<direct_print_update>(all, ec, ec.pred.multiclass);
  VW::finish_example(all, ec);
}
}  // namespace MULTICLASS
