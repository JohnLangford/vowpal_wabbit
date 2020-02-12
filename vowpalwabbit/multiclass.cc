// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cstring>
#include <climits>
#include "global_data.h"
#include "vw.h"
#include "vw_exception.h"
#include "vw_string_view.h"

namespace MULTICLASS
{
char* bufread_label(label_t& ld, char* c)
{
  memcpy(&ld.label, c, sizeof(ld.label));
  c += sizeof(ld.label);
  memcpy(&ld.weight, c, sizeof(ld.weight));
  c += sizeof(ld.weight);
  return c;
}

size_t read_cached_label(shared_data*, polylabel& v, io_buf& cache)
{
  auto& ld = v.multi();
  char* c;
  size_t total = sizeof(ld.label) + sizeof(ld.weight);
  if (cache.buf_read(c, total) < total)
    return 0;
  bufread_label(ld, c);

  return total;
}

float weight(polylabel& v)
{
 auto& ld = v.multi();
  return (ld.weight > 0) ? ld.weight : 0.f;
}

char* bufcache_label(label_t& ld, char* c)
{
  memcpy(c, &ld.label, sizeof(ld.label));
  c += sizeof(ld.label);
  memcpy(c, &ld.weight, sizeof(ld.weight));
  c += sizeof(ld.weight);
  return c;
}

void cache_label(polylabel& v, io_buf& cache)
{
  char* c;
 auto& ld = v.multi();
  cache.buf_write(c, sizeof(ld.label) + sizeof(ld.weight));
  bufcache_label(ld, c);
}

void default_label(polylabel& v)
{
  if (v.get_type() != label_type_t::multi)
  {
    v.reset();
    v.init_as_multi();
  }
  auto& ld = v.multi();
  ld.label = (uint32_t)-1;
  ld.weight = 1.;
}

bool test_label(polylabel& v)
{
 auto& ld = v.multi();
  return ld.label == (uint32_t)-1;
}

void parse_label(parser*, shared_data* sd, polylabel& v, v_array<VW::string_view>& words)
{
 auto& ld = v.multi();

  switch (words.size())
  {
    case 0:
      break;
    case 1:
      ld.label = sd->ldict ? (uint32_t)sd->ldict->get(words[0]) : int_of_string(words[0]);
      ld.weight = 1.0;
      break;
    case 2:
      ld.label = sd->ldict ? (uint32_t)sd->ldict->get(words[0]) : int_of_string(words[0]);
      ld.weight = float_of_string(words[1]);
      break;
    default:
      std::cerr << "malformed example!\n";
      std::cerr << "words.size() = " << words.size() << std::endl;
  }
  if (ld.label == 0)
    THROW("label 0 is not allowed for multiclass.  Valid labels are {1,k}"
        << (sd->ldict ? "\nthis likely happened because you specified an invalid label with named labels" : ""));
}

label_parser mc_label = {default_label, parse_label, cache_label, read_cached_label, polylabel_delete_label, weight, polylabel_copy_label, test_label, sizeof(label_t)};

void print_label_pred(vw& all, example& ec, uint32_t prediction)
{
  VW::string_view sv_label = all.sd->ldict->get(ec.l.multi().label);
  VW::string_view sv_pred = all.sd->ldict->get(prediction);
  all.sd->print_update(all.holdout_set_off, all.current_pass,
      sv_label.empty() ? "unknown" : sv_label.to_string(),
      sv_pred.empty() ? "unknown" : sv_pred.to_string(), ec.num_features,
      all.progress_add, all.progress_arg);
}

void print_probability(vw& all, example& ec, uint32_t prediction)
{
  std::stringstream pred_ss;
  pred_ss << prediction << "(" << std::setw(2) << std::setprecision(0) << std::fixed
          << 100 * ec.pred.scalars()[prediction - 1] << "%)";

  std::stringstream label_ss;
  label_ss << ec.l.multi().label;

  all.sd->print_update(all.holdout_set_off, all.current_pass, label_ss.str(), pred_ss.str(), ec.num_features,
      all.progress_add, all.progress_arg);
}

void print_score(vw& all, example& ec, uint32_t prediction)
{
  std::stringstream pred_ss;
  pred_ss << prediction;

  std::stringstream label_ss;
  label_ss << ec.l.multi().label;

  all.sd->print_update(all.holdout_set_off, all.current_pass, label_ss.str(), pred_ss.str(), ec.num_features,
      all.progress_add, all.progress_arg);
}

void direct_print_update(vw& all, example& ec, uint32_t prediction)
{
  all.sd->print_update(all.holdout_set_off, all.current_pass, ec.l.multi().label, prediction, ec.num_features,
      all.progress_add, all.progress_arg);
}

template <void (*T)(vw&, example&, uint32_t)>
void print_update(vw& all, example& ec, uint32_t prediction)
{
  if (all.sd->weighted_examples() >= all.sd->dump_interval && !all.quiet && !all.bfgs)
  {
    if (!all.sd->ldict)
      T(all, ec, prediction);
    else
      print_label_pred(all, ec, ec.pred.multiclass());
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
  if (ec.l.multi().label != (uint32_t)ec.pred.multiclass() && ec.l.multi().label != (uint32_t)-1)
    loss = ec.weight;

  all.sd->update(ec.test_only, update_loss && (ec.l.multi().label != (uint32_t)-1), loss, ec.weight, ec.num_features);

  for (int sink : all.final_prediction_sink)
    if (!all.sd->ldict)
      all.print_by_ref(sink, (float)ec.pred.multiclass(), 0, ec.tag);
    else
    {
      VW::string_view sv_pred = all.sd->ldict->get(ec.pred.multiclass());
      all.print_text_by_ref(sink, sv_pred.to_string(), ec.tag);
    }

  MULTICLASS::print_update<direct_print_update>(all, ec, ec.pred.multiclass());
  VW::finish_example(all, ec);
}
}  // namespace MULTICLASS
