// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <cstdio>
#include <cfloat>
#include <cerrno>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cassert>

#include "global_data.h"
#include "gd.h"
#include "vw_exception.h"
#include "future_compat.h"
#include "vw_allreduce.h"
#include "named_labels.h"

struct global_prediction
{
  float p;
  float weight;
};

size_t really_read(VW::io::reader* sock, void* in, size_t count)
{
  char* buf = (char*)in;
  size_t done = 0;
  ssize_t r = 0;
  while (done < count)
  {
    if ((r = sock->read(buf, static_cast<unsigned int>(count - done))) == 0)
    {
      return 0;
    }
    else if (r < 0)
    {
      THROWERRNO("read(" << sock << "," << count << "-" << done << ")");
    }
    else
    {
      done += r;
      buf += r;
    }
  }
  return done;
}

void get_prediction(VW::io::reader* f, float& res, float& weight)
{
  global_prediction p;
  really_read(f, &p, sizeof(p));
  res = p.p;
  weight = p.weight;
}

void send_prediction(VW::io::writer* f, global_prediction p)
{
  if (f->write(reinterpret_cast<const char*>(&p), sizeof(p)) < static_cast<int>(sizeof(p)))
    THROWERRNO("send_prediction write(unknown socket fd)");
}

void binary_print_result(VW::io::writer* f, float res, float weight, v_array<char> array)
{
  binary_print_result_by_ref(f, res, weight, array);
}

void binary_print_result_by_ref(VW::io::writer* f, float res, float weight, const v_array<char>&)
{
  if (f != nullptr)
  {
    global_prediction ps = {res, weight};
    send_prediction(f, ps);
  }
}

int print_tag_by_ref(std::stringstream& ss, const v_array<char>& tag)
{
  if (tag.begin() != tag.end())
  {
    ss << ' ';
    ss.write(tag.begin(), sizeof(char) * tag.size());
  }
  return tag.begin() != tag.end();
}

int print_tag(std::stringstream& ss, v_array<char> tag)
{
  return print_tag_by_ref(ss, tag);
}

void print_result(VW::io::writer* f, float res, float unused, v_array<char> tag)
{
  print_result_by_ref(f, res, unused, tag);
}

void print_result_by_ref(VW::io::writer* f, float res, float, const v_array<char>& tag)
{
  if (f != nullptr)
  {
    std::stringstream ss;
    auto saved_precision = ss.precision();
    if (floorf(res) == res)
      ss << std::setprecision(0);
    ss << std::fixed << res << std::setprecision(saved_precision);
    print_tag_by_ref(ss, tag);
    ss << '\n';
    ssize_t len = ss.str().size();
    ssize_t t = f->write(ss.str().c_str(), (unsigned int)len);
    if (t != len)
    {
      std::cerr << "write error: " << VW::strerror_to_string(errno) << std::endl;
    }
  }
}

void print_raw_text(VW::io::writer* f, std::string s, v_array<char> tag)
{
  if (f == nullptr)
    return;

  std::stringstream ss;
  ss << s;
  print_tag_by_ref(ss, tag);
  ss << '\n';
  ssize_t len = ss.str().size();
  ssize_t t = f->write(ss.str().c_str(), (unsigned int)len);
  if (t != len)
  {
    std::cerr << "write error: " << VW::strerror_to_string(errno) << std::endl;
  }
}

void print_raw_text_by_ref(VW::io::writer* f, const std::string& s, const v_array<char>& tag)
{
  if (f == nullptr)
    return;

  std::stringstream ss;
  ss << s;
  print_tag_by_ref(ss, tag);
  ss << '\n';
  ssize_t len = ss.str().size();
  ssize_t t = f->write(ss.str().c_str(), (unsigned int)len);
  if (t != len)
  {
    std::cerr << "write error: " << VW::strerror_to_string(errno) << std::endl;
  }
}


void set_mm(shared_data* sd, float label)
{
  sd->min_label = std::min(sd->min_label, label);
  if (label != FLT_MAX)
    sd->max_label = std::max(sd->max_label, label);
}

void noop_mm(shared_data*, float) {}

void vw::learn(example& ec)
{
  if (l->is_multiline)
    THROW("This reduction does not support single-line examples.");

  if (ec.test_only || !training)
    VW::LEARNER::as_singleline(l)->predict(ec);
  else
  {
    if (l->learn_returns_prediction)
    {
      VW::LEARNER::as_singleline(l)->learn(ec);
    }
    else
    {
      VW::LEARNER::as_singleline(l)->predict(ec);
      std::swap(_predict_buffer, ec.pred);
      auto restore_guard = VW::scope_exit([&ec, this] {
        std::swap(ec.pred, _predict_buffer);
        });
      VW::LEARNER::as_singleline(l)->learn(ec);
    }
  }
}

void vw::learn(multi_ex& ec)
{
  if (!l->is_multiline)
    THROW("This reduction does not support multi-line example.");

  if (!training)
    VW::LEARNER::as_multiline(l)->predict(ec);
  else
  {
    if (l->learn_returns_prediction)
    {
      VW::LEARNER::as_multiline(l)->learn(ec);
    }
    else
    {
      VW::LEARNER::as_multiline(l)->predict(ec);
      copy_prediction(ec[0]->pred);
      auto restore_guard = VW::scope_exit([&ec, this] { std::swap(ec[0]->pred, _predict_buffer); });
      VW::LEARNER::as_multiline(l)->learn(ec);
    }
  }
}

void vw::predict(example& ec)
{
  if (l->is_multiline)
    THROW("This reduction does not support single-line examples.");

  // be called directly in library mode, test_only must be explicitly set here. If the example has a label but is passed
  // to predict it would otherwise be incorrectly labelled as test_only = false.
  ec.test_only = true;
  VW::LEARNER::as_singleline(l)->predict(ec);
}

void vw::predict(multi_ex& ec)
{
  if (!l->is_multiline)
    THROW("This reduction does not support multi-line example.");

  // be called directly in library mode, test_only must be explicitly set here. If the example has a label but is passed
  // to predict it would otherwise be incorrectly labelled as test_only = false.
  for (auto& ex : ec)
  {
    ex->test_only = true;
  }

  VW::LEARNER::as_multiline(l)->predict(ec);
}

void vw::finish_example(example& ec)
{
  if (l->is_multiline)
    THROW("This reduction does not support single-line examples.");

  VW::LEARNER::as_singleline(l)->finish_example(*this, ec);
}

void vw::finish_example(multi_ex& ec)
{
  if (!l->is_multiline)
    THROW("This reduction does not support multi-line example.");

  VW::LEARNER::as_multiline(l)->finish_example(*this, ec);
}

void compile_limits(std::vector<std::string> limits, std::array<uint32_t, NUM_NAMESPACES>& dest, bool quiet)
{
  for (size_t i = 0; i < limits.size(); i++)
  {
    std::string limit = limits[i];
    if (isdigit(limit[0]))
    {
      int n = atoi(limit.c_str());
      if (!quiet)
        std::cerr << "limiting to " << n << "features for each namespace." << std::endl;
      for (size_t j = 0; j < 256; j++) dest[j] = n;
    }
    else if (limit.size() == 1)
      std::cout << "You must specify the namespace index before the n" << std::endl;
    else
    {
      int n = atoi(limit.c_str() + 1);
      dest[(uint32_t)limit[0]] = n;
      if (!quiet)
        std::cerr << "limiting to " << n << " for namespaces " << limit[0] << std::endl;
    }
  }
}

void trace_listener_cerr(void*, const std::string& message)
{
  std::cerr << message;
  std::cerr.flush();
}

int vw_ostream::vw_streambuf::sync()
{
  int ret = std::stringbuf::sync();
  if (ret)
    return ret;

  parent.trace_listener(parent.trace_context, str());
  str("");
  return 0;  // success
}

vw_ostream::vw_ostream() : std::ostream(&buf), buf(*this), trace_context(nullptr)
{
  trace_listener = trace_listener_cerr;
}

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_DEPRECATED_USAGE

void vw::copy_prediction(const polyprediction& from_pred)
{
  if (l == nullptr) return;

  switch (l->pred_type)
  {
    case prediction_type_t::action_probs:
      copy_array(_predict_buffer.a_s, from_pred.a_s);
      break;
    case prediction_type_t::action_scores:
      copy_array(_predict_buffer.a_s, from_pred.a_s);
      break;
    case prediction_type_t::decision_probs:
      copy_array(_predict_buffer.decision_scores, from_pred.decision_scores);
      break;
    case prediction_type_t::multilabels:
      copy_array(_predict_buffer.multilabels.label_v, from_pred.multilabels.label_v);
      break;
    case prediction_type_t::pdf:
      copy_array(_predict_buffer.pdf, from_pred.pdf);
      break;
    case prediction_type_t::scalars:
      copy_array(_predict_buffer.scalars, from_pred.scalars);
      break;
    case prediction_type_t::multiclass:
      _predict_buffer = from_pred;
      break;
    case prediction_type_t::action_pdf_value:
      _predict_buffer = from_pred;
      break;
    case prediction_type_t::prob:
      _predict_buffer = from_pred;
      break;
    case prediction_type_t::scalar:
      _predict_buffer = from_pred;
      break;
    default:
      THROW("Unhandled prediction type");
  }
}

void vw::cleanup_prediction()
{
  if (l == nullptr) return;

  switch (l->pred_type)
  {
    case prediction_type_t::action_probs:
      _predict_buffer.a_s.delete_v();
      break;
    case prediction_type_t::action_scores:
      _predict_buffer.a_s.delete_v();
      break;
    case prediction_type_t::decision_probs:
      _predict_buffer.decision_scores.delete_v();
      break;
    case prediction_type_t::multilabels:
      _predict_buffer.multilabels.label_v.delete_v();
      break;
    case prediction_type_t::pdf:
      _predict_buffer.pdf.delete_v();
      break;
    case prediction_type_t::scalars:
      _predict_buffer.scalars.delete_v();
      break;
    case prediction_type_t::multiclass:
      break;
    case prediction_type_t::action_pdf_value:
      break;
    case prediction_type_t::prob:
      break;
    case prediction_type_t::scalar:
      break;
    default:
      THROW("Unhandled prediction type");
  }
}

vw::vw()
{
  sd = &calloc_or_throw<shared_data>();
  sd->dump_interval = 1.;  // next update progress dump
  sd->contraction = 1.;
  sd->first_observed_label = FLT_MAX;
  sd->is_more_than_two_labels_observed = false;
  sd->max_label = 0;
  sd->min_label = 0;

  label_type = label_type_t::simple;

  l = nullptr;
  scorer = nullptr;
  cost_sensitive = nullptr;
  loss = nullptr;
  example_parser = nullptr;

  reg_mode = 0;
  current_pass = 0;

  data_filename = "";
  delete_prediction = nullptr;

  bfgs = false;
  no_bias = false;
  hessian_on = false;
  active = false;
  num_bits = 18;
  default_bits = true;
  daemon = false;
  num_children = 10;
  save_resume = false;
  preserve_performance_counters = false;

  random_positive_weights = false;

  weights.sparse = false;

  set_minmax = set_mm;

  power_t = 0.5;
  eta = 0.5;  // default learning rate for normalized adaptive updates, this is switched to 10 by default for the other
              // updates (see parse_args.cc)
  numpasses = 1;

  print = print_result;
  print_text = print_raw_text;
  print_by_ref = print_result_by_ref;
  print_text_by_ref = print_raw_text_by_ref;
  lda = 0;
  random_seed = 0;
  random_weights = false;
  normal_weights = false;
  tnormal_weights = false;
  per_feature_regularizer_input = "";
  per_feature_regularizer_output = "";
  per_feature_regularizer_text = "";

  stdout_adapter = VW::io::open_stdout();

  searchstr = nullptr;

  nonormalize = false;
  l1_lambda = 0.0;
  l2_lambda = 0.0;

  eta_decay_rate = 1.0;
  initial_weight = 0.0;
  initial_constant = 0.0;

  all_reduce = nullptr;

  for (size_t i = 0; i < NUM_NAMESPACES; i++)
  {
    limit[i] = INT_MAX;
    affix_features[i] = 0;
    spelling_features[i] = 0;
  }

  invariant_updates = true;
  normalized_idx = 2;

  add_constant = true;
  audit = false;

  pass_length = std::numeric_limits<size_t>::max();
  passes_complete = 0;

  save_per_pass = false;

  stdin_off = false;
  do_reset_source = false;
  holdout_set_off = true;
  holdout_after = 0;
  check_holdout_every_n_passes = 1;
  early_terminate = false;

  max_examples = std::numeric_limits<size_t>::max();

  hash_inv = false;
  print_invert = false;

  // Set by the '--progress <arg>' option and affect sd->dump_interval
  progress_add = false;  // default is multiplicative progress dumps
  progress_arg = 2.0;    // next update progress dump multiplier

  sd->is_more_than_two_labels_observed = false;
  sd->first_observed_label = FLT_MAX;
  sd->second_observed_label = FLT_MAX;

  sd->report_multiclass_log_loss = false;
  sd->multiclass_log_loss = 0;
  sd->holdout_multiclass_log_loss = 0;
  std::memset(&_predict_buffer, 0, sizeof(_predict_buffer));
}
VW_WARNING_STATE_POP

vw::~vw()
{
  cleanup_prediction();

  if (l != nullptr)
  {
    l->finish();
    free(l);
  }

  // Check if options object lifetime is managed internally.
  if (should_delete_options)
    delete options;

  // TODO: migrate all finalization into parser destructor
  if (example_parser != nullptr)
  {
    free_parser(*this);
    delete example_parser;
  }

  const bool seeded = weights.seeded() > 0;
  if (!seeded)
  {
    if (sd->ldict)
    {
      sd->ldict->~named_labels();
      free(sd->ldict);
    }
    free(sd);
  }

  delete all_reduce;
}
