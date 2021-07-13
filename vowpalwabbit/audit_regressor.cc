// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "interactions.h"
#include "parse_args.h"
#include "vw.h"
#include "shared_data.h"
#include "gd.h"

using namespace VW::config;

struct audit_regressor_data
{
  vw* all;
  size_t increment;
  size_t cur_class;
  size_t total_class_cnt;
  std::vector<std::string>* ns_pre;
  io_buf* out_file;
  size_t loaded_regressor_values;
  size_t values_audited;
};

inline void audit_regressor_interaction(audit_regressor_data& dat, const audit_strings* f)
{
  // same as audit_interaction in gd.cc
  if (f == nullptr)
  {
    dat.ns_pre->pop_back();
    return;
  }

  std::string ns_pre;
  if (!dat.ns_pre->empty()) ns_pre += '*';

  if (f->first != "" && ((f->first) != " "))
  {
    ns_pre.append(f->first);
    ns_pre += '^';
  }
  if (f->second != "")
  {
    ns_pre.append(f->second);
    dat.ns_pre->push_back(ns_pre);
  }
}

inline void audit_regressor_feature(audit_regressor_data& dat, const float, const uint64_t ft_idx)
{
  parameters& weights = dat.all->weights;
  if (weights[ft_idx] != 0)
    ++dat.values_audited;
  else
    return;

  std::string ns_pre;
  for (std::vector<std::string>::const_iterator s = dat.ns_pre->begin(); s != dat.ns_pre->end(); ++s) ns_pre += *s;

  std::ostringstream tempstream;
  tempstream << ':' << ((ft_idx & weights.mask()) >> weights.stride_shift()) << ':' << weights[ft_idx];

  std::string temp = ns_pre + tempstream.str() + '\n';
  if (dat.total_class_cnt > 1)  // add class prefix for multiclass problems
    temp = std::to_string(dat.cur_class) + ':' + temp;

  dat.out_file->bin_write_fixed(temp.c_str(), static_cast<uint32_t>(temp.size()));

  weights[ft_idx] = 0.;  // mark value audited
}

void audit_regressor_lda(audit_regressor_data& rd, VW::LEARNER::single_learner& /* base */, example& ec)
{
  vw& all = *rd.all;

  std::ostringstream tempstream;
  parameters& weights = rd.all->weights;
  for (auto& bucket : ec)
  {
    for (const auto& feat_group : bucket)
    {
      for (size_t j = 0; j < feat_group.feats.size(); ++j)
      {
        tempstream << '\t' << feat_group.feats.space_names[j].first << '^' << feat_group.feats.space_names[j].second
                   << ':' << ((feat_group.feats.indicies[j] >> weights.stride_shift()) & all.parse_mask);
        for (size_t k = 0; k < all.lda; k++)
        {
          weight& w = weights[(feat_group.feats.indicies[j] + k)];
          tempstream << ':' << w;
          w = 0.;
        }
        tempstream << std::endl;
      }
    }
  }

  rd.out_file->bin_write_fixed(tempstream.str().c_str(), static_cast<uint32_t>(tempstream.str().size()));
}

// This is a learner which does nothing with examples.
// void learn(audit_regressor_data&, VW::LEARNER::base_learner&, example&) {}

void audit_regressor(audit_regressor_data& rd, VW::LEARNER::single_learner& base, example& ec)
{
  vw& all = *rd.all;

  if (all.lda > 0)
    audit_regressor_lda(rd, base, ec);
  else
  {
    rd.cur_class = 0;
    uint64_t old_offset = ec.ft_offset;

    while (rd.cur_class < rd.total_class_cnt)
    {
      for (const auto& bucket : ec)
      {
        for (const auto& feat_group : bucket)
        {
          if (feat_group.feats.space_names.size() > 0)
          {
            for (size_t j = 0; j < feat_group.feats.size(); ++j)
            {
              audit_regressor_interaction(rd, &feat_group.feats.space_names[j]);
              audit_regressor_feature(
                  rd, feat_group.feats.values[j], static_cast<uint32_t>(feat_group.feats.indicies[j]) + ec.ft_offset);
              audit_regressor_interaction(rd, nullptr);
            }
          }

          else
          {
            for (size_t j = 0; j < feat_group.feats.size(); ++j)
            {
              audit_regressor_feature(
                  rd, feat_group.feats.values[j], static_cast<uint32_t>(feat_group.feats.indicies[j]) + ec.ft_offset);
            }
          }
        }
      }

      size_t num_interacted_features = 0;
      if (rd.all->weights.sparse)
        INTERACTIONS::generate_interactions<audit_regressor_data, const uint64_t, audit_regressor_feature, true,
            audit_regressor_interaction, sparse_parameters>(rd.all->interactions, rd.all->permutations, ec, rd,
            rd.all->weights.sparse_weights, num_interacted_features);
      else
        INTERACTIONS::generate_interactions<audit_regressor_data, const uint64_t, audit_regressor_feature, true,
            audit_regressor_interaction, dense_parameters>(
            rd.all->interactions, rd.all->permutations, ec, rd, rd.all->weights.dense_weights, num_interacted_features);

      ec.ft_offset += rd.increment;
      ++rd.cur_class;
    }

    ec.ft_offset = old_offset;  // make sure example is not changed.
  }
}
void end_examples(audit_regressor_data& d)
{
  d.out_file->flush();  // close_file() should do this for me ...
  d.out_file->close_file();
  delete (d.out_file);
  d.out_file = nullptr;
  delete d.ns_pre;
  d.ns_pre = nullptr;
}

inline void print_ex(vw& all, size_t ex_processed, size_t vals_found, size_t progress)
{
  *(all.trace_message) << std::left << std::setw(shared_data::col_example_counter) << ex_processed << " " << std::right
                       << std::setw(9) << vals_found << " " << std::right << std::setw(12) << progress << '%'
                       << std::endl;
}

void finish_example(vw& all, audit_regressor_data& dd, example& ec)
{
  bool printed = false;
  if (ec.example_counter + 1 >= all.sd->dump_interval && !all.logger.quiet)
  {
    print_ex(all, ec.example_counter + 1, dd.values_audited, dd.values_audited * 100 / dd.loaded_regressor_values);
    all.sd->weighted_unlabeled_examples = static_cast<double>(ec.example_counter + 1);  // used in update_dump_interval
    all.sd->update_dump_interval(all.progress_add, all.progress_arg);
    printed = true;
  }

  if (dd.values_audited == dd.loaded_regressor_values)
  {
    // all regressor values were audited
    if (!printed) print_ex(all, ec.example_counter + 1, dd.values_audited, 100);
    set_done(all);
  }

  VW::finish_example(all, ec);
}

void finish(audit_regressor_data& dat)
{
  if (dat.values_audited < dat.loaded_regressor_values)
    *dat.all->trace_message << "Note: for some reason audit couldn't find all regressor values in dataset ("
                            << dat.values_audited << " of " << dat.loaded_regressor_values << " found)." << std::endl;
}

template <class T>
void regressor_values(audit_regressor_data& dat, T& w)
{
  for (typename T::iterator iter = w.begin(); iter != w.end(); ++iter)
    if (*iter != 0) dat.loaded_regressor_values++;
}

void init_driver(audit_regressor_data& dat)
{
  // checks a few settings that might be applied after audit_regressor_setup() is called
  if ((dat.all->options->was_supplied("cache_file") || dat.all->options->was_supplied("cache")) &&
      !dat.all->options->was_supplied("kill_cache"))
  { THROW("audit_regressor is incompatible with a cache file.  Use it in single pass mode only."); }

  dat.all->sd->dump_interval = 1.;  // regressor could initialize these if saved with --save_resume
  dat.all->sd->example_number = 0;

  dat.increment = dat.all->l->increment / dat.all->l->weights;
  dat.total_class_cnt = dat.all->l->weights;

  if (dat.all->options->was_supplied("csoaa"))
  {
    size_t n = dat.all->options->get_typed_option<uint32_t>("csoaa").value();
    if (n != dat.total_class_cnt)
    {
      dat.total_class_cnt = n;
      dat.increment = dat.all->l->increment / n;
    }
  }

  // count non-null feature values in regressor
  if (dat.all->weights.sparse)
    regressor_values(dat, dat.all->weights.sparse_weights);
  else
    regressor_values(dat, dat.all->weights.dense_weights);

  if (dat.loaded_regressor_values == 0) THROW("regressor has no non-zero weights. Nothing to audit.");

  if (!dat.all->logger.quiet)
  {
    *dat.all->trace_message << "Regressor contains " << dat.loaded_regressor_values << " values\n";
    *dat.all->trace_message << std::left << std::setw(shared_data::col_example_counter) << "example"
                            << " " << std::setw(shared_data::col_example_weight) << "values"
                            << " " << std::setw(shared_data::col_current_label) << "total" << std::endl;
    *dat.all->trace_message << std::left << std::setw(shared_data::col_example_counter) << "counter"
                            << " " << std::setw(shared_data::col_example_weight) << "audited"
                            << " " << std::setw(shared_data::col_current_label) << "progress" << std::endl;
  }
}

VW::LEARNER::base_learner* audit_regressor_setup(options_i& options, vw& all)
{
  std::string out_file;

  option_group_definition new_options("Audit Regressor");
  new_options.add(make_option("audit_regressor", out_file)
                      .keep()
                      .necessary()
                      .help("stores feature names and their regressor values. Same dataset must be used for both "
                            "regressor training and this mode."));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  if (out_file.empty()) THROW("audit_regressor argument (output filename) is missing.");

  if (all.numpasses > 1) THROW("audit_regressor can't be used with --passes > 1.");

  all.audit = true;

  auto dat = scoped_calloc_or_throw<audit_regressor_data>();
  dat->all = &all;
  dat->ns_pre = new std::vector<std::string>();  // explicitly invoking std::vector's constructor
  dat->out_file = new io_buf();
  dat->out_file->add_file(VW::io::open_file_writer(out_file));

  VW::LEARNER::learner<audit_regressor_data, example>& ret =
      VW::LEARNER::init_learner(dat, as_singleline(setup_base(options, all)), audit_regressor, audit_regressor, 1,
          all.get_setupfn_name(audit_regressor_setup), true /*audit.learn does not predict or learn.
                                                               nothing to be gained by calling
                                                               predict() before learn()*/
      );
  ret.set_end_examples(end_examples);
  ret.set_finish_example(finish_example);
  ret.set_finish(finish);
  ret.set_init_driver(init_driver);

  return VW::LEARNER::make_base<audit_regressor_data>(ret);
}
