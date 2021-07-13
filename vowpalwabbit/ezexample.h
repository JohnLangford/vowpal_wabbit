// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once
#include <cstdio>
#include "parser.h"
#include "vw.h"
#include "interactions.h"

VW_DEPRECATED(
    "fid is deprecated and will be removed in 9.0. See issue:"
    "https://github.com/VowpalWabbit/vowpal_wabbit/issues/3091")
typedef uint32_t fid;

struct VW_DEPRECATED(
    "vw_namespace is deprecated and will be removed in 9.0. See issue:"
    "https://github.com/VowpalWabbit/vowpal_wabbit/issues/3091") vw_namespace
{
  char namespace_letter;

public:
  vw_namespace(const char c) : namespace_letter(c) {}
};

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_DEPRECATED_USAGE
class VW_DEPRECATED(
    "ezexample is deprecated and will be removed in 9.0. See issue:"
    "https://github.com/VowpalWabbit/vowpal_wabbit/issues/3091") ezexample
{
private:
  vw* vw_ref;
  vw* vw_par_ref;  // an extra parser if we're multithreaded
  bool is_multiline;

  char str[2];
  example* ec;
  bool we_create_ec;
  std::vector<fid> past_seeds;
  fid current_seed;
  float quadratic_features_sqr;
  char current_ns;
  bool ns_exists[256];
  bool example_changed_since_prediction;

  v_array<example*> example_copies;

  ezexample(const ezexample& ex) = delete;
  ezexample& operator=(const ezexample& ex) = delete;

  example* get_new_example()
  {
    example* new_ec = VW::new_unused_example(*vw_par_ref);
    vw_par_ref->example_parser->lbl_parser.default_label(&new_ec->l);
    new_ec->tag.clear();
    new_ec->feature_space.clear();
    new_ec->ft_offset = 0;
    new_ec->num_features = 0;
    new_ec->partial_prediction = 0.;
    new_ec->updated_prediction = 0.;
    new_ec->passthrough = nullptr;
    new_ec->loss = 0.;
    new_ec->reset_total_sum_feat_sq();
    new_ec->confidence = 0.;
    return new_ec;
  }

  void setup_new_ezexample(vw* this_vw, bool multiline, vw* this_vw_parser)
  {
    vw_ref = this_vw;
    vw_par_ref = (this_vw_parser == nullptr) ? this_vw : this_vw_parser;
    is_multiline = multiline;

    str[0] = 0;
    str[1] = 0;
    current_seed = 0;
    current_ns = 0;

    for (bool& ns_exist : ns_exists) ns_exist = false;

    example_changed_since_prediction = true;
  }

  void setup_for_predict()
  {
    static example* empty_example = is_multiline ? VW::read_example(*vw_par_ref, (char*)"") : nullptr;
    if (example_changed_since_prediction)
    {
      mini_setup_example();
      vw_ref->learn(*ec);
      if (is_multiline) vw_ref->learn(*empty_example);
      example_changed_since_prediction = false;
    }
  }

public:
  // REAL FUNCTIONALITY
  // create a new ezexample by asking the vw parser for an example
  ezexample(vw* this_vw, bool multiline = false, vw* this_vw_parser = nullptr)
  {
    setup_new_ezexample(this_vw, multiline, this_vw_parser);
    ec = get_new_example();
    we_create_ec = true;

    if (vw_ref->add_constant) VW::add_constant_feature(*vw_ref, ec);
  }

  // create a new ezexample by wrapping around an already existing example
  // we do NOT copy your data, therefore, WARNING:
  //   do NOT touch the underlying example unless you really know what you're done)
  ezexample(vw* this_vw, example* this_ec, bool multiline = false, vw* this_vw_parser = nullptr)
  {
    setup_new_ezexample(this_vw, multiline, this_vw_parser);

    ec = this_ec;
    we_create_ec = false;

    for (auto ns : ec->indices) ns_exists[ns] = true;
    if (current_ns != 0)
    {
      str[0] = current_ns;
      current_seed = static_cast<fid>(VW::hash_space(*vw_ref, str));
    }
  }

  ~ezexample()  // calls finish_example *only* if we created our own example!
  {
    if (VW::is_ring_example(*vw_par_ref, ec)) VW::finish_example(*vw_par_ref, *ec);
    for (auto ecc : example_copies)
      if (VW::is_ring_example(*vw_par_ref, ec)) VW::finish_example(*vw_par_ref, *ecc);
    example_copies.clear();
    free(example_copies.begin());
  }

  bool ensure_ns_exists(char c)  // returns TRUE iff we should ignore it :)
  {
    if (vw_ref->ignore_some && vw_ref->ignore[static_cast<int>(c)]) return true;
    if (ns_exists[static_cast<int>(c)]) return false;
    std::string s(1, c);
    fid hash = static_cast<fid>(VW::hash_space(*vw_ref, s));
    ec->feature_space.get_or_create(c, hash);
    ns_exists[static_cast<int>(c)] = true;
    return false;
  }

  void addns(char c)
  {
    if (ensure_ns_exists(c)) return;
    past_seeds.push_back(current_seed);
    current_ns = c;
    str[0] = c;
    current_seed = static_cast<fid>(VW::hash_space(*vw_ref, str));
    ec->feature_space[current_seed].clear();
  }

  void remns()
  {
    if (ec->feature_space.empty())
    {
      current_seed = 0;
      current_ns = 0;
    }
    else
    {
      if (ns_exists[static_cast<int>(current_ns)])
      {
        ec->reset_total_sum_feat_sq();
        ec->feature_space[current_seed].clear();
        ec->num_features -= ec->feature_space[current_seed].size();

        ns_exists[static_cast<int>(current_ns)] = false;
      }

      ec->feature_space.remove(current_seed);
      current_seed = past_seeds.back();
      past_seeds.pop_back();
      example_changed_since_prediction = true;
    }
  }

  inline fid addf(char to_ns, fid fint, float v)
  {
    if (to_ns == 0) return 0;
    if (ensure_ns_exists(to_ns)) return 0;

    ec->feature_space[current_seed].push_back(v, fint << vw_ref->weights.stride_shift());
    ec->reset_total_sum_feat_sq();
    ec->num_features++;
    example_changed_since_prediction = true;
    return fint;
  }

  inline fid addf(fid fint, float v) { return addf(current_ns, fint, v); }

  // copy an entire namespace from this other example, you can even give it a new namespace name if you want!
  void add_other_example_ns(example& other, char other_ns, char to_ns)
  {
    if (ensure_ns_exists(to_ns)) return;
    features& fs = other.feature_space.get(current_seed);
    for (size_t i = 0; i < fs.size(); i++) ec->feature_space[current_seed].push_back(fs.values[i], fs.indicies[i]);
    ec->reset_total_sum_feat_sq();
    ec->num_features += fs.size();
    example_changed_since_prediction = true;
  }
  void add_other_example_ns(example& other, char ns)  // default to_ns to other_ns
  {
    add_other_example_ns(other, ns, ns);
  }

  void add_other_example_ns(ezexample& other, char other_ns, char to_ns)
  {
    add_other_example_ns(*other.ec, other_ns, to_ns);
  }
  void add_other_example_ns(ezexample& other, char ns) { add_other_example_ns(*other.ec, ns); }

  inline ezexample& set_label(std::string label)
  {
    VW::parse_example_label(*vw_par_ref, *ec, label);
    example_changed_since_prediction = true;
    return *this;
  }

  void mini_setup_example()
  {
    ec->partial_prediction = 0.;
    ec->weight = vw_par_ref->example_parser->lbl_parser.get_weight(&ec->l, ec->_reduction_features);

    ec->reset_total_sum_feat_sq();
    ec->num_features = 0;
    for (const features& fs : *ec) { ec->num_features += fs.size(); }
    ec->interactions = &vw_ref->interactions;
  }

  size_t get_num_features() { return ec->get_num_features(); }

  example* get()
  {
    if (example_changed_since_prediction) mini_setup_example();
    return ec;
  }

  float predict()
  {
    setup_for_predict();
    return ec->pred.scalar;
  }

  float predict_partial()
  {
    setup_for_predict();
    return ec->partial_prediction;
  }

  void train()  // if multiline, add to stack; otherwise, actually train
  {
    if (example_changed_since_prediction)
    {
      mini_setup_example();
      example_changed_since_prediction = false;
    }

    if (!is_multiline) { vw_ref->learn(*ec); }
    else  // is multiline
    {     // we need to make a copy
      example* copy = get_new_example();
      VW::copy_example_data_with_label(copy, ec);
      vw_ref->learn(*copy);
      example_copies.push_back(copy);
    }
  }

  void clear_features()
  {
    for (size_t i = 0; i < 256; i++)
    {
      if (current_ns == 0) break;
      remns();
    }
  }

  void finish()
  {
    static example* empty_example = is_multiline ? VW::read_example(*vw_par_ref, (char*)"") : nullptr;
    if (is_multiline)
    {
      vw_ref->learn(*empty_example);
      for (auto ecc : example_copies) VW::finish_example(*vw_par_ref, *ecc);
      example_copies.clear();
    }
  }

  // HELPER FUNCTIONALITY

  inline fid hash(std::string fstr) { return static_cast<fid>(VW::hash_feature(*vw_ref, fstr, current_seed)); }
  inline fid hash(char* fstr) { return static_cast<fid>(VW::hash_feature_cstr(*vw_ref, fstr, current_seed)); }
  inline fid hash(char c, std::string fstr)
  {
    str[0] = c;
    return static_cast<fid>(VW::hash_feature(*vw_ref, fstr, VW::hash_space(*vw_ref, str)));
  }
  inline fid hash(char c, char* fstr)
  {
    str[0] = c;
    return static_cast<fid>(VW::hash_feature_cstr(*vw_ref, fstr, VW::hash_space(*vw_ref, str)));
  }

  inline fid addf(fid fint) { return addf(fint, 1.0); }
  inline fid addf(std::string fstr, float val) { return addf(hash(fstr), val); }
  inline fid addf(std::string fstr) { return addf(hash(fstr), 1.0); }

  inline fid addf(char ns, fid fint) { return addf(ns, fint, 1.0); }
  inline fid addf(char ns, std::string fstr, float val) { return addf(ns, hash(ns, fstr), val); }
  inline fid addf(char ns, std::string fstr) { return addf(ns, hash(ns, fstr), 1.0); }

  inline ezexample& operator()(const vw_namespace& n)
  {
    addns(n.namespace_letter);
    return *this;
  }

  inline ezexample& operator()(fid fint)
  {
    addf(fint, 1.0);
    return *this;
  }
  inline ezexample& operator()(std::string fstr)
  {
    addf(fstr, 1.0);
    return *this;
  }
  inline ezexample& operator()(const char* fstr)
  {
    addf(fstr, 1.0);
    return *this;
  }
  inline ezexample& operator()(fid fint, float val)
  {
    addf(fint, val);
    return *this;
  }
  inline ezexample& operator()(std::string fstr, float val)
  {
    addf(fstr, val);
    return *this;
  }
  inline ezexample& operator()(const char* fstr, float val)
  {
    addf(fstr, val);
    return *this;
  }

  inline ezexample& operator()(char ns, fid fint)
  {
    addf(ns, fint, 1.0);
    return *this;
  }
  inline ezexample& operator()(char ns, std::string fstr)
  {
    addf(ns, fstr, 1.0);
    return *this;
  }
  inline ezexample& operator()(char ns, const char* fstr)
  {
    addf(ns, fstr, 1.0);
    return *this;
  }
  inline ezexample& operator()(char ns, fid fint, float val)
  {
    addf(ns, fint, val);
    return *this;
  }
  inline ezexample& operator()(char ns, std::string fstr, float val)
  {
    addf(ns, fstr, val);
    return *this;
  }
  inline ezexample& operator()(char ns, const char* fstr, float val)
  {
    addf(ns, fstr, val);
    return *this;
  }

  inline ezexample& operator()(example& other, char other_ns, char to_ns)
  {
    add_other_example_ns(other, other_ns, to_ns);
    return *this;
  }
  inline ezexample& operator()(example& other, char ns)
  {
    add_other_example_ns(other, ns);
    return *this;
  }
  inline ezexample& operator()(ezexample& other, char other_ns, char to_ns)
  {
    add_other_example_ns(other, other_ns, to_ns);
    return *this;
  }
  inline ezexample& operator()(ezexample& other, char ns)
  {
    add_other_example_ns(other, ns);
    return *this;
  }

  inline ezexample& operator--()
  {
    remns();
    return *this;
  }

  inline float operator()() { return predict(); }
};
VW_WARNING_STATE_POP