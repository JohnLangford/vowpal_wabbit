// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "vw.h"

#include "multiclass.h"
#include "cost_sensitive.h"
#include "cb.h"
#include "search.h"
#include "search_hooktask.h"
#include "parse_example.h"
#include "gd.h"
#include "options_boost_po.h"
#include "options_serializer_boost_po.h"
#include "future_compat.h"
#include "slates_label.h"
#include "simple_label_parser.h"
#include "shared_data.h"
#include "reduction_stack.h"

// see http://www.boost.org/doc/libs/1_56_0/doc/html/bbv2/installation.html
#define BOOST_PYTHON_USE_GCC_SYMBOL_VISIBILITY 1
#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/utility.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// Brings VW_DLL_PUBLIC to help control exports
#define VWDLL_EXPORTS
#include "../vowpalwabbit/vwdll.h"

namespace py = boost::python;

class py_log_wrapper;

typedef boost::shared_ptr<vw> vw_ptr;
typedef boost::shared_ptr<example> example_ptr;
typedef boost::shared_ptr<Search::search> search_ptr;
typedef boost::shared_ptr<Search::predictor> predictor_ptr;
typedef boost::shared_ptr<py_log_wrapper> py_log_wrapper_ptr;

const size_t lDEFAULT = 0;
const size_t lBINARY = 1;
const size_t lMULTICLASS = 2;
const size_t lCOST_SENSITIVE = 3;
const size_t lCONTEXTUAL_BANDIT = 4;
const size_t lMAX = 5;
const size_t lCONDITIONAL_CONTEXTUAL_BANDIT = 6;
const size_t lSLATES = 7;
const size_t lCONTINUOUS = 8;

const size_t pSCALAR = 0;
const size_t pSCALARS = 1;
const size_t pACTION_SCORES = 2;
const size_t pACTION_PROBS = 3;
const size_t pMULTICLASS = 4;
const size_t pMULTILABELS = 5;
const size_t pPROB = 6;
const size_t pMULTICLASSPROBS = 7;
const size_t pDECISION_SCORES = 8;
const size_t pACTION_PDF_VALUE = 9;
const size_t pPDF = 10;
const size_t pACTIVE_MULTICLASS = 11;

void dont_delete_me(void* arg) {}

class OptionManager
{
  std::map<std::string, std::vector<VW::config::option_group_definition>> m_option_group_dic;
  // see pyvw.py class VWOption
  py::object m_py_opt_class;
  VW::config::options_i& m_opt;
  std::vector<std::string>& m_enabled_reductions;
  std::string default_group_name;

public:
  OptionManager(VW::config::options_i& options, std::vector<std::string>& enabled_reductions, py::object py_class)
      : m_opt(options)
      , m_enabled_reductions(enabled_reductions)
      , m_option_group_dic(options.get_collection_of_options())
      , m_py_opt_class(py_class)
  {
    default_group_name = static_cast<VW::config::options_boost_po*>(&options)->m_default_tint;
  }

  py::object* value_to_pyobject(VW::config::typed_option<bool>& opt)
  {
    if (m_opt.was_supplied(opt.m_name))
    {
      if (opt.default_value_supplied())
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.value(), true, opt.default_value(), true));
      else
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.value(), true, false, true));
    }
    else
    {
      if (opt.default_value_supplied())
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.default_value(), false, opt.default_value(), true));
      else
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, false, false, false, true));
    }
  }

  template <typename T>
  py::object* value_to_pyobject(VW::config::typed_option<T>& opt)
  {
    T not_supplied{};

    if (m_opt.was_supplied(opt.m_name))
    {
      if (opt.default_value_supplied())
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.value(), true, opt.default_value(), true));
      else
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.value(), true, not_supplied, false));
    }
    else
    {
      if (opt.default_value_supplied())
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, opt.default_value(), false, opt.default_value(), true));
      else
        return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
            opt.m_allow_override, py::object(), false, not_supplied, false));
    }
  }

  template <typename T>
  py::object* value_to_pyobject(VW::config::typed_option<std::vector<T>>& opt)
  {
    py::list values;

    if (m_opt.was_supplied(opt.m_name))
    {
      auto vec = opt.value();
      if (vec.size() > 0)
      {
        for (auto const& opt : vec) { values.append(py::object(opt)); }
      }
    }

    return new py::object(m_py_opt_class(opt.m_name, opt.m_help, opt.m_short_name, opt.m_keep, opt.m_necessary,
        opt.m_allow_override, values, m_opt.was_supplied(opt.m_name), py::list(), opt.default_value_supplied()));
  }

  template <typename T>
  py::object* transform_if_t(VW::config::base_option& base_option)
  {
    if (base_option.m_type_hash == typeid(T).hash_code())
    {
      auto typed = dynamic_cast<VW::config::typed_option<T>&>(base_option);
      return value_to_pyobject(typed);
    }

    return nullptr;
  }

  template <typename TTypes>
  py::object base_option_to_pyobject(VW::config::base_option& options)
  {
    py::object* temp = transform_if_t<typename TTypes::head>(options);
    if (temp != nullptr)
    {
      auto repack = py::object(*temp);
      delete temp;
      return repack;
    }

    return base_option_to_pyobject<typename TTypes::tail>(options);
  }

  py::object get_vw_option_pyobjects(bool enabled_only)
  {
    py::dict dres;
    auto it = m_option_group_dic.begin();

    while (it != m_option_group_dic.end())
    {
      auto reduction_enabled =
          std::find(m_enabled_reductions.begin(), m_enabled_reductions.end(), it->first) != m_enabled_reductions.end();

      if (((it->first).compare(default_group_name) != 0) && enabled_only && !reduction_enabled)
      {
        it++;
        continue;
      }

      py::list option_groups;

      for (auto options_group : it->second)
      {
        py::list options;
        for (auto opt : options_group.m_options)
        {
          auto temp = base_option_to_pyobject<VW::config::supported_options_types>(*opt.get());
          options.append(temp);
        }

        option_groups.append(py::make_tuple(options_group.m_name, options));
      }

      dres[it->first] = option_groups;

      it++;
    }
    return dres;
  }
};

// specialization needed to compile, this should never be reached since we always use
// VW::config::supported_options_types
template <>
py::object OptionManager::base_option_to_pyobject<VW::config::typelist<>>(VW::config::base_option& options)
{
  return py::object();
}

class py_log_wrapper
{
public:
  py::object py_log;
  py_log_wrapper(py::object py_log) : py_log(py_log) {}

  static void trace_listener_py(void* wrapper, const std::string& message)
  {
    try
    {
      auto inst = static_cast<py_log_wrapper*>(wrapper);
      inst->py_log.attr("log")(message);
    }
    catch (...)
    {
      // TODO: Properly translate and return Python exception. #2169
      PyErr_Print();
      PyErr_Clear();
      std::cerr << "error using python logging. ignoring." << std::endl;
    }
  }
};

vw_ptr my_initialize_with_log(std::string args, py_log_wrapper_ptr py_log)
{
  if (args.find_first_of("--no_stdin") == std::string::npos) args += " --no_stdin";

  trace_message_t trace_listener = nullptr;
  void* trace_context = nullptr;

  if (py_log)
  {
    trace_listener = (py_log_wrapper::trace_listener_py);
    trace_context = py_log.get();
  }

  vw* foo = VW::initialize(args, nullptr, false, trace_listener, trace_context);
  // return boost::shared_ptr<vw>(foo, [](vw *all){VW::finish(*all);});
  return boost::shared_ptr<vw>(foo);
}

vw_ptr my_initialize(std::string args) { return my_initialize_with_log(args, nullptr); }

void my_run_parser(vw_ptr all)
{
  VW::start_parser(*all);
  VW::LEARNER::generic_driver(*all);
  VW::end_parser(*all);
}

py::dict get_learner_metrics(vw_ptr all)
{
  py::dict dictionary;

  if (all->options->was_supplied("extra_metrics"))
  {
    VW::metric_sink metrics;
    all->l->persist_metrics(metrics);

    for (const auto& m : metrics.int_metrics_list) { dictionary[m.first] = m.second; }
    for (const auto& m : metrics.float_metrics_list) { dictionary[m.first] = m.second; }
  }

  return dictionary;
}

void my_finish(vw_ptr all)
{
  VW::finish(*all, false);  // don't delete all because python will do that for us!
}

void my_save(vw_ptr all, std::string name) { VW::save_predictor(*all, name); }

search_ptr get_search_ptr(vw_ptr all)
{
  return boost::shared_ptr<Search::search>((Search::search*)(all->searchstr), dont_delete_me);
}

py::object get_options(vw_ptr all, py::object py_class, bool enabled_only)
{
  if (all->learner_builder)
  {
    auto opt_manager = OptionManager(*all->options, all->learner_builder->enabled_reductions, py_class);
    return opt_manager.get_vw_option_pyobjects(enabled_only);
  }
  else
  {
    return py::object();
  }
}

void my_audit_example(vw_ptr all, example_ptr ec) { GD::print_audit_features(*all, *ec); }

const char* get_model_id(vw_ptr all) { return all->id.c_str(); }

std::string get_arguments(vw_ptr all)
{
  VW::config::options_serializer_boost_po serializer;
  for (auto const& option : all->options->get_all_options())
  {
    if (all->options->was_supplied(option->m_name)) serializer.add(*option);
  }

  return serializer.str();
}

py::list get_enabled_reductions(vw_ptr all)
{
  py::list enabled_reductions;

  if (all->learner_builder)
  {
    for (auto ex : all->learner_builder->enabled_reductions) { enabled_reductions.append(ex); }
  }

  return enabled_reductions;
}

predictor_ptr get_predictor(search_ptr sch, ptag my_tag)
{
  Search::predictor* P = new Search::predictor(*sch, my_tag);
  return boost::shared_ptr<Search::predictor>(P);
}

label_parser* get_label_parser(vw* all, size_t labelType)
{
  switch (labelType)
  {
    case lDEFAULT:
      return all ? &all->example_parser->lbl_parser : NULL;
    case lBINARY:
      return &simple_label_parser;
    case lMULTICLASS:
      return &MULTICLASS::mc_label;
    case lCOST_SENSITIVE:
      return &COST_SENSITIVE::cs_label;
    case lCONTEXTUAL_BANDIT:
      return &CB::cb_label;
    case lCONDITIONAL_CONTEXTUAL_BANDIT:
      return &CCB::ccb_label_parser;
    case lSLATES:
      return &VW::slates::slates_label_parser;
    case lCONTINUOUS:
      return &VW::cb_continuous::the_label_parser;
    default:
      THROW("get_label_parser called on invalid label type");
  }
}

size_t my_get_label_type(vw* all)
{
  label_parser* lp = &all->example_parser->lbl_parser;
  if (lp->parse_label == simple_label_parser.parse_label) { return lBINARY; }
  else if (lp->parse_label == MULTICLASS::mc_label.parse_label)
  {
    return lMULTICLASS;
  }
  else if (lp->parse_label == COST_SENSITIVE::cs_label.parse_label)
  {
    return lCOST_SENSITIVE;
  }
  else if (lp->parse_label == CB::cb_label.parse_label)
  {
    return lCONTEXTUAL_BANDIT;
  }
  else if (lp->parse_label == CCB::ccb_label_parser.parse_label)
  {
    return lCONDITIONAL_CONTEXTUAL_BANDIT;
  }
  else if (lp->parse_label == VW::slates::slates_label_parser.parse_label)
  {
    return lSLATES;
  }
  else if (lp->parse_label == VW::cb_continuous::the_label_parser.parse_label)
  {
    return lCONTINUOUS;
  }
  else
  {
    THROW("unsupported label parser used");
  }
}

size_t my_get_prediction_type(vw_ptr all)
{
  switch (all->l->pred_type)
  {
    case prediction_type_t::scalar:
      return pSCALAR;
    case prediction_type_t::scalars:
      return pSCALARS;
    case prediction_type_t::action_scores:
      return pACTION_SCORES;
    case prediction_type_t::action_probs:
      return pACTION_PROBS;
    case prediction_type_t::multiclass:
      return pMULTICLASS;
    case prediction_type_t::multilabels:
      return pMULTILABELS;
    case prediction_type_t::prob:
      return pPROB;
    case prediction_type_t::multiclassprobs:
      return pMULTICLASSPROBS;
    case prediction_type_t::decision_probs:
      return pDECISION_SCORES;
    case prediction_type_t::action_pdf_value:
      return pACTION_PDF_VALUE;
    case prediction_type_t::pdf:
      return pPDF;
    case prediction_type_t::active_multiclass:
      return pACTIVE_MULTICLASS;
    default:
      THROW("unsupported prediction type used");
  }
}

void my_delete_example(void* voidec)
{
  example* ec = (example*)voidec;
  size_t labelType = ec->example_counter;
  label_parser* lp = get_label_parser(NULL, labelType);
  VW::dealloc_examples(ec, 1);
}

example* my_empty_example0(vw_ptr vw, size_t labelType)
{
  label_parser* lp = get_label_parser(&*vw, labelType);
  example* ec = VW::alloc_examples(1);
  lp->default_label(&ec->l);
  ec->interactions = &vw->interactions;
  if (labelType == lCOST_SENSITIVE)
  {
    COST_SENSITIVE::wclass zero = {0., 1, 0., 0.};
    ec->l.cs.costs.push_back(zero);
  }
  ec->example_counter = labelType;
  return ec;
}

example_ptr my_empty_example(vw_ptr vw, size_t labelType)
{
  example* ec = my_empty_example0(vw, labelType);
  return boost::shared_ptr<example>(ec, my_delete_example);
}

example_ptr my_read_example(vw_ptr all, size_t labelType, char* str)
{
  example* ec = my_empty_example0(all, labelType);
  VW::read_line(*all, ec, str);
  VW::setup_example(*all, ec);
  ec->example_counter = labelType;
  return boost::shared_ptr<example>(ec, my_delete_example);
}

example_ptr my_existing_example(vw_ptr all, size_t labelType, example_ptr existing_example)
{
  existing_example->example_counter = labelType;
  return existing_example;
  // return boost::shared_ptr<example>(existing_example);
}

multi_ex unwrap_example_list(py::list& ec)
{
  multi_ex ex_coll;
  for (ssize_t i = 0; i < py::len(ec); i++) { ex_coll.push_back(py::extract<example_ptr>(ec[i])().get()); }
  return ex_coll;
}

void my_finish_example(vw_ptr all, example_ptr ec) { as_singleline(all->l)->finish_example(*all, *ec); }

void my_finish_multi_ex(vw_ptr& all, py::list& ec)
{
  auto ex_col = unwrap_example_list(ec);
  as_multiline(all->l)->finish_example(*all, ex_col);
}

void my_learn(vw_ptr all, example_ptr ec)
{
  if (ec->test_only) { as_singleline(all->l)->predict(*ec); }
  else
  {
    all->learn(*ec.get());
  }
}

float my_predict(vw_ptr all, example_ptr ec)
{
  as_singleline(all->l)->predict(*ec);
  return ec->partial_prediction;
}

bool my_is_multiline(vw_ptr all) { return all->l->is_multiline; }

template <bool learn>
void predict_or_learn(vw_ptr& all, py::list& ec)
{
  multi_ex ex_coll = unwrap_example_list(ec);
  if (learn)
    all->learn(ex_coll);
  else
    as_multiline(all->l)->predict(ex_coll);
}

py::list my_parse(vw_ptr& all, char* str)
{
  v_array<example*> examples;
  examples.push_back(&VW::get_unused_example(all.get()));
  all->example_parser->text_reader(all.get(), str, strlen(str), examples);

  py::list example_collection;
  for (auto* ex : examples)
  {
    VW::setup_example(*all, ex);
    // Examples created from parsed text should not be deleted normally. Instead they need to be
    // returned to the pool using finish_example.
    example_collection.append(boost::shared_ptr<example>(ex, dont_delete_me));
  }
  return example_collection;
}

void my_learn_multi_ex(vw_ptr& all, py::list& ec) { predict_or_learn<true>(all, ec); }

void my_predict_multi_ex(vw_ptr& all, py::list& ec) { predict_or_learn<false>(all, ec); }

std::string varray_char_to_string(v_array<char>& a)
{
  std::string ret = "";
  for (auto c : a) ret += c;
  return ret;
}

std::string my_get_tag(example_ptr ec) { return varray_char_to_string(ec->tag); }

uint32_t ex_num_namespaces(example_ptr ec) { return (uint32_t)ec->indices.size(); }

unsigned char ex_namespace(example_ptr ec, uint32_t ns) { return ec->indices[ns]; }

uint32_t ex_num_features(example_ptr ec, unsigned char ns) { return (uint32_t)ec->feature_space[ns].size(); }

uint32_t ex_feature(example_ptr ec, unsigned char ns, uint32_t i)
{
  return (uint32_t)ec->feature_space[ns].indicies[i];
}

float ex_feature_weight(example_ptr ec, unsigned char ns, uint32_t i) { return ec->feature_space[ns].values[i]; }

float ex_sum_feat_sq(example_ptr ec, unsigned char ns) { return ec->feature_space[ns].sum_feat_sq; }

void ex_push_feature(example_ptr ec, unsigned char ns, uint32_t fid, float v)
{  // warning: assumes namespace exists!
  ec->feature_space[ns].push_back(v, fid);
  ec->num_features++;
  ec->reset_total_sum_feat_sq();
}

void ex_push_feature_list(example_ptr ec, vw_ptr vw, unsigned char ns, py::list& a)
{  // warning: assumes namespace exists!
  char ns_str[2] = {(char)ns, 0};
  uint64_t ns_hash = VW::hash_space(*vw, ns_str);
  size_t count = 0;
  for (ssize_t i = 0; i < len(a); i++)
  {
    feature f = {1., 0};
    py::object ai = a[i];
    py::extract<py::tuple> get_tup(ai);
    if (get_tup.check())
    {
      py::tuple fv = get_tup();
      if (len(fv) != 2)
      {
        std::cerr << "warning: malformed feature in list" << std::endl;
        continue;
      }  // TODO str(ai)
      py::extract<float> get_val(fv[1]);
      if (get_val.check())
        f.x = get_val();
      else
      {
        std::cerr << "warning: malformed feature in list" << std::endl;
        continue;
      }
      ai = fv[0];
    }

    if (f.x != 0.)
    {
      bool got = false;
      py::extract<std::string> get_str(ai);
      if (get_str.check())
      {
        f.weight_index = VW::hash_feature(*vw, get_str(), ns_hash);
        got = true;
      }
      else
      {
        py::extract<uint32_t> get_int(ai);
        if (get_int.check())
        {
          f.weight_index = get_int();
          got = true;
        }
        else
        {
          std::cerr << "warning: malformed feature in list" << std::endl;
          continue;
        }
      }
      if (got)
      {
        ec->feature_space[ns].push_back(f.x, f.weight_index);
        count++;
      }
    }
  }
  ec->num_features += count;
  ec->reset_total_sum_feat_sq();
}

void ex_push_namespace(example_ptr ec, unsigned char ns) { ec->indices.push_back(ns); }

void ex_ensure_namespace_exists(example_ptr ec, unsigned char ns)
{
  for (auto nss : ec->indices)
    if (ns == nss) return;
  ex_push_namespace(ec, ns);
}

void ex_push_dictionary(example_ptr ec, vw_ptr vw, py::dict& dict)
{
  const py::object objectKeys = py::object(py::handle<>(PyObject_GetIter(dict.keys().ptr())));
  const py::object objectVals = py::object(py::handle<>(PyObject_GetIter(dict.values().ptr())));
  unsigned long ulCount = boost::python::extract<unsigned long>(dict.attr("__len__")());
  for (size_t u = 0; u < ulCount; ++u)
  {
    py::object objectKey = py::object(py::handle<>(PyIter_Next(objectKeys.ptr())));
    py::object objectVal = py::object(py::handle<>(PyIter_Next(objectVals.ptr())));

    char chCheckKey = objectKey.ptr()->ob_type->tp_name[0];
    if (chCheckKey != 's') continue;
    chCheckKey = objectVal.ptr()->ob_type->tp_name[0];
    if (chCheckKey != 'l') continue;

    py::extract<std::string> ns_e(objectKey);
    if (ns_e().length() < 1) continue;
    py::extract<py::list> list_e(objectVal);
    py::list list = list_e();
    char ns = ns_e()[0];
    ex_ensure_namespace_exists(ec, ns);
    ex_push_feature_list(ec, vw, ns, list);
  }
}

bool ex_pop_feature(example_ptr ec, unsigned char ns)
{
  if (ec->feature_space[ns].size() == 0) return false;
  float val = ec->feature_space[ns].values.back();
  ec->feature_space[ns].values.pop_back();
  if (ec->feature_space[ns].indicies.size() > 0) ec->feature_space[ns].indicies.pop_back();
  if (ec->feature_space[ns].space_names.size() > 0) ec->feature_space[ns].space_names.pop_back();
  ec->num_features--;
  ec->feature_space[ns].sum_feat_sq -= val * val;
  ec->reset_total_sum_feat_sq();
  return true;
}

void ex_erase_namespace(example_ptr ec, unsigned char ns)
{
  ec->num_features -= ec->feature_space[ns].size();
  ec->reset_total_sum_feat_sq();
  ec->feature_space[ns].sum_feat_sq = 0.;
  ec->feature_space[ns].clear();
}

bool ex_pop_namespace(example_ptr ec)
{
  if (ec->indices.size() == 0) return false;
  unsigned char ns = ec->indices.back();
  ec->indices.pop_back();
  ex_erase_namespace(ec, ns);
  return true;
}

void my_setup_example(vw_ptr vw, example_ptr ec) { VW::setup_example(*vw, ec.get()); }

void unsetup_example(vw_ptr vwP, example_ptr ae)
{
  vw& all = *vwP;
  ae->partial_prediction = 0.;
  ae->num_features = 0;
  ae->reset_total_sum_feat_sq();
  ae->loss = 0.;

  if (all.ignore_some) { THROW("error: cannot unsetup example when some namespaces are ignored!"); }

  if (all.skip_gram_transformer != nullptr && !all.skip_gram_transformer->get_initial_ngram_definitions().empty())
  { THROW("error: cannot unsetup example when ngrams are in use!"); }

  if (all.add_constant)
  {
    ae->feature_space[constant_namespace].clear();
    int hit_constant = -1;
    size_t N = ae->indices.size();
    for (size_t i = 0; i < N; i++)
    {
      int j = (int)(N - 1 - i);
      if (ae->indices[j] == constant_namespace)
      {
        if (hit_constant >= 0) { THROW("error: hit constant namespace twice!"); }
        hit_constant = j;
        break;
      }
    }
    if (hit_constant >= 0)
    {
      for (size_t i = hit_constant; i < N - 1; i++) ae->indices[i] = ae->indices[i + 1];
      ae->indices.pop_back();
    }
  }

  uint32_t multiplier = all.wpp << all.weights.stride_shift();
  if (multiplier != 1)  // make room for per-feature information.
    for (auto ns : ae->indices)
      for (auto& idx : ae->feature_space[ns].indicies) idx /= multiplier;
}

void ex_set_label_string(example_ptr ec, vw_ptr vw, std::string label, size_t labelType)
{  // SPEEDUP: if it's already set properly, don't modify
  label_parser& old_lp = vw->example_parser->lbl_parser;
  vw->example_parser->lbl_parser = *get_label_parser(&*vw, labelType);
  VW::parse_example_label(*vw, *ec, label);
  vw->example_parser->lbl_parser = old_lp;
}

float ex_get_simplelabel_label(example_ptr ec) { return ec->l.simple.label; }
float ex_get_simplelabel_weight(example_ptr ec) { return ec->weight; }
float ex_get_simplelabel_initial(example_ptr ec)
{
  return ec->_reduction_features.template get<simple_label_reduction_features>().initial;
}
float ex_get_simplelabel_prediction(example_ptr ec) { return ec->pred.scalar; }
float ex_get_prob(example_ptr ec) { return ec->pred.prob; }

uint32_t ex_get_multiclass_label(example_ptr ec) { return ec->l.multi.label; }
float ex_get_multiclass_weight(example_ptr ec) { return ec->l.multi.weight; }
uint32_t ex_get_multiclass_prediction(example_ptr ec) { return ec->pred.multiclass; }

py::list ex_get_scalars(example_ptr ec)
{
  py::list values;
  const auto& scalars = ec->pred.scalars;

  for (float s : scalars) { values.append(s); }
  return values;
}

py::list ex_get_action_scores(example_ptr ec)
{
  py::list values;
  auto const& scores = ec->pred.a_s;
  std::vector<float> ordered_scores(scores.size());
  for (auto const& action_score : scores) { ordered_scores[action_score.action] = action_score.score; }

  for (auto action_score : ordered_scores) { values.append(action_score); }

  return values;
}

py::list ex_get_decision_scores(example_ptr ec)
{
  py::list values;
  for (auto const& scores : ec->pred.decision_scores)
  {
    py::list inner_list;
    for (auto action_score : scores) { inner_list.append(py::make_tuple(action_score.action, action_score.score)); }

    values.append(inner_list);
  }

  return values;
}

py::tuple ex_get_action_pdf_value(example_ptr ec)
{
  return py::make_tuple(ec->pred.pdf_value.action, ec->pred.pdf_value.pdf_value);
}

py::list ex_get_pdf(example_ptr ec)
{
  py::list values;
  for (auto const& segment : ec->pred.pdf)
  { values.append(py::make_tuple(segment.left, segment.right, segment.pdf_value)); }
  return values;
}

py::tuple ex_get_active_multiclass(example_ptr ec)
{
  py::list values;
  for (auto const& query_needed_class : ec->pred.active_multiclass.more_info_required_for_classes)
  { values.append(query_needed_class); }

  return py::make_tuple(ec->pred.active_multiclass.predicted_class, values);
}

py::list ex_get_multilabel_predictions(example_ptr ec)
{
  py::list values;
  MULTILABEL::labels labels = ec->pred.multilabels;

  for (uint32_t l : labels.label_v) { values.append(l); }
  return values;
}

uint32_t ex_get_costsensitive_prediction(example_ptr ec) { return ec->pred.multiclass; }
uint32_t ex_get_costsensitive_num_costs(example_ptr ec) { return (uint32_t)ec->l.cs.costs.size(); }
float ex_get_costsensitive_cost(example_ptr ec, uint32_t i) { return ec->l.cs.costs[i].x; }
uint32_t ex_get_costsensitive_class(example_ptr ec, uint32_t i) { return ec->l.cs.costs[i].class_index; }
float ex_get_costsensitive_partial_prediction(example_ptr ec, uint32_t i)
{
  return ec->l.cs.costs[i].partial_prediction;
}
float ex_get_costsensitive_wap_value(example_ptr ec, uint32_t i) { return ec->l.cs.costs[i].wap_value; }

uint32_t ex_get_cbandits_prediction(example_ptr ec) { return ec->pred.multiclass; }
uint32_t ex_get_cbandits_num_costs(example_ptr ec) { return (uint32_t)ec->l.cb.costs.size(); }
float ex_get_cbandits_cost(example_ptr ec, uint32_t i) { return ec->l.cb.costs[i].cost; }
uint32_t ex_get_cbandits_class(example_ptr ec, uint32_t i) { return ec->l.cb.costs[i].action; }
float ex_get_cbandits_probability(example_ptr ec, uint32_t i) { return ec->l.cb.costs[i].probability; }
float ex_get_cbandits_partial_prediction(example_ptr ec, uint32_t i) { return ec->l.cb.costs[i].partial_prediction; }

// example_counter is being overriden by lableType!
size_t get_example_counter(example_ptr ec) { return ec->example_counter; }
uint64_t get_ft_offset(example_ptr ec) { return ec->ft_offset; }
size_t get_num_features(example_ptr ec) { return ec->get_num_features(); }
float get_partial_prediction(example_ptr ec) { return ec->partial_prediction; }
float get_updated_prediction(example_ptr ec) { return ec->updated_prediction; }
float get_loss(example_ptr ec) { return ec->loss; }
float get_total_sum_feat_sq(example_ptr ec) { return ec->get_total_sum_feat_sq(); }

double get_sum_loss(vw_ptr vw) { return vw->sd->sum_loss; }
double get_weighted_examples(vw_ptr vw) { return vw->sd->weighted_examples(); }

bool search_should_output(search_ptr sch) { return sch->output().good(); }
void search_output(search_ptr sch, std::string s) { sch->output() << s; }

/*
uint32_t search_predict_one_all(search_ptr sch, example_ptr ec, uint32_t one_ystar) {
  return sch->predict(ec.get(), one_ystar, NULL);
}

uint32_t search_predict_one_some(search_ptr sch, example_ptr ec, uint32_t one_ystar, std::vector<uint32_t>& yallowed) {
  v_array<uint32_t> yallowed_va;
  yallowed_va.begin       = yallowed.data();
  yallowed_va.end         = yallowed_va.begin + yallowed.size();
  yallowed_va.end_array   = yallowed_va.end;
  yallowed_va.erase_count = 0;
  return sch->predict(ec.get(), one_ystar, &yallowed_va);
}

uint32_t search_predict_many_all(search_ptr sch, example_ptr ec, std::vector<uint32_t>& ystar) {
  v_array<uint32_t> ystar_va;
  ystar_va.begin       = ystar.data();
  ystar_va.end         = ystar_va.begin + ystar.size();
  ystar_va.end_array   = ystar_va.end;
  ystar_va.erase_count = 0;
  return sch->predict(ec.get(), &ystar_va, NULL);
}

uint32_t search_predict_many_some(search_ptr sch, example_ptr ec, std::vector<uint32_t>& ystar, std::vector<uint32_t>&
yallowed) { v_array<uint32_t> ystar_va; ystar_va.begin       = ystar.data(); ystar_va.end         = ystar_va.begin +
ystar.size(); ystar_va.end_array   = ystar_va.end; ystar_va.erase_count = 0; v_array<uint32_t> yallowed_va;
  yallowed_va.begin       = yallowed.data();
  yallowed_va.end         = yallowed_va.begin + yallowed.size();
  yallowed_va.end_array   = yallowed_va.end;
  yallowed_va.erase_count = 0;
  return sch->predict(ec.get(), &ystar_va, &yallowed_va);
}
*/

void verify_search_set_properly(search_ptr sch)
{
  if (sch->task_name == nullptr) { THROW("set_structured_predict_hook: search task not initialized properly"); }

  if (std::strcmp(sch->task_name, "hook") != 0)
  { THROW("set_structured_predict_hook: trying to set hook when search task is not 'hook'!"); }
}

uint32_t search_get_num_actions(search_ptr sch)
{
  verify_search_set_properly(sch);
  HookTask::task_data* d = sch->get_task_data<HookTask::task_data>();
  return (uint32_t)d->num_actions;
}

void search_run_fn(Search::search& sch)
{
  try
  {
    HookTask::task_data* d = sch.get_task_data<HookTask::task_data>();
    py::object run = *(py::object*)d->run_object;
    run.attr("__call__")();
  }
  catch (...)
  {
    // TODO: Properly translate and return Python exception. #2169
    PyErr_Print();
    PyErr_Clear();
    THROW("Exception in 'search_run_fn'");
  }
}

void search_setup_fn(Search::search& sch)
{
  try
  {
    HookTask::task_data* d = sch.get_task_data<HookTask::task_data>();
    py::object run = *(py::object*)d->setup_object;
    run.attr("__call__")();
  }
  catch (...)
  {
    // TODO: Properly translate and return Python exception. #2169
    PyErr_Print();
    PyErr_Clear();
    THROW("Exception in 'search_setup_fn'");
  }
}

void search_takedown_fn(Search::search& sch)
{
  try
  {
    HookTask::task_data* d = sch.get_task_data<HookTask::task_data>();
    py::object run = *(py::object*)d->takedown_object;
    run.attr("__call__")();
  }
  catch (...)
  {
    // TODO: Properly translate and return Python exception. #2169
    PyErr_Print();
    PyErr_Clear();
    THROW("Exception in 'search_takedown_fn'");
  }
}

void py_delete_run_object(void* pyobj)
{
  py::object* o = (py::object*)pyobj;
  delete o;
}

void set_force_oracle(search_ptr sch, bool useOracle)
{
  verify_search_set_properly(sch);
  sch->set_force_oracle(useOracle);
}

void set_structured_predict_hook(
    search_ptr sch, py::object run_object, py::object setup_object, py::object takedown_object)
{
  verify_search_set_properly(sch);
  HookTask::task_data* d = sch->get_task_data<HookTask::task_data>();
  d->run_f = &search_run_fn;
  delete (py::object*)d->run_object;
  d->run_object = NULL;
  delete (py::object*)d->setup_object;
  d->setup_object = NULL;
  delete (py::object*)d->takedown_object;
  d->takedown_object = NULL;
  sch->set_force_oracle(false);
  d->run_object = new py::object(run_object);
  if (setup_object.ptr() != Py_None)
  {
    d->setup_object = new py::object(setup_object);
    d->run_setup_f = &search_setup_fn;
  }
  if (takedown_object.ptr() != Py_None)
  {
    d->takedown_object = new py::object(takedown_object);
    d->run_takedown_f = &search_takedown_fn;
  }
  d->delete_run_object = &py_delete_run_object;
}

void my_set_test_only(example_ptr ec, bool val) { ec->test_only = val; }

bool po_exists(search_ptr sch, std::string arg)
{
  HookTask::task_data* d = sch->get_task_data<HookTask::task_data>();
  return d->arg->was_supplied(arg);
}

std::string po_get_string(search_ptr sch, std::string arg)
{
  HookTask::task_data* d = sch->get_task_data<HookTask::task_data>();
  return d->arg->get_typed_option<std::string>(arg).value();
}

int32_t po_get_int(search_ptr sch, std::string arg)
{
  HookTask::task_data* d = sch->get_task_data<HookTask::task_data>();
  try
  {
    return d->arg->get_typed_option<int>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return (int32_t)d->arg->get_typed_option<size_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return (int32_t)d->arg->get_typed_option<uint32_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return (int32_t)d->arg->get_typed_option<uint64_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return d->arg->get_typed_option<uint16_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return d->arg->get_typed_option<int32_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return (int32_t)d->arg->get_typed_option<int64_t>(arg).value();
  }
  catch (...)
  {
  }
  try
  {
    return (int32_t)d->arg->get_typed_option<int16_t>(arg).value();
  }
  catch (...)
  {
  }
  // we know this'll fail but do it anyway to get the exception
  return d->arg->get_typed_option<int>(arg).value();
}

PyObject* po_get(search_ptr sch, std::string arg)
{
  try
  {
    return py::incref(py::object(po_get_string(sch, arg)).ptr());
  }
  catch (...)
  {
  }
  try
  {
    return py::incref(py::object(po_get_int(sch, arg)).ptr());
  }
  catch (...)
  {
  }
  // return None
  return py::incref(py::object().ptr());
}

void my_set_input(predictor_ptr P, example_ptr ec) { P->set_input(*ec); }
void my_set_input_at(predictor_ptr P, size_t posn, example_ptr ec) { P->set_input_at(posn, *ec); }

void my_add_oracle(predictor_ptr P, action a) { P->add_oracle(a); }
void my_add_oracles(predictor_ptr P, py::list& a)
{
  for (ssize_t i = 0; i < len(a); i++) P->add_oracle(py::extract<action>(a[i]));
}
void my_add_allowed(predictor_ptr P, action a) { P->add_allowed(a); }
void my_add_alloweds(predictor_ptr P, py::list& a)
{
  for (ssize_t i = 0; i < len(a); i++) P->add_allowed(py::extract<action>(a[i]));
}
void my_add_condition(predictor_ptr P, ptag t, char c) { P->add_condition(t, c); }
void my_add_condition_range(predictor_ptr P, ptag hi, ptag count, char name0)
{
  P->add_condition_range(hi, count, name0);
}
void my_set_oracle(predictor_ptr P, action a) { P->set_oracle(a); }
void my_set_oracles(predictor_ptr P, py::list& a)
{
  if (len(a) > 0)
    P->set_oracle(py::extract<action>(a[0]));
  else
    P->erase_oracles();
  for (ssize_t i = 1; i < len(a); i++) P->add_oracle(py::extract<action>(a[i]));
}
void my_set_allowed(predictor_ptr P, action a) { P->set_allowed(a); }
void my_set_alloweds(predictor_ptr P, py::list& a)
{
  if (len(a) > 0)
    P->set_allowed(py::extract<action>(a[0]));
  else
    P->erase_alloweds();
  for (ssize_t i = 1; i < len(a); i++) P->add_allowed(py::extract<action>(a[i]));
}
void my_set_condition(predictor_ptr P, ptag t, char c) { P->set_condition(t, c); }
void my_set_condition_range(predictor_ptr P, ptag hi, ptag count, char name0)
{
  P->set_condition_range(hi, count, name0);
}
void my_set_learner_id(predictor_ptr P, size_t id) { P->set_learner_id(id); }
void my_set_tag(predictor_ptr P, ptag t) { P->set_tag(t); }

BOOST_PYTHON_MODULE(pylibvw)
{  // This will enable user-defined docstrings and python signatures,
  // while disabling the C++ signatures
  py::docstring_options local_docstring_options(true, true, false);

  // define the vw class
  py::class_<vw, vw_ptr, boost::noncopyable>(
      "vw", "the basic VW object that holds with weight vector, parser, etc.", py::no_init)
      .def("__init__", py::make_constructor(my_initialize))
      .def("__init__", py::make_constructor(my_initialize_with_log))
      //      .def("__del__", &my_finish, "deconstruct the VW object by calling finish")
      .def("run_parser", &my_run_parser, "parse external data file")
      .def("get_learner_metrics", &get_learner_metrics,
          "get current learner stack metrics. returns empty dict if --extra_metrics was not supplied.")
      .def("finish", &my_finish, "stop VW by calling finish (and, eg, write weights to disk)")
      .def("save", &my_save, "save model to filename")
      .def("learn", &my_learn, "given a pyvw example, learn (and predict) on that example")
      .def("predict", &my_predict, "given a pyvw example, predict on that example")
      .def("hash_space", &VW::hash_space, "given a namespace (as a string), compute the hash of that namespace")
      .def("hash_feature", &VW::hash_feature,
          "given a feature string (arg2) and a hashed namespace (arg3), hash that feature")
      .def("_finish_example", &my_finish_example, "tell VW that you're done with a given example")
      .def("_finish_example_multi_ex", &my_finish_multi_ex, "tell VW that you're done with the given examples")
      .def("setup_example", &my_setup_example,
          "given an example that you've created by hand, prepare it for learning (eg, compute quadratic feature)")
      .def("unsetup_example", &unsetup_example,
          "reverse the process of setup, so that you can go back and modify this example")

      .def("num_weights", &VW::num_weights, "how many weights are we learning?")
      .def("get_weight", &VW::get_weight, "get the weight for a particular index")
      .def("set_weight", &VW::set_weight, "set the weight for a particular index")
      .def("get_stride", &VW::get_stride, "return the internal stride")

      .def("get_label_type", &my_get_label_type, "return parse label type")
      .def("get_prediction_type", &my_get_prediction_type, "return prediction type")
      .def("get_sum_loss", &get_sum_loss, "return the total cumulative loss suffered so far")
      .def("get_weighted_examples", &get_weighted_examples, "return the total weight of examples so far")

      .def("get_search_ptr", &get_search_ptr, "return a pointer to the search data structure")
      .def("get_options", &get_options, "get available vw options")
      .def("audit_example", &my_audit_example, "print example audit information")
      .def("get_id", &get_model_id, "return the model id")
      .def("get_arguments", &get_arguments, "return the arguments after resolving all dependencies")
      .def("get_enabled_reductions", &get_enabled_reductions, "return the list of names of the enabled reductions")

      .def("learn_multi", &my_learn_multi_ex, "given a list pyvw examples, learn (and predict) on those examples")
      .def("predict_multi", &my_predict_multi_ex, "given a list of pyvw examples, predict on that example")
      .def("_parse", &my_parse, "Parse a string into a collection of VW examples")
      .def("_is_multiline", &my_is_multiline, "true if the base reduction is multiline")

      .def_readonly("lDefault", lDEFAULT,
          "Default label type (whatever vw was initialized with) -- used as input to the example() initializer")
      .def_readonly("lBinary", lBINARY, "Binary label type -- used as input to the example() initializer")
      .def_readonly("lMulticlass", lMULTICLASS, "Multiclass label type -- used as input to the example() initializer")
      .def_readonly("lCostSensitive", lCOST_SENSITIVE,
          "Cost sensitive label type (for LDF!) -- used as input to the example() initializer")
      .def_readonly("lContextualBandit", lCONTEXTUAL_BANDIT,
          "Contextual bandit label type -- used as input to the example() initializer")
      .def_readonly("lConditionalContextualBandit", lCONDITIONAL_CONTEXTUAL_BANDIT,
          "Conditional Contextual bandit label type -- used as input to the example() initializer")
      .def_readonly("lSlates", lSLATES, "Slates label type -- used as input to the example() initializer")
      .def_readonly("lContinuous", lCONTINUOUS, "Continuous label type -- used as input to the example() initializer")

      .def_readonly("pSCALAR", pSCALAR, "Scalar prediction type")
      .def_readonly("pSCALARS", pSCALARS, "Multiple scalar-valued prediction type")
      .def_readonly("pACTION_SCORES", pACTION_SCORES, "Multiple action scores prediction type")
      .def_readonly("pACTION_PROBS", pACTION_PROBS, "Multiple action probabilities prediction type")
      .def_readonly("pMULTICLASS", pMULTICLASS, "Multiclass prediction type")
      .def_readonly("pMULTILABELS", pMULTILABELS, "Multilabel prediction type")
      .def_readonly("pPROB", pPROB, "Probability prediction type")
      .def_readonly("pMULTICLASSPROBS", pMULTICLASSPROBS, "Multiclass probabilities prediction type")
      .def_readonly("pDECISION_SCORES", pDECISION_SCORES, "Decision scores prediction type")
      .def_readonly("pACTION_PDF_VALUE", pACTION_PDF_VALUE, "Action pdf value prediction type")
      .def_readonly("pPDF", pPDF, "PDF prediction type")
      .def_readonly("pACTIVE_MULTICLASS", pACTIVE_MULTICLASS, "Active multiclass prediction type");

  // define the example class
  py::class_<example, example_ptr, boost::noncopyable>("example", py::no_init)
      .def("__init__", py::make_constructor(my_read_example),
          "Given a string as an argument parse that into a VW example (and run setup on it) -- default to multiclass "
          "label type")
      .def("__init__", py::make_constructor(my_empty_example),
          "Construct an empty (non setup) example; you must provide a label type (vw.lBinary, vw.lMulticlass, etc.)")
      .def("__init__", py::make_constructor(my_existing_example),
          "Create a new example object pointing to an existing object.")

      .def("set_test_only", &my_set_test_only, "Change the test-only bit on an example")

      .def("get_tag", &my_get_tag, "Returns the tag associated with this example")
      .def("get_topic_prediction", &VW::get_topic_prediction,
          "For LDA models, returns the topic prediction for the topic id given")
      .def("get_feature_number", &VW::get_feature_number, "Returns the total number of features for this example")

      .def("get_example_counter", &get_example_counter,
          "Returns the counter of total number of examples seen up to and including this one")
      .def("get_ft_offset", &get_ft_offset,
          "Returns the feature offset for this example (used, eg, by multiclass classification to bulk offset all "
          "features)")
      .def("get_partial_prediction", &get_partial_prediction,
          "Returns the partial prediction associated with this example")
      .def("get_updated_prediction", &get_updated_prediction,
          "Returns the partial prediction as if we had updated it after learning")
      .def("get_loss", &get_loss, "Returns the loss associated with this example")
      .def("get_total_sum_feat_sq", &get_total_sum_feat_sq, "The total sum of feature-value squared for this example")

      .def("num_namespaces", &ex_num_namespaces, "The total number of namespaces associated with this example")
      .def("namespace", &ex_namespace,
          "Get the namespace id for namespace i (for i = 0.. num_namespaces); specifically returns the ord() of the "
          "corresponding character id")
      .def("sum_feat_sq", &ex_sum_feat_sq,
          "Get the sum of feature-values squared for a given namespace id (id=character-ord)")
      .def("num_features_in", &ex_num_features, "Get the number of features in a given namespace id (id=character-ord)")
      .def("feature", &ex_feature, "Get the feature id for the ith feature in a given namespace id (id=character-ord)")
      .def("feature_weight", &ex_feature_weight, "The the feature value (weight) per .feature(...)")

      .def("push_hashed_feature", &ex_push_feature, "Add a hashed feature to a given namespace (id=character-ord)")
      .def("push_feature_list", &ex_push_feature_list, "Add a (Python) list of features to a given namespace")
      .def("push_feature_dict", &ex_push_dictionary, "Add a (Python) dictionary of namespace/feature-list pairs")
      .def("pop_feature", &ex_pop_feature,
          "Remove the top feature from a given namespace; returns True iff the list was non-empty")
      .def("push_namespace", &ex_push_namespace, "Add a new namespace")
      .def("ensure_namespace_exists", &ex_ensure_namespace_exists, "Add a new namespace if it doesn't already exist")
      .def("pop_namespace", &ex_pop_namespace, "Remove the top namespace off; returns True iff the list was non-empty")
      .def("erase_namespace", &ex_erase_namespace, "Remove all the features from a given namespace")

      .def("set_label_string", &ex_set_label_string, "(Re)assign the label of this example to this string")
      .def("get_simplelabel_label", &ex_get_simplelabel_label,
          "Assuming a simple_label label type, return the corresponding label (class/regression target/etc.)")
      .def("get_simplelabel_weight", &ex_get_simplelabel_weight,
          "Assuming a simple_label label type, return the importance weight")
      .def("get_simplelabel_initial", &ex_get_simplelabel_initial,
          "Assuming a simple_label label type, return the initial (baseline) prediction")
      .def("get_simplelabel_prediction", &ex_get_simplelabel_prediction,
          "Assuming a simple_label label type, return the final prediction")
      .def("get_multiclass_label", &ex_get_multiclass_label, "Assuming a multiclass label type, get the true label")
      .def("get_multiclass_weight", &ex_get_multiclass_weight,
          "Assuming a multiclass label type, get the importance weight")
      .def("get_multiclass_prediction", &ex_get_multiclass_prediction,
          "Assuming a multiclass label type, get the prediction")
      .def("get_prob", &ex_get_prob, "Get probability from example prediction")
      .def("get_scalars", &ex_get_scalars, "Get scalar values from example prediction")
      .def("get_action_scores", &ex_get_action_scores, "Get action scores from example prediction")
      .def("get_decision_scores", &ex_get_decision_scores, "Get decision scores from example prediction")
      .def("get_action_pdf_value", &ex_get_action_pdf_value, "Get action and pdf value from example prediction")
      .def("get_pdf", &ex_get_pdf, "Get pdf from example prediction")
      .def("get_active_multiclass", &ex_get_active_multiclass, "Get active multiclass from example prediction")
      .def("get_multilabel_predictions", &ex_get_multilabel_predictions,
          "Get multilabel predictions from example prediction")
      .def("get_costsensitive_prediction", &ex_get_costsensitive_prediction,
          "Assuming a cost_sensitive label type, get the prediction")
      .def("get_costsensitive_num_costs", &ex_get_costsensitive_num_costs,
          "Assuming a cost_sensitive label type, get the total number of label/cost pairs")
      .def("get_costsensitive_cost", &ex_get_costsensitive_cost,
          "Assuming a cost_sensitive label type, get the cost for a given pair (i=0.. get_costsensitive_num_costs)")
      .def("get_costsensitive_class", &ex_get_costsensitive_class,
          "Assuming a cost_sensitive label type, get the label for a given pair (i=0.. get_costsensitive_num_costs)")
      .def("get_costsensitive_partial_prediction", &ex_get_costsensitive_partial_prediction,
          "Assuming a cost_sensitive label type, get the partial prediction for a given pair (i=0.. "
          "get_costsensitive_num_costs)")
      .def("get_costsensitive_wap_value", &ex_get_costsensitive_wap_value,
          "Assuming a cost_sensitive label type, get the weighted-all-pairs recomputed cost for a given pair (i=0.. "
          "get_costsensitive_num_costs)")
      .def("get_cbandits_prediction", &ex_get_cbandits_prediction,
          "Assuming a contextual_bandits label type, get the prediction")
      .def("get_cbandits_num_costs", &ex_get_cbandits_num_costs,
          "Assuming a contextual_bandits label type, get the total number of label/cost pairs")
      .def("get_cbandits_cost", &ex_get_cbandits_cost,
          "Assuming a contextual_bandits label type, get the cost for a given pair (i=0.. get_cbandits_num_costs)")
      .def("get_cbandits_class", &ex_get_cbandits_class,
          "Assuming a contextual_bandits label type, get the label for a given pair (i=0.. get_cbandits_num_costs)")
      .def("get_cbandits_probability", &ex_get_cbandits_probability,
          "Assuming a contextual_bandits label type, get the bandits probability for a given pair (i=0.. "
          "get_cbandits_num_costs)")
      .def("get_cbandits_partial_prediction", &ex_get_cbandits_partial_prediction,
          "Assuming a contextual_bandits label type, get the partial prediction for a given pair (i=0.. "
          "get_cbandits_num_costs)");

  py::class_<Search::predictor, predictor_ptr>("predictor", py::no_init)
      .def("set_input", &my_set_input, "set the input (an example) for this predictor (non-LDF mode only)")
      //.def("set_input_ldf", &my_set_input_ldf, "set the inputs (a list of examples) for this predictor (LDF mode
      // only)")
      .def(
          "set_input_length", &Search::predictor::set_input_length, "declare the length of an LDF-sequence of examples")
      .def("set_input_at", &my_set_input_at,
          "put a given example at position in the LDF sequence (call after set_input_length)")
      .def("add_oracle", &my_add_oracle, "add an action to the current list of oracle actions")
      .def("add_oracles", &my_add_oracles, "add a list of actions to the current list of oracle actions")
      .def("add_allowed", &my_add_allowed, "add an action to the current list of allowed actions")
      .def("add_alloweds", &my_add_alloweds, "add a list of actions to the current list of allowed actions")
      .def("add_condition", &my_add_condition, "add a (tag,char) pair to the list of variables on which to condition")
      .def("add_condition_range", &my_add_condition_range,
          "given (tag,len,char), add (tag,char), (tag-1,char+1), ..., (tag-len,char+len) to the list of conditionings")
      .def("set_oracle", &my_set_oracle, "set an action as the current list of oracle actions")
      .def("set_oracles", &my_set_oracles, "set a list of actions as the current list of oracle actions")
      .def("set_allowed", &my_set_allowed, "set an action as the current list of allowed actions")
      .def("set_alloweds", &my_set_alloweds, "set a list of actions as the current list of allowed actions")
      .def("set_condition", &my_set_condition, "set a (tag,char) pair as the list of variables on which to condition")
      .def("set_condition_range", &my_set_condition_range,
          "given (tag,len,char), set (tag,char), (tag-1,char+1), ..., (tag-len,char+len) as the list of conditionings")
      .def("set_learner_id", &my_set_learner_id, "select the learner with which to make this prediction")
      .def("set_tag", &my_set_tag, "change the tag of this prediction")
      .def("predict", &Search::predictor::predict, "make a prediction");

  py::class_<py_log_wrapper, py_log_wrapper_ptr>(
      "vw_log", "do not use, see pyvw.vw.init(enable_logging..)", py::init<py::object>());

  py::class_<Search::search, search_ptr>("search")
      .def("set_options", &Search::search::set_options, "Set global search options (auto conditioning, etc.)")
      //.def("set_num_learners", &Search::search::set_num_learners, "Set the total number of learners you want to
      // train")
      .def("get_history_length", &Search::search::get_history_length,
          "Get the value specified by --search_history_length")
      .def("loss", &Search::search::loss, "Declare a (possibly incremental) loss")
      .def("should_output", &search_should_output,
          "Check whether search wants us to output (only happens if you have -p running)")
      .def("predict_needs_example", &Search::search::predictNeedsExample,
          "Check whether a subsequent call to predict is actually going to use the example you pass---i.e., can you "
          "skip feature computation?")
      .def("output", &search_output, "Add a string to the coutput (should only do if should_output returns True)")
      .def("get_num_actions", &search_get_num_actions, "Return the total number of actions search was initialized with")
      .def("set_structured_predict_hook", &set_structured_predict_hook,
          "Set the hook (function pointer) that search should use for structured prediction (you don't want to call "
          "this yourself!")
      .def("set_force_oracle", &set_force_oracle, "For oracle decoding when .predict is run")
      .def("is_ldf", &Search::search::is_ldf, "check whether this search task is running in LDF mode")

      .def("po_exists", &po_exists,
          "For program (cmd line) options, check to see if a given option was specified; eg sch.po_exists(\"search\") "
          "should be True")
      .def("po_get", &po_get,
          "For program (cmd line) options, if an option was specified, get its value; eg sch.po_get(\"search\") should "
          "return the # of actions (returns either int or string)")
      .def("po_get_str", &po_get_string, "Same as po_get, but specialized for string return values.")
      .def("po_get_int", &po_get_int, "Same as po_get, but specialized for integer return values.")

      .def("get_predictor", &get_predictor,
          "Get a predictor object that can be used for making predictions; requires a tag argument to tag the "
          "prediction.")

      .def_readonly("AUTO_CONDITION_FEATURES", Search::AUTO_CONDITION_FEATURES,
          "Tell search to automatically add features based on conditioned-on variables")
      .def_readonly("AUTO_HAMMING_LOSS", Search::AUTO_HAMMING_LOSS,
          "Tell search to automatically compute hamming loss over predictions")
      .def_readonly("EXAMPLES_DONT_CHANGE", Search::EXAMPLES_DONT_CHANGE,
          "Tell search that on a single structured 'run', you don't change the examples you pass to predict")
      .def_readonly("IS_LDF", Search::IS_LDF, "Tell search that this is an LDF task");
}
