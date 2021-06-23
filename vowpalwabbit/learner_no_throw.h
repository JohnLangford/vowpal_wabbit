#pragma once

#include "reductions_fwd.h"
#include "example.h"
#include "metric_sink.h"
// #include "simple_label.h"
#include "debug_log.h"

#undef VW_DEBUG_LOG
#define VW_DEBUG_LOG vw_dbg::learner

enum class prediction_type_t
{
  scalar,
  scalars,
  action_scores,
  pdf,
  action_probs,
  multiclass,
  multilabels,
  prob,
  multiclassprobs,
  decision_probs,
  action_pdf_value,
  active_multiclass
};

namespace VW
{
/// \brief Contains the VW::LEARNER::learner object and utilities for
/// interacting with it.
namespace LEARNER
{
template <class T, class E>
struct learner;

/// \brief Used to type erase the object and pass around common type.
using base_learner = learner<char, char>;

/// \brief Used for reductions that process single ::example objects at at time.
/// It type erases the specific reduction object type.
using single_learner = learner<char, example>;

struct func_data
{
  using fn = void (*)(void* data);
  void* data = nullptr;
  base_learner* base = nullptr;
  fn func = nullptr;
};

inline func_data tuple_dbf(void* data, base_learner* base, void (*func)(void*))
{
  func_data foo;
  foo.data = data;
  foo.base = base;
  foo.func = func;
  return foo;
}

inline void noop_save_load(void*, io_buf&, bool, bool) {}
inline void noop(void*) {}
inline void noop_persist_metrics(void*, metric_sink&) {}
inline float noop_sensitivity(void*, base_learner&, example&)
{
  // TODO: where should this print to? Just delete this?
  std::cout << std::endl;
  return 0.;
}

float recur_sensitivity(void*, base_learner&, example&);

inline void debug_increment_depth(example& ex)
{
  if (vw_dbg::track_stack) ++ex._debug_current_reduction_depth;
}

inline void debug_increment_depth(multi_ex& ec_seq)
{
  if (vw_dbg::track_stack)
  {
    for (auto& ec : ec_seq) { ++ec->_debug_current_reduction_depth; }
  }
}

inline void debug_decrement_depth(example& ex)
{
  if (vw_dbg::track_stack) --ex._debug_current_reduction_depth;
}

inline void debug_decrement_depth(multi_ex& ec_seq)
{
  if (vw_dbg::track_stack)
  {
    for (auto& ec : ec_seq) { --ec->_debug_current_reduction_depth; }
  }
}

inline void increment_offset(example& ex, const size_t increment, const size_t i)
{
  ex.ft_offset += static_cast<uint32_t>(increment * i);
  debug_increment_depth(ex);
}

inline void increment_offset(multi_ex& ec_seq, const size_t increment, const size_t i)
{
  for (auto& ec : ec_seq) { ec->ft_offset += static_cast<uint32_t>(increment * i); }
  debug_increment_depth(ec_seq);
}

inline void decrement_offset(example& ex, const size_t increment, const size_t i)
{
  assert(ex.ft_offset >= increment * i);
  ex.ft_offset -= static_cast<uint32_t>(increment * i);
  debug_decrement_depth(ex);
}

inline void decrement_offset(multi_ex& ec_seq, const size_t increment, const size_t i)
{
  for (auto ec : ec_seq)
  {
    assert(ec->ft_offset >= increment * i);
    ec->ft_offset -= static_cast<uint32_t>(increment * i);
  }
  debug_decrement_depth(ec_seq);
}

struct learn_data
{
  using fn = void (*)(void* data, base_learner& base, void* ex);
  using multi_fn = void (*)(void* data, base_learner& base, void* ex, size_t count, size_t step, polyprediction* pred,
      bool finalize_predictions);

  void* data = nullptr;
  base_learner* base = nullptr;
  fn learn_f = nullptr;
  fn predict_f = nullptr;
  fn update_f = nullptr;
  multi_fn multipredict_f = nullptr;
};

struct save_metric_data
{
  using fn = void (*)(void*, metric_sink& metrics);
  void* data = nullptr;
  base_learner* base = nullptr;
  fn save_metric_f = nullptr;
};

struct sensitivity_data
{
  using fn = float (*)(void* data, base_learner& base, example& ex);
  void* data = nullptr;
  fn sensitivity_f = nullptr;
};

struct save_load_data
{
  using fn = void (*)(void*, io_buf&, bool read, bool text);
  void* data = nullptr;
  base_learner* base = nullptr;
  fn save_load_f = nullptr;
};

struct finish_example_data
{
  using fn = void (*)(vw&, void* data, void* ex);
  void* data = nullptr;
  base_learner* base = nullptr;
  fn finish_example_f = nullptr;
  fn print_example_f = nullptr;
};

/// \brief Defines the interface for a learning algorithm.
///
/// Learner is implemented as a struct of pointers, and associated methods. It
/// implements a sort of virtual inheritance through the use of bundling
/// function pointers with the associated objects to call them with. A reduction
/// will recursively call the base given to it, whereas a base learner will not
/// recurse and will simply return the result. Learner is not intended to be
/// inherited from. Instead it is used through composition, and created through
/// the various VW::LEARNER::init_learner overloaded functions that chain to the
/// central factor function VW::LEARNER::learner::init_learner The state of this
/// learner, or reduction, is stored in the learner_data field. A
/// <code>std::shared_pointer<void></code> is used as this class uses type
/// erasure to allow for an arbitrary reduction to be implemented. It is
/// extremely important that the function pointers given to the class match the
/// expected types of the object. If the learner is constructed using
/// VW::LEARNER::learner::init_learner and assembled before it is transformed
/// into a VW::LEARNER::base_learner with VW::LEARNER::make_base then the usage
/// of the templated functions should ensure types are correct.
///
/// \tparam T Type of the reduction data object stored. This allows this
/// specific reduction to have it's own state.
/// \tparam E Example type this reduction supports. Must be one of ::example or
/// ::multi_ex
template <class T, class E>
struct learner
{
private:
  template <class FluentBuilderT, class DataT, class ExampleT, class BaseLearnerT>
  friend struct common_learner_builder;
  template <class DataT, class ExampleT>
  friend struct base_learner_builder;
  template <class DataT, class ExampleT, class BaseLearnerT>
  friend struct reduction_learner_builder;
  template <class ExampleT, class BaseLearnerT>
  friend struct reduction_no_data_learner_builder;

  func_data init_fd;
  learn_data learn_fd;
  sensitivity_data sensitivity_fd;
  finish_example_data finish_example_fd;
  save_load_data save_load_fd;
  func_data end_pass_fd;
  func_data end_examples_fd;
  save_metric_data persist_metrics_fd;
  func_data finisher_fd;
  std::string name;  // Name of the reduction.  Used in VW_DBG to trace nested learn() and predict() calls

  std::shared_ptr<void> learner_data;

  learner(){};  // Should only be able to construct a learner through
                // init_learner function

public:
  prediction_type_t pred_type;
  size_t weights;  // this stores the number of "weight vectors" required by the learner.
  size_t increment;
  bool is_multiline;  // Is this a single-line or multi-line reduction?

  // learn will return a prediction.  The framework should
  // not call predict before learn
  bool learn_returns_prediction = false;

  using end_fptr_type = void (*)(vw&, void*, void*);
  using finish_fptr_type = void (*)(void*);

  void debug_log_message(example& ec, const std::string& msg)
  {
    VW_DBG(ec) << "[" << name << "." << msg << "]" << std::endl;
  }

  void debug_log_message(multi_ex& ec, const std::string& msg)
  {
    VW_DBG(*ec[0]) << "[" << name << "." << msg << "]" << std::endl;
  }

  /// \brief Will update the model according to the labels and examples supplied.
  /// \param ec The ::example object or ::multi_ex to be operated on. This
  /// object **must** have a valid label set for every ::example in the field
  /// example::l that corresponds to the type this reduction expects.
  /// \param i This is the offset used for the weights in this call. If using
  /// multiple regressors/learners you can increment this value for each call.
  /// \returns While some reductions may fill the example::pred, this is not
  /// guaranteed and is undefined behavior if accessed.
  inline void learn(E& ec, size_t i = 0)
  {
    assert((is_multiline && std::is_same<multi_ex, E>::value) ||
        (!is_multiline && std::is_same<example, E>::value));  // sanity check under debug compile
    increment_offset(ec, increment, i);
    debug_log_message(ec, "learn");
    learn_fd.learn_f(learn_fd.data, *learn_fd.base, (void*)&ec);
    decrement_offset(ec, increment, i);
  }

  /// \brief Make a prediction for the given example.
  /// \param ec The ::example object or ::multi_ex to be operated on. This
  /// object **must** have a valid prediction allocated in the field
  /// example::pred that corresponds to this reduction type.
  /// \param i This is the offset used for the weights in this call. If using
  /// multiple regressors/learners you can increment this value for each call.
  /// \returns The prediction calculated by this reduction be set on
  /// example::pred. If <code>E</code> is ::multi_ex then the prediction is set
  /// on the 0th item in the list.
  inline void predict(E& ec, size_t i = 0)
  {
    assert((is_multiline && std::is_same<multi_ex, E>::value) ||
        (!is_multiline && std::is_same<example, E>::value));  // sanity check under debug compile
    increment_offset(ec, increment, i);
    debug_log_message(ec, "predict");
    learn_fd.predict_f(learn_fd.data, *learn_fd.base, (void*)&ec);
    decrement_offset(ec, increment, i);
  }

  inline void multipredict(E& ec, size_t lo, size_t count, polyprediction* pred, bool finalize_predictions)
  {
    assert((is_multiline && std::is_same<multi_ex, E>::value) ||
        (!is_multiline && std::is_same<example, E>::value));  // sanity check under debug compile
    if (learn_fd.multipredict_f == nullptr)
    {
      increment_offset(ec, increment, lo);
      debug_log_message(ec, "multipredict");
      for (size_t c = 0; c < count; c++)
      {
        learn_fd.predict_f(learn_fd.data, *learn_fd.base, (void*)&ec);
        if (finalize_predictions)
          pred[c] = std::move(ec.pred);  // TODO: this breaks for complex labels because = doesn't do deep copy! (XXX we
                                         // "fix" this by moving)
        else
          pred[c].scalar = ec.partial_prediction;
        // pred[c].scalar = finalize_prediction ec.partial_prediction; // TODO: this breaks for complex labels because =
        // doesn't do deep copy! // note works if ec.partial_prediction, but only if finalize_prediction is run????
        increment_offset(ec, increment, 1);
      }
      decrement_offset(ec, increment, lo + count);
    }
    else
    {
      increment_offset(ec, increment, lo);
      debug_log_message(ec, "multipredict");
      learn_fd.multipredict_f(learn_fd.data, *learn_fd.base, (void*)&ec, count, increment, pred, finalize_predictions);
      decrement_offset(ec, increment, lo);
    }
  }

  template <class L>
  inline void set_predict(void (*u)(T&, L&, E&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    learn_fd.predict_f = (learn_data::fn)u;
    VW_WARNING_STATE_POP
  }
  template <class L>
  inline void set_learn(void (*u)(T&, L&, E&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    learn_fd.learn_f = (learn_data::fn)u;
    VW_WARNING_STATE_POP
  }
  template <class L>
  inline void set_multipredict(void (*u)(T&, L&, E&, size_t, size_t, polyprediction*, bool))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    learn_fd.multipredict_f = (learn_data::multi_fn)u;
    VW_WARNING_STATE_POP
  }

  inline void update(E& ec, size_t i = 0)
  {
    assert((is_multiline && std::is_same<multi_ex, E>::value) ||
        (!is_multiline && std::is_same<example, E>::value));  // sanity check under debug compile
    increment_offset(ec, increment, i);
    debug_log_message(ec, "update");
    learn_fd.update_f(learn_fd.data, *learn_fd.base, (void*)&ec);
    decrement_offset(ec, increment, i);
  }
  template <class L>
  inline void set_update(void (*u)(T& data, L& base, E&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    learn_fd.update_f = (learn_data::fn)u;
    VW_WARNING_STATE_POP
  }

  // used for active learning and confidence to determine how easily predictions are changed
  inline void set_sensitivity(float (*u)(T& data, base_learner& base, example&))
  {
    sensitivity_fd.data = learn_fd.data;
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    sensitivity_fd.sensitivity_f = (sensitivity_data::fn)u;
    VW_WARNING_STATE_POP
  }
  inline float sensitivity(example& ec, size_t i = 0)
  {
    increment_offset(ec, increment, i);
    debug_log_message(ec, "sensitivity");
    const float ret = sensitivity_fd.sensitivity_f(sensitivity_fd.data, *learn_fd.base, ec);
    decrement_offset(ec, increment, i);
    return ret;
  }

  // called anytime saving or loading needs to happen. Autorecursive.
  inline void save_load(io_buf& io, const bool read, const bool text)
  {
    save_load_fd.save_load_f(save_load_fd.data, io, read, text);
    if (save_load_fd.base) save_load_fd.base->save_load(io, read, text);
  }
  inline void set_save_load(void (*sl)(T&, io_buf&, bool, bool))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    save_load_fd.save_load_f = (save_load_data::fn)sl;
    VW_WARNING_STATE_POP
    save_load_fd.data = learn_fd.data;
    save_load_fd.base = learn_fd.base;
  }

  // called when metrics is enabled.  Autorecursive.
  void persist_metrics(metric_sink& metrics)
  {
    persist_metrics_fd.save_metric_f(persist_metrics_fd.data, metrics);
    if (persist_metrics_fd.base) persist_metrics_fd.base->persist_metrics(metrics);
  }
  void set_persist_metrics(void (*f)(T&, metric_sink&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    persist_metrics_fd.save_metric_f = (save_metric_data::fn)f;
    VW_WARNING_STATE_POP
    persist_metrics_fd.data = learn_fd.data;
    persist_metrics_fd.base = learn_fd.base;
  }

  // called to clean up state.  Autorecursive.
  void set_finish(void (*f)(T&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    finisher_fd = tuple_dbf(learn_fd.data, learn_fd.base, (finish_fptr_type)(f));
    VW_WARNING_STATE_POP
  }

  inline void finish()
  {
    if (finisher_fd.data) { finisher_fd.func(finisher_fd.data); }
    if (finisher_fd.base)
    {
      finisher_fd.base->finish();
      delete finisher_fd.base;
    }
  }

  void end_pass()
  {
    end_pass_fd.func(end_pass_fd.data);
    if (end_pass_fd.base) end_pass_fd.base->end_pass();
  }  // autorecursive

  void set_end_pass(void (*f)(T&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    end_pass_fd = tuple_dbf(learn_fd.data, learn_fd.base, (func_data::fn)f);
    VW_WARNING_STATE_POP
  }

  // called after parsing of examples is complete.  Autorecursive.
  void end_examples()
  {
    end_examples_fd.func(end_examples_fd.data);
    if (end_examples_fd.base) end_examples_fd.base->end_examples();
  }
  void set_end_examples(void (*f)(T&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    end_examples_fd = tuple_dbf(learn_fd.data, learn_fd.base, (func_data::fn)f);
    VW_WARNING_STATE_POP
  }

  // Called at the beginning by the driver.  Explicitly not recursive.
  void init_driver() { init_fd.func(init_fd.data); }
  void set_init_driver(void (*f)(T&))
  {
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    init_fd = tuple_dbf(learn_fd.data, learn_fd.base, (func_data::fn)f);
    VW_WARNING_STATE_POP
  }

  // called after learn example for each example.  Explicitly not recursive.
  inline void finish_example(vw& all, E& ec)
  {
    debug_log_message(ec, "finish_example");
    finish_example_fd.finish_example_f(all, finish_example_fd.data, (void*)&ec);
  }
  // called after learn example for each example.  Explicitly not recursive.
  void set_finish_example(void (*f)(vw& all, T&, E&))
  {
    finish_example_fd.data = learn_fd.data;
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    finish_example_fd.finish_example_f = (end_fptr_type)(f);
    VW_WARNING_STATE_POP
  }

  // never called, convienience method in case reduction has a seperate
  // print fn that mirrors the fn received on set_finish_example(..)
  //
  // usually finish_example routine prints and then deallocs
  // that printing logic can be registered here
  void set_print_example(void (*f)(vw& all, T&, E&))
  {
    finish_example_fd.data = learn_fd.data;
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    finish_example_fd.print_example_f = (end_fptr_type)(f);
    VW_WARNING_STATE_POP
  }

  inline void print_example(vw& all, E& ec);

  template <class L>
  static learner<T, E>& init_learner(T* dat, L* base, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&), size_t ws,
      prediction_type_t pred_type, const std::string& name, bool learn_returns_prediction = false)
  {
    learner<T, E>& ret = *(new learner<T, E>());

    if (base != nullptr)
    {  // a reduction

      // This is a copy assignment into the current object. The purpose is to copy all of the
      // function data objects so that if this reduction does not define a function such as
      // save_load then calling save_load on this object will essentially result in forwarding the
      // call the next reduction that actually implements it.
      ret = *(learner<T, E>*)(base);

      ret.learn_fd.base = make_base(*base);
      ret.sensitivity_fd.sensitivity_f = static_cast<sensitivity_data::fn>(recur_sensitivity);
      ret.finisher_fd.data = dat;
      ret.finisher_fd.base = make_base(*base);
      ret.finisher_fd.func = static_cast<func_data::fn>(noop);
      ret.weights = ws;
      ret.increment = base->increment * ret.weights;
    }
    else  // a base learner
    {
      ret.weights = 1;
      ret.increment = ws;
      ret.end_pass_fd.func = static_cast<func_data::fn>(noop);
      ret.end_examples_fd.func = static_cast<func_data::fn>(noop);
      ret.persist_metrics_fd.save_metric_f = static_cast<save_metric_data::fn>(noop_persist_metrics);
      ret.init_fd.func = static_cast<func_data::fn>(noop);
      ret.save_load_fd.save_load_f = static_cast<save_load_data::fn>(noop_save_load);
      ret.finisher_fd.data = dat;
      ret.finisher_fd.func = static_cast<func_data::fn>(noop);
      ret.sensitivity_fd.sensitivity_f = static_cast<sensitivity_data::fn>(noop_sensitivity);
      ret.finish_example_fd.data = dat;
      VW_WARNING_STATE_PUSH
      VW_WARNING_DISABLE_CAST_FUNC_TYPE
      ret.finish_example_fd.finish_example_f = reinterpret_cast<finish_example_data::fn>(return_simple_example);
      VW_WARNING_STATE_POP
    }

    ret.name = name;
    ret.learner_data = std::shared_ptr<T>(dat, [](T* ptr) {
      ptr->~T();
      free(ptr);
    });

    ret.learn_fd.data = dat;
    VW_WARNING_STATE_PUSH
    VW_WARNING_DISABLE_CAST_FUNC_TYPE
    ret.learn_fd.learn_f = reinterpret_cast<learn_data::fn>(learn);
    ret.learn_fd.update_f = (learn_data::fn)learn;
    ret.learn_fd.predict_f = (learn_data::fn)predict;
    VW_WARNING_STATE_POP
    ret.learn_fd.multipredict_f = nullptr;
    ret.pred_type = pred_type;
    ret.is_multiline = std::is_same<multi_ex, E>::value;
    ret.learn_returns_prediction = learn_returns_prediction;

    VW_DBG_0 << "Added Reduction: " << name << std::endl;

    return ret;
  }

  base_learner* get_learner_by_name_prefix(std::string reduction_name);
};

// OLD WAY:
// The init_learner set of functions is the old way to create a learner. They
// have been replaced with:
//   - make_reduction_learner
//   - make_base_learner
// They were replaced due to how many different overloads there are and the
// fact defaults are very hard to express. This problem got worse as more
// arguments got added.
template <class T, class E, class L>
learner<T, E>& init_learner(free_ptr<T>& dat, L* base, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&),
    size_t ws, prediction_type_t pred_type, const std::string& name, bool learn_returns_prediction = false)
{
  auto ret =
      &learner<T, E>::init_learner(dat.get(), base, learn, predict, ws, pred_type, name, learn_returns_prediction);

  dat.release();
  return *ret;
}

// base learner/predictor
template <class T, class E, class L>
learner<T, E>& init_learner(free_ptr<T>& dat, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&),
    size_t params_per_weight, const std::string& name, bool learn_returns_prediction = false)
{
  auto ret = &learner<T, E>::init_learner(dat.get(), (L*)nullptr, learn, predict, params_per_weight,
      prediction_type_t::scalar, name, learn_returns_prediction);

  dat.release();
  return *ret;
}

// base predictor only
template <class T, class E, class L>
learner<T, E>& init_learner(void (*predict)(T&, L&, E&), size_t params_per_weight, const std::string& name)
{
  return learner<T, E>::init_learner(
      nullptr, (L*)nullptr, predict, predict, params_per_weight, prediction_type_t::scalar, name);
}

template <class T, class E, class L>
learner<T, E>& init_learner(free_ptr<T>& dat, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&),
    size_t params_per_weight, prediction_type_t pred_type, const std::string& name,
    bool learn_returns_prediction = false)
{
  auto ret = &learner<T, E>::init_learner(
      dat.get(), (L*)nullptr, learn, predict, params_per_weight, pred_type, name, learn_returns_prediction);
  dat.release();
  return *ret;
}

// reduction with default prediction type
template <class T, class E, class L>
learner<T, E>& init_learner(free_ptr<T>& dat, L* base, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&),
    size_t ws, const std::string& name, bool learn_returns_prediction = false)
{
  auto ret = &learner<T, E>::init_learner(
      dat.get(), base, learn, predict, ws, base->pred_type, name, learn_returns_prediction);

  dat.release();
  return *ret;
}

// reduction with default num_params
template <class T, class E, class L>
learner<T, E>& init_learner(free_ptr<T>& dat, L* base, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&),
    const std::string& name, bool learn_returns_prediction = false)
{
  auto ret =
      &learner<T, E>::init_learner(dat.get(), base, learn, predict, 1, base->pred_type, name, learn_returns_prediction);

  dat.release();
  return *ret;
}

// Reduction with no data.
template <class T, class E, class L>
learner<T, E>& init_learner(L* base, void (*learn)(T&, L&, E&), void (*predict)(T&, L&, E&), const std::string& name,
    bool learn_returns_prediction = false)
{
  return learner<T, E>::init_learner(nullptr, base, learn, predict, 1, base->pred_type, name, learn_returns_prediction);
}

template <class T, class E>
base_learner* make_base(learner<T, E>& base)
{
  return (base_learner*)(&base);
}

template <class T, class E>
single_learner* as_singleline_no_throw(learner<T, E>* l, bool& success)
{
  success = false;
  if (!l->is_multiline)  // Tried to use a multiline reduction as a singleline reduction
  {
    success = true;
    return (single_learner*)(l);
  }
  return nullptr;
}

VW_WARNING_STATE_PUSH
VW_WARNING_DISABLE_CAST_FUNC_TYPE
template <class FluentBuilderT, class DataT, class ExampleT, class BaseLearnerT>
struct common_learner_builder
{
  learner<DataT, ExampleT>* _learner = nullptr;

  using end_fptr_type = void (*)(vw&, void*, void*);
  using finish_fptr_type = void (*)(void*);

  common_learner_builder(learner<DataT, ExampleT>* learner, std::unique_ptr<DataT>&& data, const std::string& name)
  {
    _learner = learner;
    _learner->name = name;
    _learner->is_multiline = std::is_same<multi_ex, ExampleT>::value;
    _learner->learner_data = std::shared_ptr<DataT>(data.release());
  }

  common_learner_builder(std::unique_ptr<DataT>&& data, const std::string& name)
      : common_learner_builder(new learner<DataT, ExampleT>(), std::move(data), name)
  {
  }

  FluentBuilderT& set_predict(void (*fn_ptr)(DataT&, BaseLearnerT&, ExampleT&))
  {
    this->_learner->learn_fd.predict_f = (learn_data::fn)fn_ptr;
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_learn(void (*fn_ptr)(DataT&, BaseLearnerT&, ExampleT&))
  {
    this->_learner->learn_fd.learn_f = (learn_data::fn)fn_ptr;
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_multipredict(
      void (*fn_ptr)(DataT&, BaseLearnerT&, ExampleT&, size_t, size_t, polyprediction*, bool))
  {
    this->_learner->learn_fd.multipredict_f = (learn_data::multi_fn)fn_ptr;
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_update(void (*u)(DataT& data, BaseLearnerT& base, ExampleT&))
  {
    this->_learner->learn_fd.update_f = (learn_data::fn)u;
    return *static_cast<FluentBuilderT*>(this);
  }

  // used for active learning and confidence to determine how easily predictions are changed
  FluentBuilderT& set_sensitivity(float (*fn_ptr)(DataT& data, base_learner& base, example&))
  {
    this->_learner->sensitivity_fd.data = this->_learner->learn_fd.data;
    this->_learner->sensitivity_fd.sensitivity_f = (sensitivity_data::fn)fn_ptr;

    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_learn_returns_prediction(bool learn_returns_prediction)
  {
    _learner->learn_returns_prediction = learn_returns_prediction;
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_save_load(void (*fn_ptr)(DataT&, io_buf&, bool, bool))
  {
    _learner->save_load_fd.save_load_f = (save_load_data::fn)fn_ptr;
    _learner->save_load_fd.data = _learner->learn_fd.data;
    _learner->save_load_fd.base = _learner->learn_fd.base;
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_finish(void (*fn_ptr)(DataT&))
  {
    _learner->finisher_fd = tuple_dbf(_learner->learn_fd.data, _learner->learn_fd.base, (finish_fptr_type)(fn_ptr));
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_end_pass(void (*fn_ptr)(DataT&))
  {
    _learner->end_pass_fd = tuple_dbf(_learner->learn_fd.data, _learner->learn_fd.base, (func_data::fn)fn_ptr);
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_end_examples(void (*fn_ptr)(DataT&))
  {
    _learner->end_examples_fd = tuple_dbf(_learner->learn_fd.data, _learner->learn_fd.base, (func_data::fn)fn_ptr);
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_init_driver(void (*fn_ptr)(DataT&))
  {
    _learner->init_fd = tuple_dbf(_learner->learn_fd.data, _learner->learn_fd.base, (func_data::fn)fn_ptr);
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_finish_example(void (*fn_ptr)(vw& all, DataT&, ExampleT&))
  {
    _learner->finish_example_fd.data = _learner->learn_fd.data;
    _learner->finish_example_fd.finish_example_f = (end_fptr_type)(fn_ptr);
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_print_example(void (*fn_ptr)(vw& all, DataT&, ExampleT&))
  {
    _learner->finish_example_fd.data = _learner->learn_fd.data;
    _learner->finish_example_fd.print_example_f = (end_fptr_type)(fn_ptr);
    return *static_cast<FluentBuilderT*>(this);
  }

  FluentBuilderT& set_persist_metrics(void (*fn_ptr)(DataT&, metric_sink&))
  {
    _learner->persist_metrics_fd.save_metric_f = (save_metric_data::fn)fn_ptr;
    _learner->persist_metrics_fd.data = _learner->learn_fd.data;
    _learner->persist_metrics_fd.base = _learner->learn_fd.base;
    return *static_cast<FluentBuilderT*>(this);
  }
};

template <class DataT, class ExampleT, class BaseLearnerT>
struct reduction_learner_builder
    : public common_learner_builder<reduction_learner_builder<DataT, ExampleT, BaseLearnerT>, DataT, ExampleT,
          BaseLearnerT>
{
  reduction_learner_builder(std::unique_ptr<DataT>&& data, BaseLearnerT* base, const std::string& name)
      // NOTE: This is a copy of the base! The purpose is to copy all of the
      // function data objects so that if this reduction does not define a function such as
      // save_load then calling save_load on this object will essentially result in forwarding the
      // call the next reduction that actually implements it.
      : common_learner_builder<reduction_learner_builder<DataT, ExampleT, BaseLearnerT>, DataT, ExampleT, BaseLearnerT>(
            new learner<DataT, ExampleT>(*reinterpret_cast<learner<DataT, ExampleT>*>(base)), std::move(data), name)
  {
    this->_learner->learn_fd.base = make_base(*base);
    this->_learner->learn_fd.data = this->_learner->learner_data.get();
    this->_learner->sensitivity_fd.sensitivity_f = static_cast<sensitivity_data::fn>(recur_sensitivity);
    this->_learner->finisher_fd.data = this->_learner->learner_data.get();
    this->_learner->finisher_fd.base = make_base(*base);
    this->_learner->finisher_fd.func = static_cast<func_data::fn>(noop);

    set_params_per_weight(1);

    this->_learner->pred_type = base->pred_type;
    // TODO add label type as something learner knows about itself, this will enable more type checking and better
    // description of the learner. this->_learner.label_type = label_type;
  }

  reduction_learner_builder<DataT, ExampleT, BaseLearnerT>& set_prediction_type(prediction_type_t pred_type)
  {
    this->_learner->pred_type = pred_type;
    return *this;
  }

  reduction_learner_builder<DataT, ExampleT, BaseLearnerT>& set_label_type(label_type_t label_type)
  {
    // TODO add label type as something learner knows about itself, this will enable more type checking and better
    // description of the learner. this->_learner.label_type = label_type;
    std::ignore = label_type;
    return *this;
  }

  reduction_learner_builder<DataT, ExampleT, BaseLearnerT>& set_params_per_weight(size_t params_per_weight)
  {
    this->_learner->weights = params_per_weight;
    this->_learner->increment = this->_learner->learn_fd.base->increment * this->_learner->weights;
    return *this;
  }

  learner<DataT, ExampleT>* build() { return this->_learner; }
};

template <class ExampleT, class BaseLearnerT>
struct reduction_no_data_learner_builder
    : public common_learner_builder<reduction_learner_builder<char, ExampleT, BaseLearnerT>, char, ExampleT,
          BaseLearnerT>
{
  reduction_no_data_learner_builder(BaseLearnerT* base, const std::string& name)
      // NOTE: This is a copy of the base! The purpose is to copy all of the
      // function data objects so that if this reduction does not define a function such as
      // save_load then calling save_load on this object will essentially result in forwarding the
      // call the next reduction that actually implements it.
      : common_learner_builder<reduction_learner_builder<char, ExampleT, BaseLearnerT>, char, ExampleT, BaseLearnerT>(
            new learner<char, ExampleT>(*reinterpret_cast<learner<char, ExampleT>*>(base)), nullptr, name)
  {
    this->_learner->learn_fd.base = make_base(*base);
    this->_learner->sensitivity_fd.sensitivity_f = static_cast<sensitivity_data::fn>(recur_sensitivity);
    this->_learner->finisher_fd.data = this->_learner->learner_data.get();
    this->_learner->finisher_fd.base = make_base(*base);
    this->_learner->finisher_fd.func = static_cast<func_data::fn>(noop);

    set_params_per_weight(1);

    this->_learner->pred_type = base->pred_type;
    // TODO add label type as something learner knows about itself, this will enable more type checking and better
    // description of the learner. this->_learner.label_type = label_type;
  }

  reduction_no_data_learner_builder<ExampleT, BaseLearnerT>& set_prediction_type(prediction_type_t pred_type)
  {
    this->_learner->pred_type = pred_type;
    return *this;
  }

  reduction_no_data_learner_builder<ExampleT, BaseLearnerT>& set_label_type(label_type_t label_type)
  {
    // TODO add label type as something learner knows about itself, this will enable more type checking and better
    // description of the learner. this->_learner.label_type = label_type;
    std::ignore = label_type;
    return *this;
  }

  reduction_no_data_learner_builder<ExampleT, BaseLearnerT>& set_params_per_weight(size_t params_per_weight)
  {
    this->_learner->weights = params_per_weight;
    this->_learner->increment = this->_learner->learn_fd.base->increment * this->_learner->weights;
    return *this;
  }

  learner<char, ExampleT>* build() { return this->_learner; }
};

inline float noop_sensitivity_base(void*, example&) { return 0.; }

template <class DataT, class ExampleT>
struct base_learner_builder
    : public common_learner_builder<base_learner_builder<DataT, ExampleT>, DataT, ExampleT, base_learner>
{
  base_learner_builder(
      std::unique_ptr<DataT>&& data, const std::string& name, prediction_type_t pred_type, label_type_t label_type)
      : common_learner_builder<base_learner_builder<DataT, ExampleT>, DataT, ExampleT, base_learner>(
            std::move(data), name)
  {
    this->_learner->end_pass_fd.func = static_cast<func_data::fn>(noop);
    this->_learner->end_examples_fd.func = static_cast<func_data::fn>(noop);
    this->_learner->init_fd.func = static_cast<func_data::fn>(noop);
    this->_learner->save_load_fd.save_load_f = static_cast<save_load_data::fn>(noop_save_load);
    this->_learner->finisher_fd.data = this->_learner->learner_data.get();
    this->_learner->finisher_fd.func = static_cast<func_data::fn>(noop);
    this->_learner->sensitivity_fd.sensitivity_f = reinterpret_cast<sensitivity_data::fn>(noop_sensitivity_base);
    this->_learner->finish_example_fd.data = this->_learner->learner_data.get();
    this->_learner->finish_example_fd.finish_example_f =
        reinterpret_cast<finish_example_data::fn>(return_simple_example);

    this->_learner->learn_fd.data = this->_learner->learner_data.get();
    this->_learner->pred_type = pred_type;
    // TODO add label type as something learner knows about itself.
    std::ignore = label_type;
    // this->_learner.label_type = label_type;

    set_params_per_weight(1);
  }

  base_learner_builder<DataT, ExampleT>& set_params_per_weight(size_t params_per_weight)
  {
    this->_learner->weights = 1;
    this->_learner->increment = params_per_weight;
    return *this;
  }

  learner<DataT, ExampleT>* build() { return this->_learner; }
};

VW_WARNING_STATE_POP

// NEW WAY:
// Use these two functions when creating a new learner.
template <class DataT, class ExampleT, class BaseLearnerT>
reduction_learner_builder<DataT, ExampleT, BaseLearnerT> make_reduction_learner(std::unique_ptr<DataT>&& data,
    BaseLearnerT* base, void (*learn_fn)(DataT&, BaseLearnerT&, ExampleT&),
    void (*predict_fn)(DataT&, BaseLearnerT&, ExampleT&), const std::string& name)
{
  auto builder = reduction_learner_builder<DataT, ExampleT, BaseLearnerT>(std::move(data), base, name);
  builder.set_learn(learn_fn);
  builder.set_update(learn_fn);
  builder.set_predict(predict_fn);
  return builder;
}

template <class ExampleT, class BaseLearnerT>
reduction_no_data_learner_builder<ExampleT, BaseLearnerT> make_no_data_reduction_learner(BaseLearnerT* base,
    void (*learn_fn)(char&, BaseLearnerT&, ExampleT&), void (*predict_fn)(char&, BaseLearnerT&, ExampleT&),
    const std::string& name)
{
  auto builder = reduction_no_data_learner_builder<ExampleT, BaseLearnerT>(base, name);
  builder.set_learn(learn_fn);
  builder.set_update(learn_fn);
  builder.set_predict(predict_fn);
  return builder;
}

template <class DataT, class ExampleT>
base_learner_builder<DataT, ExampleT> make_base_learner(std::unique_ptr<DataT>&& data,
    void (*learn_fn)(DataT&, base_learner&, ExampleT&), void (*predict_fn)(DataT&, base_learner&, ExampleT&),
    const std::string& name, prediction_type_t pred_type, label_type_t label_type)
{
  auto builder = base_learner_builder<DataT, ExampleT>(std::move(data), name, pred_type, label_type);
  builder.set_learn(learn_fn);
  builder.set_update(learn_fn);
  builder.set_predict(predict_fn);
  return builder;
}

}  // namespace LEARNER
}  // namespace VW