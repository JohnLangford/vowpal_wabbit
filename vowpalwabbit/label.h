#pragma once

/*
When a new label type needs to be added the following actions must be taken:
- LABEL_TYPE is the type that will be used
- LABEL_NAME is the name to identify this label type
Steps:
  1. Add a new variant to label_type_t called LABEL_NAME
  2. Add the corresponding row to to_string:
    TO_STRING_CASE(label_type_t::LABEL_NAME)
  3. Add the new type to the union:
    LABEL_TYPE _LABEL_NAME;
  3. Add the corresponding row to polylabel::copy_from
    case (label_type_t::LABEL_NAME):
      init_as_LABEL_NAME(std::move(other._LABEL_NAME));
      break;
  4. Add the corresponding row to polylabel::move_from
    case (label_type_t::LABEL_NAME):
      init_as_LABEL_NAME(std::move(other._LABEL_NAME));
      break;
  5. Add the corresponding row to polylabel::reset
    case (label_type_t::LABEL_NAME):
        destruct(_LABEL_NAME);
        break;
  6. Add another three methods that correspond to the new type according to this template
    template <typename... Args>
    LABEL_TYPE& init_as_LABEL_NAME(Args&&... args)
    {
      ensure_is_type(label_type_t::unset);
      new (&_LABEL_NAME) LABEL_TYPE(std::forward<Args>(args)...);
      _tag = label_type_t::LABEL_NAME;
      return _LABEL_NAME;
    }

    const LABEL_TYPE& LABEL_NAME() const
    {
      ensure_is_type(label_type_t::LABEL_NAME);
      return _LABEL_NAME;
    }

    LABEL_TYPE& LABEL_NAME()
    {
      ensure_is_type(label_type_t::LABEL_NAME);
      return _LABEL_NAME;
    }
*/

#include "no_label.h"
#include "simple_label.h"
#include "multiclass.h"
#include "multilabel.h"
#include "cost_sensitive.h"
#include "cb.h"
#include "example_predict.h"
#include "ccb_label.h"

#define TO_STRING_CASE(enum_type) \
  case enum_type:                 \
    return #enum_type;

enum class label_type_t
{
  unset,
  empty,
  simple,
  multi,
  cs,
  cb,
  conditional_contextual_bandit,
  cb_eval,
  multilabels
};

inline const char* to_string(label_type_t label_type)
{
  switch (label_type)
  {
    TO_STRING_CASE(label_type_t::unset)
    TO_STRING_CASE(label_type_t::empty)
    TO_STRING_CASE(label_type_t::simple)
    TO_STRING_CASE(label_type_t::multi)
    TO_STRING_CASE(label_type_t::cs)
    TO_STRING_CASE(label_type_t::cb)
    TO_STRING_CASE(label_type_t::conditional_contextual_bandit)
    TO_STRING_CASE(label_type_t::cb_eval)
    TO_STRING_CASE(label_type_t::multilabels)
    default:
      return "<unsupported>";
  }
}

struct polylabel
{
 private:
  union {
    no_label::no_label _empty;
    label_data _simple;
    MULTICLASS::label_t _multi;
    COST_SENSITIVE::label _cs;
    CB::label _cb;
    CCB::label _conditional_contextual_bandit;
    CB_EVAL::label _cb_eval;
    MULTILABEL::labels _multilabels;
  };
  label_type_t _tag;

  inline void ensure_is_type(label_type_t type) const
  {
#ifndef NDEBUG
    if (_tag != type)
    {
      THROW("Expected type: " << to_string(type) << ", but found: " << to_string(_tag));
    }
#else
    _UNUSED(type);
#endif
  }

  template <typename T>
  void destruct(T& item)
  {
    item.~T();
  }

  // These two functions only differ by parameter
  void copy_from(const polylabel& other)
  {
    switch (other._tag)
    {
      case (label_type_t::unset):
        break;
      case (label_type_t::empty):
        init_as_empty(other._empty);
        break;
      case (label_type_t::simple):
        init_as_simple(other._simple);
        break;
      case (label_type_t::multi):
        init_as_multi(other._multi);
        break;
      case (label_type_t::cs):
        init_as_cs(other._cs);
        break;
      case (label_type_t::cb):
        init_as_cb(other._cb);
        break;
      case (label_type_t::conditional_contextual_bandit):
        init_as_conditional_contextual_bandit(other._conditional_contextual_bandit);
        break;
      case (label_type_t::cb_eval):
        init_as_cb_eval(other._cb_eval);
        break;
      case (label_type_t::multilabels):
        init_as_multilabels(other._multilabels);
        break;
      default:;
    }
  }

  void move_from(polylabel&& other)
  {
    switch (other._tag)
    {
      case (label_type_t::unset):
        break;
      case (label_type_t::empty):
        init_as_empty(std::move(other._empty));
        break;
      case (label_type_t::simple):
        init_as_simple(std::move(other._simple));
        break;
      case (label_type_t::multi):
        init_as_multi(std::move(other._multi));
        break;
      case (label_type_t::cs):
        init_as_cs(std::move(other._cs));
        break;
      case (label_type_t::cb):
        init_as_cb(std::move(other._cb));
        break;
      case (label_type_t::conditional_contextual_bandit):
        init_as_conditional_contextual_bandit(std::move(other._conditional_contextual_bandit));
        break;
      case (label_type_t::cb_eval):
        init_as_cb_eval(std::move(other._cb_eval));
        break;
      case (label_type_t::multilabels):
        init_as_multilabels(std::move(other._multilabels));
        break;
      default:;
    }
  }

 public:
  polylabel() { _tag = label_type_t::unset; // Perhaps we should memset here?
  };
  ~polylabel() { reset(); }

  polylabel(polylabel&& other)
  {
    _tag = label_type_t::unset;
    move_from(std::move(other));
  }

  polylabel& operator=(polylabel&& other)
  {
    reset();
    move_from(std::move(other));
    return *this;
  }

  polylabel(const polylabel& other) {
    _tag = label_type_t::unset;
    copy_from(other);
  }

  polylabel& operator=(const polylabel& other) {
    reset();
    copy_from(other);
    return *this;
  }

  label_type_t get_type() const { return _tag; }

  void reset()
  {
    switch (_tag)
    {
      case (label_type_t::unset):
        // Nothing to do! Whatever was in here has already been destroyed.
        break;
      case (label_type_t::empty):
        destruct(_empty);
        break;
      case (label_type_t::simple):
        destruct(_simple);
        break;
      case (label_type_t::multi):
        destruct(_multi);
        break;
      case (label_type_t::cs):
        destruct(_cs);
        break;
      case (label_type_t::cb):
        destruct(_cb);
        break;
      case (label_type_t::conditional_contextual_bandit):
        destruct(_conditional_contextual_bandit);
        break;
      case (label_type_t::cb_eval):
        destruct(_cb_eval);
        break;
      case (label_type_t::multilabels):
        destruct(_multilabels);
        break;
      default:;
    }

    _tag = label_type_t::unset;
  }

  template <typename... Args>
  no_label::no_label& init_as_empty(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_empty) no_label::no_label(std::forward<Args>(args)...);
    _tag = label_type_t::empty;
    return _empty;
  }

  const no_label::no_label& empty() const
  {
    ensure_is_type(label_type_t::empty);
    return _empty;
  }

  no_label::no_label& empty()
  {
    ensure_is_type(label_type_t::empty);
    return _empty;
  }

  template <typename... Args>
  label_data& init_as_simple(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_simple) label_data(std::forward<Args>(args)...);
    _tag = label_type_t::simple;
    return _simple;
  }

  const label_data& simple() const
  {
    ensure_is_type(label_type_t::simple);
    return _simple;
  }

  label_data& simple()
  {
    ensure_is_type(label_type_t::simple);
    return _simple;
  }

  template <typename... Args>
  MULTICLASS::label_t& init_as_multi(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_multi) MULTICLASS::label_t(std::forward<Args>(args)...);
    _tag = label_type_t::multi;
    return _multi;
  }

  const MULTICLASS::label_t& multi() const
  {
    ensure_is_type(label_type_t::multi);
    return _multi;
  }

  MULTICLASS::label_t& multi()
  {
    ensure_is_type(label_type_t::multi);
    return _multi;
  }

  template <typename... Args>
  COST_SENSITIVE::label& init_as_cs(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_cs) COST_SENSITIVE::label(std::forward<Args>(args)...);
    _tag = label_type_t::cs;
    return _cs;
  }

  const COST_SENSITIVE::label& cs() const
  {
    ensure_is_type(label_type_t::cs);
    return _cs;
  }

  COST_SENSITIVE::label& cs()
  {
    ensure_is_type(label_type_t::cs);
    return _cs;
  }

  template <typename... Args>
  CB::label& init_as_cb(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_cb) CB::label(std::forward<Args>(args)...);
    _tag = label_type_t::cb;
    return _cb;
  }
  const CB::label& cb() const
  {
    ensure_is_type(label_type_t::cb);
    return _cb;
  }

  CB::label& cb()
  {
    ensure_is_type(label_type_t::cb);
    return _cb;
  }

  template <typename... Args>
  CCB::label& init_as_conditional_contextual_bandit(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_conditional_contextual_bandit) CCB::label(std::forward<Args>(args)...);
    _tag = label_type_t::conditional_contextual_bandit;
    return _conditional_contextual_bandit;
  }

  const CCB::label& ccb() const
  {
    ensure_is_type(label_type_t::conditional_contextual_bandit);
    return _conditional_contextual_bandit;
  }

  CCB::label& ccb()
  {
    ensure_is_type(label_type_t::conditional_contextual_bandit);
    return _conditional_contextual_bandit;
  }

  template <typename... Args>
  CB_EVAL::label& init_as_cb_eval(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_cb_eval) CB_EVAL::label(std::forward<Args>(args)...);
    _tag = label_type_t::cb_eval;
    return _cb_eval;
  }

  const CB_EVAL::label& cb_eval() const
  {
    ensure_is_type(label_type_t::cb_eval);
    return _cb_eval;
  }

  CB_EVAL::label& cb_eval()
  {
    ensure_is_type(label_type_t::cb_eval);
    return _cb_eval;
  }

  template <typename... Args>
  MULTILABEL::labels& init_as_multilabels(Args&&... args)
  {
    ensure_is_type(label_type_t::unset);
    new (&_multilabels) MULTILABEL::labels(std::forward<Args>(args)...);
    _tag = label_type_t::multilabels;
    return _multilabels;
  }

  const MULTILABEL::labels& multilabels() const
  {
    ensure_is_type(label_type_t::multilabels);
    return _multilabels;
  }

  MULTILABEL::labels& multilabels()
  {
    ensure_is_type(label_type_t::multilabels);
    return _multilabels;
  }
};
