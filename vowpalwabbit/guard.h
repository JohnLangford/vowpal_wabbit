// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.
#pragma once

#include <utility>
#include "future_compat.h"

// While it is a general concept this specific implemenation of scope based cleanup was inspired by
// https://github.com/microsoft/wil/blob/cd51fa9d3c24e9e9f5d4cdf1a9768d68f441fc48/include/wil/resource.h#L543

namespace VW
{
namespace details
{
template <typename T>
class swap_guard_impl
{
 public:
  swap_guard_impl(T& original_location, T& value_to_swap) noexcept : _original_location(original_location), _value_to_swap(value_to_swap)
  {
    std::swap(_original_location, _value_to_swap);
  }

  swap_guard_impl(const swap_guard_impl&) = delete;
  swap_guard_impl& operator=(const swap_guard_impl&) = delete;

  swap_guard_impl(swap_guard_impl&& other) noexcept
      : _original_location(other._original_location), _value_to_swap(other._value_to_swap), _will_swap_back(other._will_swap_back)
  {
    other._will_swap_back = false;
  }

  swap_guard_impl& operator=(swap_guard_impl&& other) noexcept
  {
    if (this == &other)
    {
      return *this;
    }

    _original_location = other._original_location;
    _value_to_swap = other._value_to_swap;
    _will_swap_back = other._will_swap_back;
    other._will_swap_back = false;
    return *this;
  }

  ~swap_guard_impl() noexcept { force_swap(); }

  void cancel() noexcept { _will_swap_back = false; }
  void force_swap() noexcept
  {
    if (_will_swap_back == true)
    {
      std::swap(_original_location, _value_to_swap);
      _will_swap_back = false;
    }
  }

 private:
  T& _original_location;
  T& _value_to_swap;
  bool _will_swap_back = true;
};


template <typename T>
class swap_guard_impl_rvalue
{
 public:
  swap_guard_impl_rvalue(T& original_location, T&& value_to_swap) noexcept : _original_location(original_location), _value_to_swap(std::move(value_to_swap))
  {
    std::swap(_original_location, _value_to_swap);
  }

  swap_guard_impl_rvalue(const swap_guard_impl_rvalue&) = delete;
  swap_guard_impl_rvalue& operator=(const swap_guard_impl_rvalue&) = delete;

  swap_guard_impl_rvalue(swap_guard_impl_rvalue&& other) noexcept
      : _original_location(other._original_location), _value_to_swap(std::move(other._value_to_swap)), _will_swap_back(other._will_swap_back)
  {
    other._will_swap_back = false;
  }

  swap_guard_impl_rvalue& operator=(swap_guard_impl_rvalue&& other) noexcept
  {
    if (this == &other)
    {
      return *this;
    }

    _original_location = other._original_location;
    _value_to_swap = std::move(other._value_to_swap);
    _will_swap_back = other._will_swap_back;
    other._will_swap_back = false;
    return *this;
  }

  ~swap_guard_impl_rvalue() noexcept { force_swap(); }

  void cancel() noexcept { _will_swap_back = false; }
  void force_swap() noexcept
  {
    if (_will_swap_back == true)
    {
      std::swap(_original_location, _value_to_swap);
      _will_swap_back = false;
    }
  }

 private:
  T& _original_location;
  T _value_to_swap;
  bool _will_swap_back = true;
};

}  // namespace details

/// This guard will swap the two locations on creation and upon deletion swap them back.
template <typename T>
VW_ATTR(nodiscard)
inline details::swap_guard_impl<T> swap_guard(T& original_location, T& value_to_swap) noexcept
{
  return details::swap_guard_impl<T>(original_location, value_to_swap);
}

/// This guard will swap the two locations on creation and upon deletion swap them back.
/// Note: This overload allows for a temporary value to be passed in.
template <typename T>
VW_ATTR(nodiscard)
inline details::swap_guard_impl_rvalue<T> swap_guard(T& original_location, T&& value_to_swap) noexcept
{
  return details::swap_guard_impl_rvalue<T>(original_location, std::forward<T>(value_to_swap));
}

}  // namespace VW
