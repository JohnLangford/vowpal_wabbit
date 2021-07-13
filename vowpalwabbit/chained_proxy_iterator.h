// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <iterator>
#include <cstddef>

namespace VW
{
// This is a bit non-idiomatic but this class's value type is the iterator itself in order to expose
// any custom fields that the inner iterator type may expose.
// This isn't exactly generic either since it uses audit_begin() directly
//
// Structure of begin and end iterators:
// Begin:
//    [a, b] , [c, d]
//    ^begin_outer
//     ^begin_inner
// End:
//    [a, b] , [c, d, past_end_of_list_end_iterator]
//             ^end_outer
//                    ^end_inner
template <typename InnerIterator, typename IteratorT>
struct chained_proxy_iterator
{
private:
  InnerIterator _outer_current;
  InnerIterator _outer_end;
  IteratorT _current;

public:
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = IteratorT;
  using reference = value_type&;
  using const_reference = const value_type&;

  chained_proxy_iterator(InnerIterator outer_current, InnerIterator outer_end, IteratorT current)
      : _outer_current(outer_current), _outer_end(outer_end), _current(current)
  {
  }

  chained_proxy_iterator(const chained_proxy_iterator&) = default;
  chained_proxy_iterator& operator=(const chained_proxy_iterator&) = default;
  chained_proxy_iterator(chained_proxy_iterator&&) = default;
  chained_proxy_iterator& operator=(chained_proxy_iterator&&) = default;

  inline reference operator*() { return *_current; }
  inline const_reference operator*() const { return *_current; }

  chained_proxy_iterator& operator++()
  {
    ++_current;
    // TODO: don't rely on audit_end
    if (_current == (*_outer_current).feats.audit_end() && (_outer_current != _outer_end))
    {
      ++_outer_current;
      _current = (*_outer_current).feats.audit_begin();
    }
    return *this;
  }

  // UB if diff < 0
  chained_proxy_iterator& operator+=(difference_type diff)
  {
    assert(diff >= 0);
    while (true)
    {
      auto current_group_distance_to_end = std::distance(_current, (*_outer_current).feats.audit_end());
      if (diff > current_group_distance_to_end)
      {
        diff -= current_group_distance_to_end;
        ++_outer_current;
        _current = (*_outer_current).feats.audit_begin();
      }
      else
      {
        _current += diff;

        if ((diff == current_group_distance_to_end) && (_outer_current != _outer_end))
        {
          ++_outer_current;
          _current = (*_outer_current).feats.audit_begin();
        }
        return *this;
      }
    }
  }

  // end - begin
  friend difference_type operator-(const chained_proxy_iterator& lhs, chained_proxy_iterator rhs)
  {
    // assert(lhs._outer_current >= rhs._outer_current);
    size_t accumulator = 0;
    while (lhs._outer_current != rhs._outer_current)
    {
      accumulator += std::distance(rhs._current, (*(rhs._outer_current)).feats.audit_end());
      ++rhs._outer_current;
      rhs._current = (*rhs._outer_current).feats.audit_begin();
    }
    accumulator += std::distance(rhs._current, lhs._current);
    return accumulator;
  }

  friend bool operator==(const chained_proxy_iterator& lhs, const chained_proxy_iterator& rhs)
  {
    return (lhs._outer_current == rhs._outer_current) && (lhs._current == rhs._current);
  }

  friend bool operator!=(const chained_proxy_iterator& lhs, const chained_proxy_iterator& rhs) { return !(lhs == rhs); }
};
}  // namespace VW
