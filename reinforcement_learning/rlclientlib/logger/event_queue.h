#pragma once

#include "ranking_event.h"

#include <list>
#include <queue>
#include <mutex>
#include <type_traits>

namespace reinforcement_learning {

  //a moving concurrent queue with locks and mutex
  template <class T>
  class event_queue {
    using queue_t = std::list<T>;

    queue_t _queue;
    std::mutex _mutex;
    int _drop_pass{ 0 };
    size_t _capacity{ 0 };

  public:
    event_queue() {
      static_assert(std::is_base_of<event, T>::value, "T must be a descendant of event");
    }

    void pop(T* item)
    {
      std::unique_lock<std::mutex> mlock(_mutex);
      if (!_queue.empty())
      {
        *item = std::move(_queue.front());
        _capacity = (std::max)((int)0, (int)_capacity - (int)(item->size()));
        _queue.pop_front();
      }
    }

    void push(T& item) {
      push(std::move(item));
    }

    void push(T&& item)
    {
      std::unique_lock<std::mutex> mlock(_mutex);
      _capacity += item.size();
      _queue.push_back(std::forward<T>(item));
    }

    void prune(float pass_prob)
    {
      std::unique_lock<std::mutex> mlock(_mutex);
      for (typename queue_t::iterator it = _queue.begin(); it != _queue.end();) {
        it = it->try_drop(pass_prob, _drop_pass) ? erase(it) : (++it);
      }
      ++_drop_pass;
    }

    //approximate size
    size_t size()
    {
      std::unique_lock<std::mutex> mlock(_mutex);
      return _queue.size();
    }

    size_t capacity() const 
    {
      return _capacity;
    }

  private:
    //thread-unsafe
    typename queue_t::iterator erase(typename queue_t::iterator it) {
      _capacity = (std::max)((int)0, (int)_capacity - (int)(it->size()));
      return _queue.erase(it);
    }
  };
}

