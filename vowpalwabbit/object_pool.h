// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include <set>
#include <queue>
#include <stack>

// Mutex and CV cannot be used in managed C++, tell the compiler that this is unmanaged even if included in a managed
// project.
#ifdef _M_CEE
#  pragma managed(push, off)
#  undef _M_CEE
#  include <mutex>
#  include <condition_variable>
#  define _M_CEE 001
#  pragma managed(pop)
#else
#  include <mutex>
#  include <condition_variable>
#endif

namespace VW
{

template <typename T>
struct no_lock_object_pool
{
  no_lock_object_pool() = default;
  no_lock_object_pool(size_t initial_chunk_size, size_t chunk_size = 8)
      : m_initial_chunk_size(initial_chunk_size), m_chunk_size(chunk_size)
  {
    new_chunk(initial_chunk_size);
  }

  ~no_lock_object_pool()
  {
    assert(m_pool.size() == size());
    for (auto* ptr : m_allocated_objects) { delete ptr; }
  }

  void return_object(T* obj)
  {
    assert(is_from_pool(obj));
    m_pool.push(obj);
  }

  T* get_object()
  {
    if (m_pool.empty()) { new_chunk(m_chunk_size); }

    auto obj = m_pool.front();
    m_pool.pop();
    return obj;
  }

  bool empty() const { return m_pool.empty(); }

  size_t size() const { return m_allocated_objects.size(); }

  bool is_from_pool(T* obj) const { return m_allocated_objects.count(obj) > 0; }

private:
  void new_chunk(size_t size)
  {
    if (size == 0) { return; }
    for (size_t i = 0; i < size; i++)
    {
      auto* new_obj = new T{};
      m_allocated_objects.insert(new_obj);
      m_pool.push(new_obj);
    }
  }

  size_t m_initial_chunk_size = 0;
  size_t m_chunk_size = 8;
  std::queue<T*> m_pool;
  std::set<T*> m_allocated_objects;
};

template <typename T>
struct moved_object_pool
{
  moved_object_pool() = default;

  void reclaim_object(T&& obj) { m_pool.push(std::move(obj)); }

  void acquire_object(T& dest)
  {
    if (m_pool.empty())
    {
      dest = T();
      return;
    }

    dest = std::move(m_pool.top());
    m_pool.pop();
  }

  T&& take_back()
  {
    if (m_pool.empty()) { m_pool.push(T{}); }
    return std::move(m_pool.top());
  }

  void pop_back()
  {
    assert(!m_pool.empty());
    m_pool.pop();
  }

  bool empty() const { return m_pool.empty(); }

  size_t size() const { return m_pool.size(); }

private:
  std::stack<T> m_pool;
};

template <typename T>
struct object_pool
{
  object_pool() = default;
  object_pool(size_t initial_chunk_size, size_t chunk_size = 8) : inner_pool(initial_chunk_size, chunk_size) {}

  void return_object(T* obj)
  {
    std::unique_lock<std::mutex> lock(m_lock);
    inner_pool.return_object(obj);
  }

  T* get_object()
  {
    std::unique_lock<std::mutex> lock(m_lock);
    return inner_pool.get_object();
  }

  bool empty() const
  {
    std::unique_lock<std::mutex> lock(m_lock);
    return inner_pool.empty();
  }

  size_t size() const
  {
    std::unique_lock<std::mutex> lock(m_lock);
    return inner_pool.size();
  }

  bool is_from_pool(T* obj) const
  {
    std::unique_lock<std::mutex> lock(m_lock);
    return inner_pool.is_from_pool(obj);
  }

private:
  mutable std::mutex m_lock;
  no_lock_object_pool<T> inner_pool;
};
}  // namespace VW
