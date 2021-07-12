#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>

#include <algorithm>
#include <deque>
#include <list>
#include <memory>

#include <condition_variable>
#include <mutex>

#include <iostream>

#ifndef NDEBUG
#include <unordered_set>
#include "vw_exception.h"
#endif

namespace VW {
namespace buffer {

class bucket_buffer
{
 private:
  // Use v_array instead? The usage is different enough that it feels wrong though
  uint8_t* _buffer_begin = nullptr;
  uint8_t* _buffer_end = nullptr;
  size_t _num_buckets = 0;

  // size is [1, n+1]
  // Data range for each bucket is [_bucket_begin[i], _bucket_begin[i+1])
  std::deque<uint8_t*> _bucket_begin;

  unsigned int _lock_count = 0;
#ifndef NDEBUG    // debug only checks
  std::unordered_set<uint8_t*> _lock_set;
#endif
 private:
  size_t buffer_capacity() { return _buffer_end - _buffer_begin; }
  void resize(size_t sz)
  {
    size_t old_size = buffer_capacity();
    if(sz < old_size) {
      // noop if we try to shrink the buffer
      return;
    }
    size_t new_size = std::max((old_size*2) + 3, sz);
    // exponential buffer growth
    auto* orig_ptr = _buffer_begin;
    _buffer_begin = (uint8_t*)realloc(_buffer_begin, new_size);
    if(_buffer_begin == nullptr){
      // TODO: Error handling
      std::cerr << "RESIZE FAILED";
      abort();
      return;
    }

    // fix all the pointers if necessary
    if(orig_ptr != _buffer_begin)
    {
      size_t offset = 0;
      for(int i = 0; i < (int)(_bucket_begin.size())-1; ++i)
      {
	size_t len = _bucket_begin[i+1] - _bucket_begin[i];
	_bucket_begin[i] = _buffer_begin + offset;
	offset+= len;
      }
      _bucket_begin.back() = _buffer_begin + offset;
    }
    _buffer_end = _buffer_begin + new_size;
  }
  size_t remaining_buffer_len()
  {
    return _buffer_end - _bucket_begin.back();
  }
  
 public:
 bucket_buffer(size_t num)
   : _num_buckets(num)
     //, _locked(_num_buckets)
  {
    _bucket_begin.push_back(_buffer_begin);
    resize(_num_buckets);
  }
  ~bucket_buffer() {
    if(_buffer_begin) {
      free(_buffer_begin);
    }
  }
  void push_bucket(const uint8_t* buffer, size_t len)
  {
    
    // TODO: error handling?
    assert(_num_buckets < size());

    if(len == 0) {
      return;
    }
    if(remaining_buffer_len() < len)
    {
      resize(buffer_capacity() + len);
    }
    auto* new_bucket = _bucket_begin.back();
    memcpy(new_bucket, buffer, len);
    _bucket_begin.push_back(new_bucket + len);
  }

  void take_bucket(uint8_t*& buffer, size_t& len)
  {
    if(size() == 0)
    {
      // TODO: error handling?
      buffer = nullptr;
      len = 0;
      return;
    }
    auto begin = _bucket_begin.front();
    _bucket_begin.pop_front();
    ++_lock_count;
#ifndef NDEBUG
    // this is safe because we don't allow the insertion of 0 length buckets
    auto iter_pair = _lock_set.insert(begin);
    if(!iter_pair.second) {
      // Somehow taking the same pointer more than once. This should never happen
      THROW("Attemping to take a bucket twice");
    }
#endif
    
    buffer = begin;
    len = _bucket_begin.front() - begin;
  }
  void release_bucket(uint8_t*)
  {
    // TODO: error handling?
    assert(_lock_count > 0);
    --_lock_count;
#ifndef NDEBUG
    auto iter = _lock_set.find(buffer);
    if(iter != _lock_set.end()) {
      _lock_set.erase(iter);
    }
    else {
      THROW("Attemping to release a bucket that was not taken");
    }
#endif
  }
  bool finished() { return size() == 0 && lock_count() == 0; }
  void reset()
  {
    _lock_count = 0;
    _bucket_begin.clear();
    _bucket_begin.push_back(_buffer_begin);
  }
  size_t size() { return _bucket_begin.size() - 1; }
  size_t capacity() { return _num_buckets; }
  size_t lock_count() {
    // in debug mode, lock count is defined by the size of _lock_set. Otherwise use the counter.
#ifndef NDEBUG
    return _lock_set.size();
#else
    return _lock_count;
#endif
  }
  bool ready() { return size() == capacity(); }
};

class swap_buffer;
 
class buffer_data
{
  friend class swap_buffer;
 private:
  using buffer_list = std::list<bucket_buffer>;
  buffer_list::iterator _container;
  uint8_t* _data = nullptr;
  size_t _size = 0;
 public:
  uint8_t* data() { return _data; }
  size_t size() { return _size; }
};
  
class swap_buffer
{
 private:
  using buffer_list = std::list<bucket_buffer>;
  
  size_t _num_buffers = 0;
  buffer_list _write_buffers;
  buffer_list _pending_buffers;
  std::deque<buffer_list::iterator> _read_buffers;
  bool _finished = false;
  std::mutex _read_buffer_lock;
  std::mutex _write_buffer_lock;
  std::condition_variable _read_available;
  std::condition_variable _write_available;
  
 public:
  swap_buffer(size_t num_buffers, size_t num_buckets)
    : _num_buffers(num_buffers)
  {
    for(size_t i = 0; i < _num_buffers; ++i) {
      _write_buffers.emplace_back(num_buckets);
    }
  }

  void write_data(const uint8_t* buffer, size_t len)
  {
    //TODO: error handling
    assert(!_finished);

    {
      // !!! Assumption that there's only 1 writer !!!
      std::unique_lock<std::mutex> write_guard(_write_buffer_lock);
      _write_available.wait(write_guard, [this]{ return !_write_buffers.empty(); });
    }
    
    _write_buffers.front().push_bucket(buffer, len);
    if(_write_buffers.front().ready())
    {
      // NOTE: This works because splice() does not invalidate any iterators
      auto iter = _write_buffers.begin();

      std::unique_lock<std::mutex> read_guard(_read_buffer_lock);
      std::unique_lock<std::mutex> write_guard(_write_buffer_lock);
      _pending_buffers.splice(_pending_buffers.end(), _write_buffers, iter);
      _read_buffers.push_back(iter);
      _read_available.notify_all();
    }
  }
  buffer_data get_data()
  {
    std::unique_lock<std::mutex> read_guard(_read_buffer_lock);

    _read_available.wait(read_guard, [this]{ return !_read_buffers.empty() || _finished; });

    buffer_data ret;

    if(_finished && _read_buffers.empty()) {
      return ret;
    }
      
    buffer_list::iterator container_iter = _read_buffers.front();

    (container_iter)->take_bucket(ret._data, ret._size);
    ret._container = container_iter;

    if ((container_iter)->size() == 0) {
      _read_buffers.pop_front();
    }
    return ret;
  }

  void return_buffer(buffer_data& data)
  {
    std::unique_lock<std::mutex> read_guard(_read_buffer_lock);
    
    if(data._data == nullptr)
    {
      return;
    }

    data._container->release_bucket(data._data);
    if(data._container->finished())
    {
      std::unique_lock<std::mutex> write_guard(_write_buffer_lock);
      
      data._container->reset();
      _write_buffers.splice(_write_buffers.end(), _pending_buffers, data._container);
      _write_available.notify_all();
    }
  }

  void finalize_input()
  {
    std::unique_lock<std::mutex> read_guard(_read_buffer_lock);
    std::unique_lock<std::mutex> write_guard(_write_buffer_lock);
    _finished = true;
    if(!_write_buffers.empty() && _write_buffers.front().size() > 0)
    {
      auto iter = _write_buffers.begin();
      _pending_buffers.splice(_pending_buffers.end(), _write_buffers, iter);
      _read_buffers.push_back(iter);
    }
    _read_available.notify_all();
  }
  size_t size()
  {
    size_t ret = 0;
    std::unique_lock<std::mutex> read_guard(_read_buffer_lock);
    for(const auto& buffer_iter : _read_buffers)
    {
      ret += (buffer_iter)->size();
    }
    return ret;
  }
};
  
}
}
