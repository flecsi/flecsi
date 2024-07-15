// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_GPU_CONTAINERS_HH
#define FLECSI_UTIL_GPU_CONTAINERS_HH

#include <cstddef>

namespace flecsi::util {
/// \cond core

/// \addtogroup GPU_utils GPU utilities
/// Containers with fixed size to use on GPUs

/// A simple queue implementation based on a std::array
/// \gpu.
template<typename T, std::size_t SIZE>
class queue
{
  using value_type = T;
  std::array<value_type, SIZE> data;
  size_t head = 0, back = 0;

public:
  /// Push an element in the queue
  /// \param v Element to be inserted
  constexpr void push(const value_type & v) {
    assert((back + 1) % SIZE != head && "Queue is full");
    data[back] = v;
    back = (back + 1) % SIZE;
  }
  /// Return the first element in the queue
  constexpr const value_type & front() const {
    assert(!empty() && "Queue is empty");
    return data[head];
  }
  /// Remove the first element in the queue and return it
  constexpr value_type pop() {
    assert(!empty() && "Queue is empty");
    const value_type & ret = data[head];
    head = (head + 1) % SIZE;
    return ret;
  }
  /// Check if the queue is empty
  constexpr bool empty() const {
    return back == head;
  }
};

/// A small implementation of std::inplace_vector as proposed for C++26.
/// \gpu.
template<typename T, std::size_t SIZE>
class inplace_vector
{
  using value_type = T;
  std::array<value_type, SIZE> data_;
  std::size_t size_ = 0;

public:
  /// Add an element at the end of a vector
  /// \param v Element to be inserted
  constexpr void push_back(const value_type & v) {
    assert(SIZE != size_ && "Vector is full");
    data_[size_++] = v;
  }
  /// Return the element i of the vector
  /// \param i The position of the element to return
  constexpr const value_type & operator[](int i) const {
    return data_[i];
  }
  /// Return a pointer to the underlaying data
  constexpr auto data() {
    return data_.data();
  }
  /// Return the size of the used portion of the vector
  constexpr std::size_t size() const {
    return size_;
  }
  /// Check if the vector is empty
  constexpr bool empty() const {
    return size_ == 0;
  }
  /// Return a pointer to the first element of the vector
  constexpr value_type * begin() {
    return data();
  }
  /// Return a pointer past the last element of the vector
  constexpr value_type * end() {
    return data() + size_;
  }
};

} // namespace flecsi::util

/// \endcond

#endif
