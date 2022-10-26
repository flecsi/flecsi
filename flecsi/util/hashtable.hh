// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_HASHTABLE_HH
#define FLECSI_UTIL_HASHTABLE_HH

#include "flecsi/flog.hh"
#include <flecsi/util/array_ref.hh>
#include <utility>

namespace flecsi {
namespace util {

template<class KEY, class TYPE, class HASH = std::hash<KEY>>
struct hashtable;

template<class KEY, class TYPE, class HASH>
class hashtableIterator
{
  using ht_t = hashtable<KEY, TYPE, HASH>;
  using ht_type_t = typename ht_t::pair_t;

private:
  ht_type_t * ptr_;
  const ht_t * h_;

  friend hashtable<KEY, TYPE, HASH>;
  hashtableIterator(ht_type_t * p, const ht_t * h) : ptr_(p), h_(h) {}

public:
  ht_type_t & operator*() const {
    return *ptr_;
  }

  hashtableIterator & operator++() {
    do {
      ++ptr_;
    } while(ptr_ < h_->end().ptr_ && ptr_->first == key_t{});
    return *this;
  }

  bool operator==(const hashtableIterator & iter) const {
    return this->ptr_ == iter.ptr_;
  }

  bool operator!=(const hashtableIterator & iter) const {
    return this->ptr_ != iter.ptr_;
  }

  constexpr ht_type_t * operator->() const {
    return ptr_;
  }
};

// hashtable implementation based on a \c util::span.
// This hashtable is based on span as 1D array.
// The hashtable is iterable.
template<class KEY, class TYPE, class HASH>
struct hashtable {

public:
  // Key use for search in the table
  using key_t = KEY;
  // Type stored with the key
  using type_t = TYPE;
  using pair_t = std::pair<key_t, type_t>;
  // Hasher
  using hash_f = HASH;

  // Iterator
  using pointer = pair_t *;
  using iterator = hashtableIterator<KEY, TYPE, HASH>;

private:
  std::size_t nelements_;
  constexpr static std::size_t modulo_ = 334214459;
  util::span<pair_t> span_;

  // Max number of search before crash
  constexpr static std::size_t max_find_ = 10;

public:
  hashtable(const util::span<pair_t> & span) {
    span_ = span;
    nelements_ = 0;
    for(auto it = this->begin(); it != this->end(); ++it) {
      ++nelements_;
    }
  }

  // Find a value in the hashtable
  // While the value or a null key is not found we keep looping
  iterator find(const key_t & key) {
    std::size_t h = hash_f()(key) % span_.size();
    pointer ptr = span_.data() + h;
    std::size_t iter = 0;
    while(ptr->first != key && ptr->first != key_t{} && iter != max_find_) {
      h = (h + modulo_) % span_.size();
      ptr = span_.data() + h;
      ++iter;
    }
    if(ptr->first != key) {
      return end();
    }
    return iterator(ptr, this);
  }

  // Insert an object in the hash map at a defined position
  // This function tries to find the first available position in case of
  // conflict using modulo method.
  template<typename... ARGS>
  iterator insert(const key_t & key, ARGS &&... args) {
    std::size_t h = hash_f()(key) % span_.size();
    pointer ptr = span_.data() + h;
    std::size_t iter = 0;
    while(ptr->first != key && ptr->first != key_t{} && iter != max_find_) {
      h = (h + modulo_) % span_.size();
      ptr = span_.data() + h;
      ++iter;
    }

    if(iter == max_find_) {
      flog(error) << "Max iteration reached, couldn't insert element: " << key
                  << std::endl;
      return end();
    }
    ++nelements_;
    ptr = new(ptr) pair_t(key, {std::forward<ARGS>(args)...});
    return iterator(ptr, this);
  }

  /**
   * @brief Return a reference to the object with corresponding key
   */
  type_t & at(const key_t & k) {
    auto f = find(k);
    if(f == end()) {
      throw std::out_of_range("Key out of range.");
    }
    return f->second;
  }

  /**
   * @brief Return a const reference to the object with corresponding key
   */
  const type_t & at(const key_t & k) const {
    return const_cast<hashtable &>(*this).at(k);
  }

  // Clear all keys frrom the table
  void clear() {
    nelements_ = 0;
    for(auto & a : *this) {
      a.first = key_t{};
    }
  }

  constexpr iterator begin() const noexcept {
    auto it = iterator(span_.begin(), this);
    if(it->first == key_t{})
      ++it;
    return it;
  }

  constexpr iterator end() const noexcept {
    return iterator(span_.end(), this);
  }

  // Number of elements currently stored in the hashtable
  constexpr std::size_t size() const noexcept {
    return nelements_;
  }

  // Check if the hashtable doesnt hold any non-null elements
  constexpr bool empty() const noexcept {
    return begin() == end();
  }

}; // class hashtable

/// \}
} // namespace util
} // namespace flecsi

#endif
