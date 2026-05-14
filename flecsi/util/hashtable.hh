// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_HASHTABLE_HH
#define FLECSI_UTIL_HASHTABLE_HH

#include "flecsi/flog.hh"
#include <flecsi/util/array_ref.hh>

#include <span>
#include <utility>

namespace flecsi {
namespace util {

template<typename KEY>
struct default_hash {
  static std::size_t hash(const KEY & k) {
    return std::hash<KEY>{}(k);
  }
};

template<class KEY, class TYPE, class HASH = default_hash<KEY>>
struct hashtable;

template<class KEY, class TYPE, class HASH>
class hashtableIterator {
  using ht_t = hashtable<KEY, TYPE, HASH>;
  using ht_type_t = typename ht_t::pair_t;
  using key_t = KEY;

private:
  ht_type_t * ptr_;
  const ht_t * h_;

  friend hashtable<KEY, TYPE, HASH>;
  constexpr hashtableIterator(ht_type_t * p, const ht_t * h) : ptr_(p), h_(h) {}

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

  constexpr bool operator==(const hashtableIterator & iter) const {
    return this->ptr_ == iter.ptr_;
  }

  constexpr bool operator!=(const hashtableIterator & iter) const {
    return this->ptr_ != iter.ptr_;
  }

  constexpr ht_type_t * operator->() const {
    return ptr_;
  }
};

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

  // Iterator
  using pointer = pair_t *;
  using iterator = hashtableIterator<KEY, TYPE, HASH>;

private:
  constexpr static std::size_t modulo_ = 334214459;
  std::span<pair_t> span_;

  constexpr pointer lookup(const key_t & k) const {
    std:size_t = HASH::hash(k) % span_.size();
    for(unsigned ttl = 10; ttl--;) { // max number of search before crash
      const pointer p = span_.data() + h;
      if(p->first == k || p->first == key_t())
        return p;
      h = (h + modulo_) % span_.size();
    }
    assert(!"hash table full");
    return nullptr;
  }

public:
  constexpr hashtable(std::span<pair_t> span) : span_(span) {}

  // Find a value in the hashtable
  // While the value or a null key is not found we keep looping
  constexpr iterator find(const key_t & key) const {
    const pointer ptr = lookup(key);
    if(ptr->first != key) {
      return end();
    }
    return iterator(ptr, this);
  }

  // Insert an object in the hash map at a defined position
  // This function tries to find the first available position in case of
  // conflict using modulo method.
  template<typename... ARGS>
  iterator insert(const key_t & key, ARGS &&... args) const {
    pointer ptr = lookup(key);
    ptr = new(ptr) pair_t(key, {std::forward<ARGS>(args)...});
    return iterator(ptr, this);
  }

  /**
   * @brief Return a reference to the object with corresponding key
   */
  constexpr type_t & at(const key_t & k) {
    auto f = find(k);
    assert(f != end() && "Key out of range.");
    return f->second;
  }

  /**
   * @brief Return a const reference to the object with corresponding key
   */
  constexpr const type_t & at(const key_t & k) const {
    return const_cast<hashtable &>(*this).at(k);
  }

  // Clear all keys frrom the table
  void clear() {
    const auto v = std::make_pair(key_t{}, type_t{});
    std::fill(span_.begin(), span_.end(), v);
  }

  constexpr iterator begin() const noexcept {
    auto it = iterator(span_.data(), this);
    if(it->first == key_t{})
      ++it;
    return it;
  }

  constexpr iterator end() const noexcept {
    return iterator(std::to_address(span_.end()), this);
  }

  // Number of elements currently stored in the hashtable
  // This computation is linear in time and should be used for debug only
  constexpr std::size_t count_entries() const noexcept {
    return std::distance(this->begin(), this->end());
  }

  // Check if the hashtable doesnt hold any non-null elements
  [[nodiscard]] constexpr bool empty() const noexcept {
    return begin() == end();
  }

}; // class hashtable

} // namespace util
} // namespace flecsi

#endif
