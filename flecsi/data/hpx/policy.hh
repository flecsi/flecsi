// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_POLICY_HH
#define FLECSI_DATA_HPX_POLICY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/config.hh"
#include "flecsi/data/local/storage.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/types.hh"

#include <cstddef>
#include <cstring>
#include <list>
#include <set>
#include <string>
#include <vector>

namespace flecsi {
namespace data {

// Asynchronous operations on field data must complete before it is destroyed.
// Some additionally use communicators that become available for reuse upon
// their completion.

// A future that will be awaited unless released.
// Copies share a variable and thus can be populated together.
struct fate {
  using future = ::hpx::shared_future<void>;

  fate() = default; // use make for non-trivial cases
  fate(const fate &) = default; // note that destroying either will wait
  fate(fate &&) = default;
  ~fate() {
    if(f && f->valid() && !f->is_ready()) {
      f->wait();
      if(f->has_exception())
        flog_fatal("destroying exceptional future:\n" +
                   ::hpx::diagnostic_information(f->get_exception_ptr()));
    }
  }
  fate & operator=(const fate & r) & noexcept {
    if(&r != this)
      *this = fate(r);
    return *this;
  }
  fate & operator=(fate && r) & noexcept {
    fate(std::move(r)).f.swap(f);
    return *this;
  }

  void assign(future v) {
    flog_assert(!f->valid(), "replacing future");
    *f = std::move(v);
  }
  void send(future v) {
    assign(v);
    f.reset();
  }

  // These can be used even if this object is empty.
  explicit operator bool() const {
    return !!f;
  }
  future get() const {
    return get(f);
  }
  future release() {
    const auto p = std::move(f);
    return get(p);
  }

  static fate make(future f = {}) {
    fate ret;
    ret.f = std::make_shared<future>(std::move(f));
    return ret;
  }

private:
  std::shared_ptr<future> f;
  static future get(const decltype(f) & p) {
    return p ? *p : future();
  }
};

// Store unique (according to C) T objects in insertion order.
template<class T, auto & C>
struct ordered_set {
  using iterator = typename std::list<T>::iterator;

  [[nodiscard]] bool empty() const {
    return l.empty();
  }
  auto size() const {
    return l.size();
  }

  bool push(T t) {
    const bool ret = s.insert(key(t)).second;
    if(ret)
      l.push_back(std::move(t));
    return ret;
  }
  T & peek() {
    return l.back();
  }
  void pop() {
    s.erase(key(peek()));
    l.pop_back();
  }

  iterator begin() {
    return l.begin();
  }
  iterator end() {
    return l.end();
  }

  iterator erase(iterator i) {
    s.erase(key(*i));
    return l.erase(i);
  }

  void merge(ordered_set && o) {
    for(iterator i = o.l.begin(), e = o.l.end(); i != e;)
      if(s.count(key(*i)))
        i = o.l.erase(i);
      else
        ++i;
    l.splice(l.end(), std::move(o.l));
    auto m = std::move(o.s);
    s.merge(m);
  }

private:
  static auto key(const T & t) { // supply const
    return C(t);
  }

  std::list<T> l;
  std::set<std::decay_t<decltype(C(std::declval<const T &>()))>> s;
};

// Communicators are stored in a graph that summarizes task dependencies; they
// are moved to later, dependent nodes that use them or that are reachable
// from a superset of (current) root nodes.  Edges in the graph are
// aggressively contracted to keep the graph small.
struct comms {
  using ptr = std::shared_ptr<comms>;
  // Provide a stable address for asynchronous operations:
  using comm = std::unique_ptr<run::communicator>;

  void depend(ptr n) {
    memo m;
    if(n && n.get() != this && !absorb(n, m))
      past.push(std::move(n));
  }

  run::communicator & get() & {
    memo m;
    collapse(m);
    if(ours.empty()) {
      if(past.empty())
        ours.push_back(make_comm());
      else {
        // Take from a direct predecessor; deeper means more broadly useful.
        const ptr & p = past.peek();
        ours.push_back(std::move(p->ours.back()));
        p->ours.pop_back();
        if(absorb(p, m))
          past.pop();
      }
    }
    return *ours.front();
  }

  static ptr make() {
    return std::make_shared<comms>();
  }
  static comm make_comm() {
    return std::make_unique<run::communicator>(
      run::context::instance().world_comm());
  }

private:
  using memo = std::set<comms *>;
  static comms * key(const ptr & p) {
    return p.get();
  }

  void collapse(memo & m) {
    auto i = past.begin();
    for(auto n = past.size(); n--;)
      if(absorb(*i, m))
        i = past.erase(i);
      else
        ++i;
  }
  bool absorb(const ptr & c, memo & m) {
    // Precheck c to minimize std::set allocations for shallow graphs.
    // use_count is safe since we're just one thread here (no tasks).
    // c might get deleted; we assume that we can still compare to it.
    if(c.use_count() == 1 || (!c->past.empty() && m.insert(c.get()).second))
      c->collapse(m);
    if(c.use_count() == 1) {
      auto v = std::move(c->ours);
      if(v.size() > ours.size()) // make the smaller insertion
        ours.swap(v);
      ours.insert(
        ours.end(), std::move_iterator(v.begin()), std::move_iterator(v.end()));
      past.merge(std::move(c->past));
    }
    else if(c->ours.empty())
      for(const auto & p : c->past)
        past.push(p);
    else
      return false;
    return true;
  }

  std::vector<comm> ours; // elements never empty
  // Avoid duplicate parents without ordering by process-local addresses:
  ordered_set<ptr, key> past;
};

// To automatically construct the task graph, we must maintain a set of
// futures that constitute its frontier.  hold objects, several for each
// field, collectively contain those futures as well as the graph of
// associated communicators.
struct hold {
  explicit operator bool() const {
    return !!c;
  }

  void assign(fate::future x) {
    f.assign(std::move(x));
  }
  void send(fate::future x) {
    f.send(std::move(x));
    c.reset();
  }
  void wait() { // preserves communicators for future use
    f = {};
  }
  run::communicator & comm() { // created if needed
    return c->get();
  }
  fate::future depend(hold & h) & {
    c->depend(h.c);
    return h.f.get();
  }
  [[nodiscard]] fate::future depend(hold && h) & {
    c->depend(std::move(h.c));
    return h.f.release();
  }

  static hold make(fate::future f = {}) {
    hold ret;
    ret.c = comms::make();
    ret.f = fate::make(f);
    return ret;
  }

private:
  // Nodes are allocated for these together but destroyed separately.
  comms::ptr c; // destroyed only after waiting on users
  fate f;
};

struct backend_storage : local::storage {
  // Synchronize with all pending writes to this storage.
  void synchronize() {
    write.wait();
  }

  template<class F>
  void do_read(F && f) {
    read.push_back(std::forward<F>(f)(write));
  }
  template<class F>
  void do_write(F && f) {
    // We neglect the case of bad_alloc stranding a released future.
    if(write) {
      if(read.empty())
        read.push_back(std::move(write));
      else
        (void)read.front().depend(std::move(write)); // reads are newer
    }
    write = std::forward<F>(f)(std::move(read));
    // In case f calls do_read:
    for(auto & r : read)
      (void)write.depend(std::move(r));
    read.clear();
  }

private:
  // In practice, these never wait, because tasks keep the region_impl alive
  // and every ghost copy is followed by a task using the same regions.
  std::vector<hold> read; // since most recent write
  hold write;
};
} // namespace data
} // namespace flecsi

#include "flecsi/data/local/policy.hh"

#endif // FLECSI_DATA_HPX_POLICY_HH
