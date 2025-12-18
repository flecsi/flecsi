// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_POLICY_HH
#define FLECSI_DATA_HPX_POLICY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/data/local/storage.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/types.hh"

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

  static hold make(run::comms::ptr c = run::comms::make()) {
    hold ret;
    ret.c = std::move(c);
    ret.f = fate::make();
    return ret;
  }

private:
  // Nodes are allocated for these together but destroyed separately.
  run::comms::ptr c; // destroyed only after waiting on users
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
