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
#include <string>
#include <vector>

namespace flecsi {
namespace data {

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
  future release() {
    const auto p = std::move(f);
    return *p;
  }

  // These can be used even if this object is empty.
  explicit operator bool() const {
    return !!f;
  }
  future get() const {
    return f ? *f : future();
  }

  static fate make(future f = {}) {
    fate ret;
    ret.f = std::make_shared<future>(f);
    return ret;
  }

private:
  std::shared_ptr<future> f;
};

struct backend_storage : local::detail::storage<> {
  // Synchronize with all pending writes to this storage.
  void synchronize() {
    write = {};
  }

  template<class F>
  void do_read(F && f) {
    read.push_back(std::forward<F>(f)(std::as_const(write)));
  }
  template<class F>
  void do_write(F && f) {
    // We neglect the case of bad_alloc stranding a released future.
    if(write) {
      if(read.empty())
        read.push_back(std::move(write));
      else
        write.release(); // reads are newer anyway
    }
    write = std::forward<F>(f)(std::move(read));
    // In case f calls do_read:
    for(auto & r : read)
      r.release();
    read.clear();
  }

private:
  // In practice, these never wait, because tasks keep the region_impl alive
  // and every ghost copy is followed by a task using the same regions.
  std::vector<fate> read; // since most recent write
  fate write;
};
} // namespace data
} // namespace flecsi

#include "flecsi/data/local/policy.hh"

#endif // FLECSI_DATA_HPX_POLICY_HH
