/*
@@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
/@@/////  /@@          @@////@@ @@////// /@@
/@@       /@@  @@@@@  @@    // /@@       /@@
/@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
/@@////   /@@/@@@@@@@/@@       ////////@@/@@
/@@       /@@/@@//// //@@    @@       /@@/@@
/@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
//       ///  //////   //////  ////////  //

Copyright (c) 2020, Triad National Security, LLC
All rights reserved.
                                                            */
#pragma once

/*! @file */

#include "flecsi/data/backend.hh"
#include "flecsi/data/field.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/serialize.hh"

// This will need to support different kind of constructors for
// src vs dst
// indirect vs direct
// indirect: range vs points

// indirect (point), direct
// indirect (point), indirect => mesh

namespace flecsi {

namespace data {
namespace detail {
struct intervals : topo::specialization<topo::array_category, intervals> {
  static const field<data::intervals::Value>::definition<intervals> field;
};
// Now that intervals is complete:
inline const field<data::intervals::Value>::definition<intervals>
  intervals::field;
} // namespace detail

struct copy_plan {
  using Sizes = detail::intervals::coloring;

  template<template<class> class C,
    class P,
    typename P::index_space S = P::default_space(),
    class D,
    class F>
  copy_plan(C<P> & t,
    const Sizes & ndests,
    D && dests,
    F && src,
    util::constant<S> = {})
    : dest_ptrs_(ndests),
      // In this first case we use a subtopology to create the
      // destination partition which supposed to be contiguous
      dest_(t.template get_region<S>(),
        dest_ptrs_,
        (std::forward<D>(dests)(detail::intervals::field(dest_ptrs_)),
          detail::intervals::field.fid)),
      // From the pointers we feed in the destination partition
      // we create the source partition
      src_partition_(t.template get_region<S>(),
        dest_,
        (std::forward<F>(src)(pointers<P, S>(t)), pointers<P, S>.fid)),
      engine(src_partition_, dest_, pointers<P, S>.fid) {}

  void issue_copy(const field_id_t & data_fid) const {
    engine(data_fid);
  }

private:
  detail::intervals::core dest_ptrs_;
  intervals dest_;
  points src_partition_;
  copy_engine engine;

  template<class T, typename T::index_space S>
  static inline const field<points::Value>::definition<T, S> pointers;
}; // struct copy_plan

namespace detail {
// This topology implements pipe-like communication in terms of our
// color-specific accessors by having one ghost index point for every edge in
// a directed communication graph.
struct buffers_base {
  using coloring = std::vector<std::vector<Color>>; // [src][dest]
  // Each edge gets one buffer which can be used for transferring arbitrary
  // data via serialization.
  struct buffer {
    // An input stream for a buffer.
    struct reader {
      // Use to select a type to read from context.
      struct convert {
        reader * r;

        template<class T>
        operator T() const {
          return r->get<T>();
        }
      };

      const buffer * b;
      const std::byte * p = b->data.data();
      std::size_t i = 0;

      explicit operator bool() const { // false if at end
        return i < b->len;
      }
      template<class T>
      T get() {
        ++i;
        return util::serial_get<T>(p);
      }
      convert operator()() {
        return {this};
      }
    };
    // An output stream for a buffer.
    struct writer {
      explicit writer(buffer & b) : b(&b) {
        b.len = 0;
      }
      writer(writer &&) = default; // actually a copy, but don't do it casually

      buffer & get_buffer() const {
        return *b;
      }
      template<class T>
      bool operator()(const T & t) { // false if full
        std::size_t o = p - b->data.data();
        util::serial_put(o, t);
        const bool ret = o <= size;
        if(ret) {
          util::serial_put(p, t);
          ++b->len;
        }
        return ret;
      }

    private:
      buffer * b;
      std::byte * p = b->data.data();
    };

    // Provided for convenience in transferring multiple objects in data:
    std::size_t off, len; // off ignored by stream helpers
    static constexpr std::size_t page = 1 << 12,
                                 size = page - sizeof off - sizeof len;
    std::array<std::byte, size> data;

    reader read() const & {
      return {this};
    }
    writer write() & {
      return writer(*this);
    }
  };
  static_assert(sizeof(buffer) <= buffer::page,
    "unexpected padding in buffer layout");

protected:
  using Intervals = std::vector<subrow>;
  using Points = std::vector<std::vector<points::Value>>;

  static void set_dests(field<data::intervals::Value>::accessor<wo> a,
    const Intervals & v) {
    assert(a.span().size() == 1);
    const auto i = color();
    a.span().front() = data::intervals::make(v[i], i);
  }
  static void set_ptrs(field<points::Value>::accessor<wo, wo> a,
    const Points & v) {
    auto & v1 = v[run::context::instance().color()];
    const auto n = v1.size();
    // Our ghosts are always a suffix:
    assert(n <= a.span().size());
    std::copy(v1.begin(), v1.end(), a.span().end() - n);
  }
};
template<class P>
struct buffers_category : buffers_base, topo::array_category<P> {
  using buffers_base::coloring; // to override that from array_category

  explicit buffers_category(const coloring & c)
    : buffers_category(c, [&c] {
        Points ret(c.size());
        Color i = 0;
        for(auto & s : c) {
          std::size_t j = 0;
          for(auto & d : s)
            ret[d].push_back(points::make(i, j++));
          ++i;
        }
        return ret;
      }()) {}

  // Low indices are send buffers, in the order specified in the graph;
  // higher indices are receive buffers, in order of sending color.
  auto operator*() {
    return field(*this);
  }
  template<auto & F, auto & G, class... AA>
  void xfer(AA &&... aa) {
    execute<F>(aa..., **this);
    while(reduce<G, exec::fold::max>(aa..., **this).get())
      ;
  }

  template<class R>
  void ghost_copy(const R & f) {
    cp.issue_copy(f.fid());
  }

  static inline const flecsi::field<buffer>::definition<P> field;

private:
  buffers_category(const coloring & c, const Points & recv)
    : topo::array_category<P>([&] {
        topo::array_base::coloring ret;
        ret.reserve(c.size());
        auto * p = recv.data();
        for(auto & s : c)
          ret.push_back(s.size() + p++->size());
        return ret;
      }()),
      cp(
        *this,
        copy_plan::Sizes(c.size(), 1),
        [&](auto f) {
          Intervals ret;
          ret.reserve(c.size());
          auto * p = recv.data();
          for(auto & s : c)
            ret.push_back({s.size(), s.size() + p++->size()});
          execute<set_dests>(f, ret);
        },
        [&](auto f) { execute<set_ptrs>(f, recv); }) {}

  copy_plan cp;
};
} // namespace detail
} // namespace data
template<>
struct topo::detail::base<data::detail::buffers_category> {
  using type = data::detail::buffers_base;
};
namespace data {
struct buffers : topo::specialization<detail::buffers_category, buffers> {
  using Buffer = base::buffer;
  using Start = field<Buffer>::accessor<wo, na>;
  // Since copy_plan supports only copies between parts of the same logical
  // region, we can't use WRITE_DISCARD for the send buffer.  We therefore use
  // rw for it so that transfer functions can use it to resume large jobs.
  using Transfer = field<Buffer>::accessor<rw, ro>;

  template<index_space>
  static constexpr PrivilegeCount privilege_count = 2;

  // Utility to transfer the contents of ragged rows via buffers.
  struct ragged {
    explicit ragged(Buffer & b) : skip(b.off), w(b) {}

    template<class R> // accessor or mutator
    bool operator()(const R & rag, std::size_t i) {
      const auto row = rag[i];
      const auto n = row.size();
      auto & b = w.get_buffer();
      if(skip < n) {
        // Each row's record is its index, the number of elements remaining to
        // write in it (which might not all fit), and then the elements.
        if(!w(i) || !w(n - skip))
          return false;
        for(auto s = std::exchange(skip, 0); s < n; ++s)
          if(w(row[s]))
            ++b.off;
          else {
            flog_assert(b.len > 2, "no data fits");
            return false;
          }
      }
      else
        skip -= n;
      return true;
    }

    // For the first use in each communication:
    static ragged truncate(Buffer & b) {
      b.off = 0;
      return ragged(b);
    }

    template<class R, class F> // F: remote/shared index -> local/ghost index
    static void read(const R & rag, const Buffer & b, F && f) {
      for(Buffer::reader r{&b}; r;) {
        const auto row = rag[f(r.get<std::size_t>())];
        if(!r)
          break; // in case the write stopped mid-record
        for(std::size_t n = r(); r && n--;)
          row.push_back(r());
      }
    }

  private:
    // Just count linearly (many ghost visitors will be sequential anyway):
    std::size_t skip;
    Buffer::writer w;
  };
};
} // namespace data
} // namespace flecsi
