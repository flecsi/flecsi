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

/// \cond core
namespace flecsi {

namespace data {
/// \addtogroup topology-data
/// \{
namespace detail {
struct intervals : topo::specialization<topo::array_category, intervals> {
  static const field<data::intervals::Value>::definition<intervals> field;
};
// Now that intervals is complete:
inline const field<data::intervals::Value>::definition<intervals>
  intervals::field;
} // namespace detail

/// Specifies a pattern of data movement among colors in an index space.
/// Copying within a color is permitted but unusual.
struct copy_plan {
  using Sizes = detail::intervals::coloring;

  template<template<class> class C, // allows deducing P
    class P,
    typename P::index_space S = P::default_space(),
    class D,
    class F>
  copy_plan(C<P> & t,
    prefixes & p,
    const Sizes & ndests,
    D && dests, // function of a field reference for intervals
    F && src, // similarly for source points
    util::constant<S> = {})
    : dest_ptrs_(ndests),
      // In this first case we use a subtopology to create the
      // destination partition which supposed to be contiguous
      dest_(p,
        dest_ptrs_,
        detail::intervals::field(dest_ptrs_).use(std::forward<D>(dests)).fid(),
        incomplete),
      // From the pointers we feed in the destination partition
      // we create the source partition
      src_partition_(p,
        dest_,
        pointers<P, S>(t).use(std::forward<F>(src)).fid(),
        incomplete),
      engine(src_partition_, dest_, pointers<P, S>.fid) {}

  void issue_copy(const field_id_t & data_fid) const {
    util::annotation::rguard<util::annotation::execute_task_copy_engine> ann;
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

/*!
 This topology implements pipe-like communication in terms of our
 color-specific accessors by having one ghost index point for every edge in
 a directed communication graph.
*/
struct buffers_base {

  /// The coloring type describes the global directed communication graph
  /// [src_color][ith_communicating_color] = dest_color, where
  /// ith_communicating_color is from [0, size of inner vector].
  using coloring = std::vector<std::vector<Color>>;

  /// The buffer type stores the data array to be used for communication,
  /// along with a reader and writer type to read from and write to the data
  /// array. Each edge in the communication graph gets one buffer which can be
  /// used for transferring arbitrary data via serialization.
  struct buffer {
    /// An input stream for a buffer.
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
        return util::serial::get<T>(p);
      }
      convert operator()() {
        return {this};
      }
    };
    /// An output stream for a buffer.
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
        util::serial::put(o, t);
        const bool ret = o <= size;
        if(ret) {
          util::serial::put(p, t);
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

  /// Low indices are send buffers, in the order specified in the graph;
  /// higher indices are receive buffers, in order of sending color.
  auto operator*() {
    return field(*this);
  }

  /*!
   This method is used to invoke the sending and receiving of data.
   @tparam F Function object, executing F initializes and loads the send
   buffers. The signature of F should include buffers::Start accessor as the
             last argument.
   @tparam G Function object, executing G accesses the receiving buffers, reads
             from them, refills them if necessary, and continues till there is
             no data to be receive. The signature of G should have
   buffers::Truncate accessor as the last argument. G should have an "int"
   return indicating whether there is still data to be read.
   @tparam AA Variadic list of parameters for arguments to be passed to the
   method. This list of arguments would consist of the field references to the
              data fields to be communicated.

   F and G can have different parameters but are invoked with the same
   arguments.
  */
  template<auto & F, auto & G, class... AA>
  void xfer(AA &&... aa) {
    execute<F>(aa..., **this);
    while(reduce<G, exec::fold::max>(aa..., **this).get())
      ;
  }

  // Data is actually moved by ordinary ghost copies for buffer accessors:
  template<class R>
  void ghost_copy(const R & f) {
    cp.issue_copy(f.fid());
  }

  static inline const flecsi::field<buffer>::definition<P> field;

private:
  /*!
   This constructor takes two different representations of the same
   communication graph.
   @param c The global communication graph
   @param recv It has the actual source coordinates (color and source-local
   index) for each buffer to be received.
  */
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
  /// \}
} // namespace data
template<>
struct topo::detail::base<data::detail::buffers_category> {
  using type = data::detail::buffers_base;
};
namespace data {
/// \addtogroup topology-data
/// \{

/*!
 The buffers type provides an interface for dynamic amounts of data.
 This subtopology type also provides conveniences for transfer tasks.
*/
struct buffers : topo::specialization<detail::buffers_category, buffers> {
  using Buffer = base::buffer;

  /// Alias Start is used to provide accessor to the sending buffers. It should
  /// be part of the signature of the two function objects (F, G) needed by the
  /// "xfer" method of underlying buffers_category.
  using Start = field<Buffer>::accessor<wo, na>;
  /// Alias Transfer is used to provide accessor to the receiving buffers. It
  /// should be part of the signature of the two function objects (F, G) needed
  /// by the "xfer" method of underlying buffers_category. Since copy_plan
  /// supports only copies between parts of the same logical region, we can't
  /// use WRITE_DISCARD for the send buffer.  We therefore use rw for it so that
  /// transfer functions can use it to resume large jobs.
  using Transfer = field<Buffer>::accessor<rw, ro>;

  template<index_space>
  static constexpr PrivilegeCount privilege_count = 2;

  /// Utility to transfer the contents of ragged rows via buffers.
  /// Type ragged is used to create actual buffers for setting up
  /// the data to be sent. It includes setting up buffers for both
  /// send and receive data.
  struct ragged {
    explicit ragged(Buffer & b) : skip(b.off), w(b) {}

    /// This operator takes as input an accessor or mutator to the
    /// ragged field that needs to be communicated.
    template<class R> // accessor or mutator
    bool operator()(const R & rag, std::size_t i) {
      const auto row = rag[i];
      const auto n = row.size();
      auto & b = w.get_buffer();
      if(skip < n) {
        // Each row's record is its index, the number of elements remaining to
        // write in it (which might not all fit), and then the elements.
        // The first row is prefixed with a flag to indicate resumption.
        if(!b.len && !w(!!skip) || !w(i) || !w(n - skip))
          return false;
        for(auto s = std::exchange(skip, 0); s < n; ++s)
          if(w(row[s]))
            ++b.off;
          else {
            flog_assert(b.len > 3, "no data fits");
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
      Buffer::reader r{&b};
      flog_assert(r, "empty message");
      bool resume = r();
      while(r) {
        const auto row = rag[f(r.get<std::size_t>())];
        if(!r)
          break; // in case the write stopped mid-record
        if(resume)
          resume = false;
        else
          row.clear();
        std::size_t n = r();
        row.reserve(row.size() + n); // this may overallocate temporarily
        while(r && n--)
          row.push_back(r());
      }
    }

  private:
    // Just count linearly (many ghost visitors will be sequential anyway):
    std::size_t skip;
    Buffer::writer w;
  };
};
/// \}
} // namespace data
} // namespace flecsi
/// \endcond
