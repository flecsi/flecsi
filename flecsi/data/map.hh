// Support for cross-color field access.

#ifndef FLECSI_DATA_MAP_HH
#define FLECSI_DATA_MAP_HH

#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/color_map.hh"

#include <deque>
#include <stdexcept>

namespace flecsi {
namespace data::launch {
/// \defgroup launch Launch maps
/// Selecting topology colors to send to tasks.
/// \warning Only the Legion backend supports non-trivial mappings that select
///   any color but their own.
///
/// \ns{data::launch}.
/// \ingroup data
/// \{

/// Desired underlying colors for each outer color.
/// Elements need not have the same length.
using Claims = std::vector<borrow::Claims>;

/// Assign colors in blocks.
/// For example, 5 colors are assigned to 3 tasks as {0,1}, {2,3}, and {4}.
/// \param u number of colors
/// \param n number of tasks
inline Claims
block(Color u, Color n) {
  Claims ret;
  for(auto b : util::equal_map(u, n))
    ret.emplace_back(b.begin(), b.end());
  return ret;
}
/// Assign colors in a cycle.
/// For example, 5 colors are assigned to 3 tasks as {0,3}, {1,4}, and {2}.
/// \param u number of colors
/// \param n number of tasks
inline Claims
robin(Color u, Color n) {
  Claims ret;
  for(auto b : util::equal_map(u, n)) {
    Color i = ret.size();
    auto & v = ret.emplace_back();
    for(auto j = b.size(); j--;) {
      v.push_back(i);
      i += n;
    }
  }
  return ret;
}
/// Make all colors available on all point tasks.
/// \param u number of colors
/// \param n number of tasks
inline Claims
gather(Color u, Color n) {
  const util::iota_view v({}, u);
  return {n, {v.begin(), v.end()}};
}

struct colors { // for multi::components
  auto operator*() {
    return topo::claims::field(clm);
  }

  colors(scheduler & s, const borrow::Claims & c) : clm(s, c.size()) {
    s.execute<fill>(exec::on, **this, c);
  }

  topo::claims::topology clm;

private:
  static void fill(exec::cpu s,
    topo::claims::Field::accessor<wo> a,
    const borrow::Claims & c) noexcept {
    a = c[s.launch().index];
  }
};
struct claims : colors {
  claims(scheduler & s, borrow::Claims c) : colors(s, c), proj(std::move(c)) {}
  claims(claims &&) = delete; // address stability
  borrow proj;
};
struct map {
  using ptr = std::shared_ptr<map>;

  map(scheduler & s, const Claims & clm) : bound() {
    for(auto & v : clm)
      for(Color x : v)
        if(x >= bound)
          bound = x + 1;
    // Transpose clm for the data::borrow objects.
    // There is at least one round to hold metadata.
    bool more = true;
    for(borrow::Claims::size_type i = 0; more; ++i) {
      more = false;
      borrow::Claims c;
      c.reserve(clm.size());
      for(auto & v : clm) {
        const auto n = v.size();
        c.push_back(i < n ? v[i] : borrow::nil);
        if(i + 1 < n)
          more = true;
      }
      rounds.emplace_back(s, std::move(c));
    }
  }

  std::deque<launch::claims> rounds;
  Color bound;
};

/// A prepared assignment of colors.
/// Declare `multi<Topo::accessor<...>>` task parameter to use the topology.
/// \tparam P underlying topology
/// \see field::definition
template<class P>
struct mapping : convert_tag {
  using Borrow = topo::borrow<P>;

  mapping(scheduler & s, typename P::topology & t, const Claims & clm)
    : mapping(t, std::make_shared<launch::map>(s, clm)) {}
  mapping(typename P::topology & t, launch::map::ptr p) : plan(std::move(p)) {
    if(t.colors() < plan->bound)
      throw std::out_of_range("claims beyond topology colors");
    for(auto & c : plan->rounds)
      topo.emplace_back(t, c.proj, topo.empty());
  }

  Color colors() const {
    return topo.front().colors();
  }
  Color depth() const { // never 0
    return topo.size();
  }
  auto claims(Color i) {
    return *plan->rounds[i];
  }

  template<class T, layout L, typename P::index_space S>
  multi_reference<T, L, P, S> operator()(
    const field_reference<T, L, P, S> & f) {
    return {f, *this};
  }

  // Emulate multi_reference to construct topology accessors:
  mapping & map() {
    return *this;
  }
  auto & data(Color i) {
    return topo[i];
  }

  template<class Q>
  auto rebind(topology<Q> & t) {
    return mapping<Q>(t, plan);
  }
  template<class T, layout L, class Topo, typename Topo::index_space S>
  auto rebind(const field_reference<T, L, Topo, S> & f) {
    // The lambda keeps the new mapping alive in the caller.
    return
      [m = rebind(f.topology()), f]() mutable { return multi_reference(f, m); };
  }

private:
  launch::map::ptr plan; // never structurally mutated
  // Nested borrow topologies reuse our claims objects.
  std::vector<typename Borrow::topology> topo;
};
template<class T>
mapping(T &, const Claims &) -> mapping<topo::policy_t<T>>;

/// Create a \c mapping.
template<class P>
mapping<P>
make(scheduler & s, topology<P> & t, const Claims & c) {
  return {s, t, c};
}
/// Create a \c mapping for initialization using an MPI task.
/// The \c Claims are constructed using \link block() `block`\endlink.
template<class P>
mapping<P>
make(scheduler & s, topology<P> & t) {
  return make(s, t, block(t.colors(), s.runtime().processes()));
}
/// \deprecated Pass the topology instance directly (and a \c scheduler).
template<class P>
[[deprecated("pass a scheduler and t.get()")]] mapping<P>
make(topology_slot<P> & t, const Claims & c) {
  return make(*scheduler::instance, t, c);
}
/// \deprecated Pass the topology instance directly (and a \c scheduler).
template<class P>
[[deprecated("pass a scheduler and t.get()")]] mapping<P>
make(topology_slot<P> & t) {
  return make(*scheduler::instance, t.get());
}

/// \}
} // namespace data::launch

template<class P, class T>
struct exec::detail::launch<P, data::launch::mapping<T>> {
  static Index get(const data::launch::mapping<T> & m) {
    return m.colors();
  }
};

} // namespace flecsi

#endif
