// Support for cross-color field access.

#ifndef FLECSI_DATA_MAP_HH
#define FLECSI_DATA_MAP_HH

#include "flecsi/topo/index.hh"
#include "flecsi/util/color_map.hh"

#include <deque>

namespace flecsi {
namespace data::launch {
/// \defgroup launch Launch maps
/// Selecting topology colors to send to tasks.
/// \warning Only the Legion backend supports non-trivial mappings that select
///   any color but their own.
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

struct claims { // to know the colors for each multi<> component
  claims(const borrow::Claims & c) : clm(c.size()) {
    execute<fill>(topo::claims::field(clm), c);
  }

  topo::claims::core clm;

private:
  static void fill(topo::claims::Field::accessor<wo> a,
    const borrow::Claims & c) {
    a = c[color()];
  }
};

/// A prepared assignment of colors.
/// \tparam P underlying topology
template<class P>
struct mapping : convert_tag {
  using Borrow = topo::borrow<P>;

  mapping(typename P::core & t, const Claims & clm) {
    // Transpose clm for the data::borrow objects.
    // Serialization assumes that we always have at least one round.
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
      rnd.emplace_back(t, std::move(c), rnd.empty());
    }
  }

  Color colors() const {
    return rnd.front().b->colors();
  }
  Color depth() const { // never 0
    return rnd.size();
  }
  auto & operator[](Color i) {
    return rnd[i].b.get();
  }
  auto claims(Color i) {
    return topo::claims::field(rnd[i].clm);
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
    return rnd[i].b;
  }

private:
  // Owns a set of claims for potentially several (nested) borrow topologies.
  struct round : claims {
    round(typename P::core & t, borrow::Claims c, bool first)
      : claims(c), proj(std::move(c)) {
      b.allocate({&t, &proj, first});
    }
    round(round &&) = delete; // address stability

  private:
    borrow proj;

  public:
    typename Borrow::slot b;
  };

  std::deque<round> rnd;
};
template<class T>
mapping(T &, const Claims &) -> mapping<topo::policy_t<T>>;

template<class T>
mapping<topo::policy_t<T>>
make(T & t) { // convenience for subtopology initialization
  return {t, block(t.colors(), processes())};
}
/// Create a \c mapping.
template<class P>
mapping<P>
make(topology_slot<P> & t, const Claims & c) {
  return {t.get(), c};
}
/// Create a \c mapping for initialization using an MPI task.
/// The \ref\c Claims are constructed using \ref\c block.
template<class P>
mapping<P>
make(topology_slot<P> & t) {
  return make(t.get());
}

/// \}
} // namespace data::launch

template<class P, class T>
struct exec::detail::launch<P, data::launch::mapping<T>> {
  static Color get(const data::launch::mapping<T> & m) {
    return m.colors();
  }
};

} // namespace flecsi

#endif
