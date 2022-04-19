// Support for cross-color field access.

#ifndef FLECSI_DATA_MAP_HH
#define FLECSI_DATA_MAP_HH

#include "flecsi/topo/index.hh"

#include <deque>

namespace flecsi {
namespace data::launch {
/// \defgroup launch Launch maps
/// Selecting topology colors to send to tasks.
/// \ingroup data
/// \{

/// \if core
/// Parameter type for establishing a \c mapping.
/// \endif
using param = topo::claims::Field::Reference<topo::claims, topo::elements>;

/// Rule to assign colors in blocks.
/// For example, 5 colors are assigned to 3 tasks as {0,1}, {2,3}, and {4}.
inline bool
block(topo::claims::Field::accessor<wo> a, Color i, Color n) {
  const auto me = color(), us = colors(), q = n / us, r = n % us,
             mine = q + (me < r);
  a = topo::claims::row(
    i < mine ? std::optional((mine + (me == r)) * me + i) : std::nullopt);
  return i + 1 < mine;
}
/// Rule to assign colors in a cycle.
/// For example, 5 colors are assigned to 3 tasks as {0,3}, {1,4}, and {2}.
inline bool
robin(topo::claims::Field::accessor<wo> a, Color i, Color n) {
  const auto f = [me = color(), us = colors()](Color i) { return i * us + me; };
  const Color c = f(i);
  a = topo::claims::row(c < n ? std::optional(c) : std::nullopt);
  return f(i + 1) < n;
}
/// Make all colors available on all point tasks.
inline bool
gather(topo::claims::Field::accessor<wo> a, Color i, Color n) {
  a = topo::claims::row(i);
  return i < n - 1;
}

/// A prepared assignment of colors.
/// Invalidated by resizing the underlying topology (\e e.g., with a mutator).
/// \tparam P underlying topology
template<class P>
struct mapping : convert_tag {
  using Borrow = topo::borrow<P>;

  // f: param -> bool (another partition is needed)
  template<class F>
  mapping(typename P::core & t, Color n, F && f) {
    bool more;
    // We can't learn in time that 0 rounds are needed, but we make use of the
    // guaranteed single round elsewhere anyway.
    do {
      topo::claims::core clm(n);
      more = f(topo::claims::field(clm));
      rnd.emplace_back(t, std::move(clm), rnd.empty());
    } while(more);
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
  struct round {
    round(typename P::core & t, topo::claims::core && c, bool first)
      : clm(std::move(c)) {
      b.allocate({&t, &clm, first});
    }
    round(round &&) = delete; // address stability

  private:
    topo::claims::core clm;

  public:
    typename Borrow::slot b;
  };

  std::deque<round> rnd;
};
template<class T, class F>
mapping(T &, Color, F &&) -> mapping<topo::policy_t<T>>;

template<auto & F = block, class T>
mapping<topo::policy_t<T>>
make(T & t, Color n = processes()) {
  return {t, n, [c = t.colors(), i = Color()](param r) mutable {
            return reduce<F, exec::fold::max>(r, i++, c).get();
          }};
}
/// Create a \c mapping from a rule.
/// \tparam F rule task that accepts a \c topo::claims::Field::accessor<wo>, a
///   round counter, and a count of input colors and returns whether its color
///   needs more claims
/// \param n number of colors
template<auto & F = block, class P>
mapping<P>
make(topology_slot<P> & t, Color n = processes()) {
  return make<F>(t.get(), n);
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
