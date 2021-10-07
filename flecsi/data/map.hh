// Support for cross-color field access.

#ifndef FLECSI_DATA_MAP_HH
#define FLECSI_DATA_MAP_HH

#include "flecsi/topo/index.hh"

#include <deque>

namespace flecsi {
namespace data::launch {

using param = topo::claims::Field::Reference<topo::claims, topo::elements>;

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
      rnd.emplace_back(t, std::move(clm));
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
    round(typename P::core & t, topo::claims::core && c) : clm(std::move(c)) {
      b.allocate({&t, &clm});
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
mapping(T &, Color, F &&)->mapping<topo::policy_t<T>>;

} // namespace data::launch

template<class P, class T>
struct exec::detail::launch<P, data::launch::mapping<T>> {
  static Color get(const data::launch::mapping<T> & m) {
    return m.colors();
  }
};

} // namespace flecsi

#endif
