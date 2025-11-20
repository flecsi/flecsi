// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_HPX_CONTEXT_HH
#define FLECSI_RUN_HPX_CONTEXT_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/config.hh"
#include "flecsi/run/local/context.hh"

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <set>
#include <utility>

namespace flecsi::run {
/// \defgroup hpx-runtime HPX Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct communicator {
  using type = ::hpx::collectives::communicator;
  // Provide a stable address for asynchronous operations:
  using ptr = std::unique_ptr<communicator>;

  communicator() = default;
  communicator(type c) : c(std::move(c)) {}
  communicator(communicator &&) = default;
  communicator & operator=(communicator &&) & = default;

  const type & comm() const {
    return c;
  }
  auto gen() {
    return ::hpx::collectives::generation_arg(++g);
  }

private:
  type c;
  std::size_t g = 0;
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

  iterator begin() {
    return l.begin();
  }
  iterator end() {
    return l.end();
  }
  auto begin() const {
    return l.begin();
  }
  auto end() const {
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

  void depend(ptr n) {
    memo m;
    if(n && n.get() != this && !absorb(n, m))
      past.push(std::move(n));
  }

  communicator & get() & {
    memo m;
    collapse(m);
    if(ours.empty()) {
      if(past.empty())
        ours.push_back(make_comm());
      else {
        // Take from a direct predecessor; deeper means more broadly useful.
        const auto it = --past.end();
        comms & p = **it;
        ours.push_back(std::move(p.ours.back()));
        p.ours.pop_back();
        if(inherit(p))
          past.erase(it);
      }
    }
    return *ours.front();
  }

  explicit operator bool() const {
    return !ours.empty() || !past.empty();
  }

  static ptr make() {
    return std::make_shared<comms>();
  }

private:
  using memo = std::set<comms *>;
  static comms * key(const ptr & p) {
    return p.get();
  }

  static inline communicator::ptr make_comm();

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
    const bool uniq = c.use_count() == 1;
    if(uniq || (!c->past.empty() && m.insert(c.get()).second))
      c->collapse(m);
    if(uniq) {
      auto v = std::move(c->ours);
      if(v.size() > ours.size()) // make the smaller insertion
        ours.swap(v);
      ours.insert(
        ours.end(), std::move_iterator(v.begin()), std::move_iterator(v.end()));
      past.merge(std::move(c->past));
      return true;
    }
    return inherit(*c);
  }
  bool inherit(const comms & c) {
    if(!c.ours.empty())
      return false;
    for(const auto & p : c.past)
      past.push(p);
    return true;
  }

  std::vector<communicator::ptr> ours; // elements never empty
  // Avoid duplicate parents without ordering by process-local addresses:
  ordered_set<ptr, key> past;
};

struct config : config_base {
  std::vector<std::string> hpx;
};

struct context_t : local::context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//
  context_t(const config &);

  int start(const std::function<int()> &, bool);

  static int task_depth() {
    return 0;
  } // task_depth

  Color color() const {
    return process();
  }

  Color colors() const {
    return processes();
  }

  using p2p = ::hpx::collectives::channel_communicator;

  const p2p & p2p_comm() const {
    return channel;
  }
  auto p2p_tag() {
    return ::hpx::collectives::tag_arg(++tag);
  }
  communicator::ptr world_comm();
  void depend(comms::ptr c) {
    // Partly to avoid data race when using futures inside tasks:
    if(!c || !*c)
      return;
    if(world_comms.use_count() != 1) {
      auto old = std::exchange(world_comms, comms::make());
      world_comms->depend(std::move(old));
    }
    world_comms->depend(std::move(c));
  }
  comms::ptr world_comms;

private:
  struct outstanding_guard {
    outstanding_guard(context_t * c) : c(c) {
      ++c->out;
    }
    outstanding_guard(outstanding_guard && o) noexcept
      : c(std::exchange(o.c, {})) {}
    ~outstanding_guard() {
      if(c && --c->out) {
        (std::lock_guard(c->out_mutex));
        c->out_cv.notify_one();
      }
    }
    outstanding_guard operator()() {
      return std::move(*this);
    }

  private:
    context_t * c;
  };

public:
  outstanding_guard outstanding() {
    return this;
  }
  void termination_detection();

private:
  std::vector<std::string> cfg;
  p2p channel;
  std::size_t tag = 0, world = 0;
  util::ref_count<std::size_t> out{0};
  ::hpx::mutex out_mutex;
  ::hpx::condition_variable out_cv;
};

communicator::ptr
comms::make_comm() {
  return context::instance().world_comm();
}

/// \}
} // namespace flecsi::run

namespace flecsi {
namespace detail {

struct task_local_data {

  auto begin() {
    return data.begin();
  }
  auto end() {
    return data.end();
  }

  template<typename T>
  auto emplace(void * key) {
    return data.emplace(key, new T());
  }
  void *& get(void * key) noexcept {
    auto ret = data.find(key);
    flog_assert(
      ret != data.end(), "task local storage element should have been created");
    return ret->second;
  }

private:
  std::map<void *, void *> data;
};

// manage task local storage for this task
void create_storage();
task_local_data * storage() noexcept;
void reset_storage() noexcept;

} // namespace detail

template<typename T>
struct task_local : private run::task_local_base {
  T & operator*() noexcept {
    return *get();
  }
  T * operator->() noexcept {
    return get();
  }

private:
  void emplace() override {
    [[maybe_unused]] auto p = detail::storage()->emplace<T>(this);
    flog_assert(
      p.second, "task local storage element should not have been created yet");
  }
  void reset() noexcept override {
    delete static_cast<T *>(
      std::exchange(detail::storage()->get(this), nullptr));
  }
  void create_storage() override {
    detail::create_storage();
  }
  void reset_storage() noexcept override {
    detail::reset_storage();
  }

  T * get() noexcept {
    return static_cast<T *>(detail::storage()->get(this));
  }
};
} // namespace flecsi

#endif // FLECSI_RUN_HPX_CONTEXT_HH
