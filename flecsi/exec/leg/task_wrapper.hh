// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_TASK_WRAPPER_HH
#define FLECSI_EXEC_LEG_TASK_WRAPPER_HH

#include "flecsi/config.hh"
#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/leg/bind_accessors.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/serialize.hh"
#include <flecsi/flog.hh>

#include <legion.h>

#include <regex>
#include <string>
#include <utility>

namespace flecsi {

inline flog::devel_tag task_wrapper_tag("task_wrapper");
inline flog::tag task_names_tag("task_names");

// Task parameter serialization (needed only for Legion):
namespace data {
template<class, Privileges, Privileges>
struct ragged_accessor;
template<class, Privileges, bool>
struct particle_accessor;

namespace detail {
template<class A>
struct convert_accessor {
  using type = A;
  using Base = typename type::base_type;
  static const Base & put(const type & a) {
    return a.get_base();
  }
  static type get(Base b) {
    return b;
  }
};
} // namespace detail
} // namespace data

// Send and receive only the field ID:
template<data::layout L, class T, Privileges Priv>
struct util::serial::convert<data::accessor<L, T, Priv>> {
  using type = data::accessor<L, T, Priv>;
  using Rep = field_id_t;
  static Rep put(const type & r) {
    return r.field();
  }
  static type get(const Rep & r) {
    return type(r);
  }
};
template<class R, typename T>
struct util::serial::convert<data::reduction_accessor<R, T>> {
  using type = data::reduction_accessor<R, T>;
  using Rep = field_id_t;
  static Rep put(const type & r) {
    return r.field();
  }
  static type get(const Rep & r) {
    return type(r);
  }
};
template<class T, Privileges Priv>
struct util::serial::convert<data::accessor<data::single, T, Priv>>
  : data::detail::convert_accessor<data::accessor<data::single, T, Priv>> {};
template<class T, Privileges P, Privileges OP>
struct util::serial::convert<data::ragged_accessor<T, P, OP>>
  : data::detail::convert_accessor<data::ragged_accessor<T, P, OP>> {};
template<data::layout L, class T, Privileges Priv>
struct util::serial::traits<data::mutator<L, T, Priv>> {
  using type = data::mutator<L, T, Priv>;
  template<class P>
  static void put(P & p, const type & m) {
    serial::put(p, m.get_base());
  }
  static type get(const std::byte *& b) {
    return serial::get<typename type::base_type>(b);
  }
};
template<class T, Privileges P, bool M>
struct util::serial::convert<data::particle_accessor<T, P, M>>
  : data::detail::convert_accessor<data::particle_accessor<T, P, M>> {};
template<class T, Privileges Priv>
struct util::serial::traits<data::mutator<data::ragged, T, Priv>> {
  using type = data::mutator<data::ragged, T, Priv>;
  template<class P>
  static void put(P & p, const type & m) {
    serial::put(p, m.get_base(), m.get_grow());
  }
  static type get(const std::byte *& b) {
    const serial::cast r{b};
    return {r, r};
  }
};
template<class A>
struct util::serial::traits<data::multi<A>> {
  using type = data::multi<A>;
  template<class P>
  static void put(P & p, const type & m) {
    const auto a = m.accessors();
    serial::put(p, Color(a.size()), a.front());
  }
  static type get(const std::byte *& b) {
    const cast r{b};
    return {r, r};
  }
};
template<class T, Privileges Priv>
struct util::serial::traits<data::topology_accessor<T, Priv>,
  std::enable_if_t<!util::bit_copyable_v<data::topology_accessor<T, Priv>>>>
  : util::serial::value<data::topology_accessor<T, Priv>> {};

template<auto & F, class... AA>
struct util::serial::traits<exec::partial<F, AA...>,
  std::enable_if_t<!util::bit_copyable_v<exec::partial<F, AA...>>>> {
  using type = exec::partial<F, AA...>;
  using Rep = typename type::Base;
  template<class P>
  static void put(P & p, const type & t) {
    serial::put(p, static_cast<const Rep &>(t));
  }
  static type get(const std::byte *& b) {
    return serial::get<Rep>(b);
  }
};

template<class T>
struct util::serial::traits<future<T>> : util::serial::value<future<T>> {};

namespace exec::leg {
/// \addtogroup legion-execution
/// \{
using run::leg::task;

namespace detail {
template<typename RETURN, task<RETURN> * TASK, TaskAttributes A>
void register_task();

template<class>
struct tuple_get;
template<class... TT>
struct tuple_get<std::tuple<TT...>> {
  static auto get(const Legion::Task & t) {
    const auto p = static_cast<const std::byte *>(t.args);
    return util::serial::get_tuple<std::decay_t<TT>...>(p, p + t.arglen);
  }
};
} // namespace detail

/*!
  Arbitrary index for each task.

  @tparam F Legion task function.
  @tparam A task attributes mask
 */

template<auto & F, TaskAttributes A = loc | leaf>
// 'extern' works around GCC bug #96523
extern const Legion::TaskID task_id =
  (run::context::register_init(
     detail::register_task<typename util::function_t<F>::return_type, F, A>),
    Legion::Runtime::generate_static_task_id());

template<typename RETURN, task<RETURN> * TASK, TaskAttributes A>
void
detail::register_task() {
  constexpr auto processor_type = mask_to_processor_type(A);
  static_assert(processor_type != task_processor_type_t::mpi,
    "Legion tasks cannot use MPI");

  std::string name = util::symbol<*TASK>();

  {
    flog::guard guard(task_names_tag);

    // extract wrapped task
    constexpr char wrapper_prefix[] = "flecsi::exec::leg::task_wrapper<";
    if(name.rfind(wrapper_prefix, 0) == 0) {
      auto wrap_end = name.rfind(", (flecsi::exec::task_processor_type_t)");
      name = name.substr(
        sizeof(wrapper_prefix) - 1, wrap_end - sizeof(wrapper_prefix) + 1);
    }

    // replace known layouts
    const std::array<std::string, 6> layouts = {
      "raw", "single", "dense", "ragged", "sparse", "particle"};
    std::regex layout_regex{
      "(^|[^\\w:_])flecsi::data::accessor<\\(flecsi::data::layout\\)(\\d+)"};
    std::smatch m;

    while(std::regex_search(name, m, layout_regex)) {
      name.replace(m[0].first,
        m[0].second,
        std::string(m[1]) + "flecsi::data::accessor<" +
          layouts[std::stoi(m[2])]);
    }

    std::string sig = name;
    name = util::strip_return_type(util::strip_parameter_list(name));

    // hash signature and attach to short name
    std::stringstream ss;
    ss << name << " # ";
    ss << std::hex << std::hash<std::string>{}(sig);
    name = ss.str();

    // allows dumping out full signatures mappping via "task_names" tag
    flog(info) << "Registering task \"" << name << "\": " << sig << "\n";
  }

  Legion::TaskVariantRegistrar registrar(task_id<*TASK, A>, name.c_str());
  Legion::Processor::Kind kind;
  switch(processor_type) {
    case task_processor_type_t::toc:
      kind = Legion::Processor::TOC_PROC;
      break;
    case task_processor_type_t::omp:
      kind = Legion::Processor::OMP_PROC;
      break;
    default:
      kind = Legion::Processor::LOC_PROC;
      break;
  }

  registrar.add_constraint(Legion::ProcessorConstraint(kind));

  registrar.set_leaf(A & leaf || ~A & inner);
  registrar.set_inner(A & inner);
  registrar.set_idempotent(A & idempotent);

  /*
    This section of conditionals is necessary because there is still
    a distinction between void and non-void task registration with
    Legion.
   */

  if constexpr(std::is_same_v<RETURN, void>) {
    Legion::Runtime::preregister_task_variant<TASK>(registrar, name.c_str());
  }
  else {
    Legion::Runtime::preregister_task_variant<RETURN, TASK>(
      registrar, name.c_str());
  } // if
}

/*!
 The task_wrapper type provides execution
 functions for user and MPI tasks.

 \tparam F the user task
 \tparam P the target processor type
 */

template<auto & F, task_processor_type_t P>
struct task_wrapper {

  using Traits = util::function_t<F>;
  using RETURN = typename Traits::return_type;
  using param_tuple = typename Traits::arguments_type;

  static constexpr task_processor_type_t LegionProcessor = P;

  /*!
    Execution wrapper method for user tasks.
   */

  static RETURN execute(const Legion::Task * task,
    const std::vector<Legion::PhysicalRegion> & regions,
    Legion::Context context,
    Legion::Runtime * runtime) {
    {
      flog::devel_guard guard(task_wrapper_tag);
      flog_devel(info) << "In execute_user_task" << std::endl;
    }

    // Unpack task arguments
    auto task_args = detail::tuple_get<param_tuple>::get(*task);

    namespace ann = util::annotation;
    auto tname = util::symbol<F>();
    const param_buffers buf(task_args, tname);
    (ann::rguard<ann::execute_task_bind>(tname),
      bind_accessors<P>(runtime, context, regions, task->futures)(task_args));
    return ann::rguard<ann::execute_task_user>(tname),
           run::task_local_base::guard(),
           apply(F, std::forward<param_tuple>(task_args));
  } // execute_user_task

}; // struct task_wrapper

template<auto & F>
struct task_wrapper<F, task_processor_type_t::mpi> {
  using Traits = util::function_t<F>;
  using RETURN = typename Traits::return_type;
  using param_tuple = typename Traits::arguments_type;

  static constexpr auto LegionProcessor = task_processor_type_t::loc;

  static RETURN execute(const Legion::Task * task,
    const std::vector<Legion::PhysicalRegion> & regions,
    Legion::Context context,
    Legion::Runtime * runtime) {
    {
      flog::devel_guard guard(task_wrapper_tag);
      flog_devel(info) << "In execute_mpi_task" << std::endl;
    }

    flog_assert(!task->arglen, "unexpected task arguments");
    auto & c = run::context::instance();
    const auto p = static_cast<param_tuple *>(c.mpi_params);

    namespace ann = util::annotation;
    auto tname = util::symbol<F>();
    const param_buffers buf(*p, tname);
    (ann::rguard<ann::execute_task_bind>(tname)),
      bind_accessors<LegionProcessor>(runtime, context, regions, task->futures)(
        *p);

    // Set the MPI function and make the runtime active.
    if constexpr(std::is_void_v<RETURN>) {
      (ann::rguard<ann::execute_task_user>(tname)),
        c.mpi_call([&] { apply(F, std::move(*p)); });
    }
    else {
      std::optional<RETURN> result;
      (ann::rguard<ann::execute_task_user>(tname)),
        c.mpi_call([&] { result.emplace(std::apply(F, std::move(*p))); });
      return std::move(*result);
    }

  } // execute
}; // task_wrapper

/// \}
} // namespace exec::leg
} // namespace flecsi

#endif
