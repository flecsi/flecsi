// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_TASK_WRAPPER_HH
#define FLECSI_EXEC_LEG_TASK_WRAPPER_HH

#include "flecsi/config.hh"

#include "flecsi/exec/params.hh"
#include "flecsi/exec/task_attributes.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/annotation.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/function_traits.hh"
#include <flecsi/flog.hh>

#include <legion.h>

#include <regex>
#include <string>
#include <utility>

namespace flecsi::exec::leg {
/// \addtogroup legion-execution
/// \{
using run::leg::task;

namespace detail {
template<typename RETURN, task<RETURN> * TASK, TaskAttributes A>
void register_task();

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
  static_assert(
    processor_type != processor::mpi, "Legion tasks cannot use MPI");

  std::string name = util::symbol<*TASK>();

  // extract wrapped task
  constexpr char wrapper_prefix[] = "flecsi::exec::leg::task_wrapper<";
  if(name.rfind(wrapper_prefix, 0) == 0) {
    auto wrap_end = name.rfind(", (flecsi::exec::processor)");
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
      std::string(m[1]) + "flecsi::data::accessor<" + layouts[std::stoi(m[2])]);
  }

  std::string sig = name;
  name = util::strip_return_type(util::strip_parameter_list(name));

  // hash signature and attach to short name
  std::stringstream ss;
  auto hash = std::hash<std::string>{}(sig);
  ss << name << " # " << std::hex << hash;
  name = ss.str();

  flecsi::run::context::instance().task_names()[name] = sig;

  Legion::TaskVariantRegistrar registrar(task_id<*TASK, A>, name.c_str());
  Legion::Processor::Kind kind;
  switch(processor_type) {
    case processor::toc:
      kind = Legion::Processor::TOC_PROC;
      break;
    case processor::omp:
      kind = Legion::Processor::OMP_PROC;
      break;
    default:
      kind = Legion::Processor::LOC_PROC;
      break;
  }

  registrar.add_constraint(Legion::ProcessorConstraint(kind));

  registrar.set_leaf(A & leaf || ~A & inner);
  registrar.set_inner(A & inner);

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

template<class P>
struct parameters {
  parameters(P p) : params(std::move(p)) {}
  template<class Q>
  parameters(parameters<Q> && q)
    : params(std::move(q.params)), which(std::move(q.which)) {}

  P params;
  bindings which;
};

template<class>
struct decay_tuple {};
template<class... TT>
struct decay_tuple<std::tuple<TT...>> {
  using type = std::tuple<std::decay_t<TT>...>;
};

template<class... PP>
auto
bind_tuple(const std::tuple<PP...> & tup) { // to deduce a pack
  // Copy only those elements that need to be modified per point task:
  return std::tuple<
    std::conditional_t<exec::detail::must_bind_v<PP>, PP, const PP &>...>(tup);
}

/*!
 The task_wrapper type provides execution
 functions for user and MPI tasks.

 \tparam F the user task
 \tparam P the target processor type
 */

template<auto & F, processor P>
struct task_wrapper {

  using Traits = util::function_t<F>;
  using RETURN = typename Traits::return_type;
  static constexpr processor LegionProcessor =
    P == processor::mpi ? processor::loc : P;

  /*!
    Execution wrapper method for user tasks.
   */

  static RETURN execute(const Legion::Task * task,
    const std::vector<Legion::PhysicalRegion> & regions,
    Legion::Context context,
    Legion::Runtime * runtime) noexcept {
    static constexpr bool mpi = P == processor::mpi;

    auto & c = run::context::instance();

    const auto call = [&](auto && ours, const bindings & which) {
      namespace ann = util::annotation;
      auto tname = util::symbol<F>();
      (ann::rguard<ann::execute_task_bind>(tname),
        bind_parameters<P>(
          ours, runtime, context, regions, task->futures, which));
      if constexpr(mpi) {
        if constexpr(std::is_void_v<RETURN>) {
          (ann::rguard<ann::execute_task_user>(tname)),
            c.mpi_call([&] { apply(F, std::move(ours)); });
        }
        else {
          std::optional<RETURN> result;
          (ann::rguard<ann::execute_task_user>(tname)),
            c.mpi_call([&] { result.emplace(std::apply(F, std::move(ours))); });
          return std::move(*result);
        }
      }
      else
        return ann::rguard<ann::execute_task_user>(tname),
               run::task_local_base::guard(), apply(F, std::move(ours));
    };

    if constexpr(mpi) {
      flog_assert(!task->arglen, "unexpected task arguments");
      auto & p = *static_cast<parameters<typename Traits::arguments_type> *>(
        c.mpi_params);
      return call(p.params, p.which);
    }
    else {
      // There is a optimization opportunity here to move
      // the elements instead of copying the last time.
      const auto access = c.params.at(run::get1<std::size_t>(*task));
      const auto & p = access.get<parameters<
        typename decay_tuple<typename Traits::arguments_type>::type>>();
      return call(bind_tuple(p.params), p.which);
    }
  }
}; // struct task_wrapper

/// \}
} // namespace flecsi::exec::leg

#endif
