// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_CONTEXT_HH
#define FLECSI_RUN_CONTEXT_HH

#include "flecsi/config.hh"
#include "flecsi/data/field_info.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/mpi.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp> // InitializationSettings
#endif

#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <chrono>
#include <cstddef>
#include <cstdlib> // getenv
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace flecsi {

// forward declarations
namespace data {
struct region;
struct partition;
template<class Topo>
struct topology_slot;
} // namespace data

namespace topo {
struct global_base;
template<class Topo>
struct ragged;
struct with_ragged_base;
} // namespace topo

namespace run {
/// \addtogroup runtime
/// \{

struct context_t; // supplied by backend

/// Exit status returned by initialization code.
/// \deprecated Used only with \c initialize.
/// \see flecsi::initialize
/// \see flecsi::run::control::check_status
enum /* [[deprecated]] would warn for internal usage */ status : int {
  success, /// successful initialization
  help, /// user requested usage help
  control_model, /// print out control model graph in dot format
  control_model_sorted, /// print out sorted control model graph in dot format
  clean, /// any value greater than this implies an error
  command_line_error /// error parsing command line
};

/// A command line.
using argv = std::vector<std::string>;

inline std::vector<char *>
pointers(argv & v) {
  std::vector<char *> ret;
  ret.reserve(v.size() + 1);
  for(auto & s : v)
    ret.push_back(s.data());
  ret.push_back(nullptr);
  return ret;
}

/// Specification for initializing underlying libraries.
struct dependencies_config {
  argv mpi; ///< Command line for MPI.
#if defined(FLECSI_ENABLE_KOKKOS) && !defined(FLECSI_ENABLE_LEGION)
  /// Configuration for Kokkos.  Present only if support for it is enabled
  /// and Legion is not in use (since it initializes Kokkos itself).
  /// \see [Kokkos
  /// documentation](https://kokkos.github.io/kokkos-core-wiki/API/core/initialize_finalize/InitializationSettings.html)
  Kokkos::InitializationSettings kokkos;
#endif
};

/// Specification of options for FleCSI.
struct config_base {
#ifdef FLECSI_ENABLE_FLOG
  flecsi::flog::config flog; ///< Flog options, if that feature is enabled.
#endif

  argv * backend() & { // overridden by Legion
    return nullptr;
  }
};

#ifdef DOXYGEN // implemented per-backend
/// Backend-specific options.
struct config : config_base {
  /// Command line for Legion, if using it.
  argv legion;
  /// [Configuration](https://hpx-docs.stellar-group.org/branches/master/html/manual/launching_and_configuring_hpx_applications.html)
  /// assignments for HPX, if using it.
  std::vector<std::string> hpx;
};

/// RAII guard for initializing/finalizing FleCSI dependencies.
/// Which are included depends on configuration.
/// Only one guard can ever exist.
struct dependencies_guard {
  /// Construct the guard.
  dependencies_guard(dependencies_config = {});
  /// Immovable.
  dependencies_guard(dependencies_guard &&) = delete;
};
#endif

/// \cond core

struct index_space_info_t {
  const data::region * region;
  const data::partition * partition;
  data::fields fields;
  std::string index_type;
};

/*!
  The context type provides a high-level execution context interface that
  is implemented by a specific backend.
 */

struct context {
  using field_info_store_t = data::fields;

  context(context &&) = delete;

  static inline context_t & instance();

  /*--------------------------------------------------------------------------*
    Program options interface.
   *--------------------------------------------------------------------------*/

  static boost::program_options::positional_options_description &
  positional_description() {
    return positional_desc_;
  }

  static std::map<std::string, std::string> & positional_help() {
    return positional_help_;
  }

  static boost::program_options::options_description & hidden_options() {
    return hidden_options_;
  }

  static std::map<std::string,
    std::pair<bool,
      std::function<bool(boost::any const &, std::stringstream & ss)>>> &
  option_checks() {
    return option_checks_;
  }

  static auto & descriptions_map() {
    return descriptions_map_;
  }

private:
  struct fields {
    explicit fields(std::size_t n) : ff(n), used(false) {}
    std::vector<data::fields> ff; // per index space
    bool used;
  };

  template<class Topo>
  static fields & topology_fields() {
    static fields ret(Topo::index_spaces::size);
    return ret;
  }

  static auto & init_registry() {
    static std::vector<void (*)()> r;
    return r;
  }

  /*--------------------------------------------------------------------------*
    Runtime interface.
   *--------------------------------------------------------------------------*/
protected:
  context(const config_base & c, Color np, Color proc)
    : process_(proc), processes_(np) {
    if(const auto p = std::getenv("FLECSI_SLEEP")) {
      const auto n = std::atoi(p);
      std::cerr << getpid() << ": sleeping for " << n << " seconds...\n";
      std::this_thread::sleep_for(std::chrono::seconds(n));
    }

#if defined(FLECSI_ENABLE_FLOG)
    flog::state::set_instance(c.flog);
#else
    (void)c;
#endif
  }

  ~context() {
#if defined(FLECSI_ENABLE_FLOG)
    flog::state::reset_instance();
#endif
#if defined(REALM_USE_GASNETEX)
    // ensure all MPI calls have completed on all ranks, prior to letting the
    // runtime shutdown on any rank when using GASNET
    util::mpi::test(MPI_Barrier(MPI_COMM_WORLD));
#endif
  }

public:
#ifdef DOXYGEN // these functions are implemented per-backend
  /*!
    Start the FleCSI runtime.

    @param action The top-level action FleCSI should execute.
    \param check_args trap unrecognized backend configuration arguments

    @return An integer with \em 0 being success, and any other value
            being failure.
   */

  [[nodiscard]] int start(const std::function<int()> & action, bool check_args);
#endif

  /*!
    Return the current process id.
   */

  Color process() const {
    return process_;
  }

  /*!
    Return the number of processes.
   */

  Color processes() const {
    return processes_;
  }

  /*!
    Return the number of threads per process.
   */

  Color threads_per_process() const {
    return threads_per_process_;
  }

  /*!
    Return the number of execution instances with which the runtime was
    invoked. In this context a \em thread is defined as an instance of
    execution, and does not imply any other properties. This interface can be
    used to determine the full subscription of the execution instances of the
    running process that invoked the FleCSI runtime.
   */

  Color threads() const {
    return threads_;
  }

  /*!
    Return map containing the short and full signature of all registered tasks.
   */

  auto & task_names() {
    return task_names_;
  }

#ifdef DOXYGEN
  /*!
    Return the current task depth within the execution hierarchy. The
    top-level task has depth \em 0. This interface is primarily intended
    for FleCSI developers to use in enforcing runtime constraints.
   */

  static int task_depth();

  /*!
    Get the color of the current point task.
   */

  Color color() const;

  /*!
    Get the number of colors.
   */

  Color colors() const;
#endif

  static void register_init(void callback()) {
    init_registry().push_back(callback);
  }

  /*--------------------------------------------------------------------------*
    Field interface.
   *--------------------------------------------------------------------------*/

  /*!
    Register field information.

    \tparam Topo topology type
    \tparam Index topology-relative index space
    \tparam Field field data type
    \param id field ID
   */
  template<class Topo, typename Topo::index_space Index, typename Field>
  static void add_field_info(field_id_t id) {
    auto & tf = topology_fields<Topo>();
    if(tf.used)
      flog_fatal("Cannot add fields on an allocated topology");
    tf.ff[Topo::index_spaces::template index<Index>].push_back(
      std::make_shared<data::field_info_t>(
        data::field_info_t{id, sizeof(Field), util::type<Field>()}));
  }

  /*!
    Return the stored field info for the given topology type and layout
    (\c const version).

    \tparam Topo topology type
    \tparam Index topology-relative index space
   */
  template<class Topo, typename Topo::index_space Index = Topo::default_space()>
  static field_info_store_t const & field_info_store() {
    auto & tf = topology_fields<Topo>();
    tf.used = true;
    return tf.ff[Topo::index_spaces::template index<Index>];
  } // field_info_store

  /*--------------------------------------------------------------------------*
    Index space interface.
   *--------------------------------------------------------------------------*/

  template<class Topo>
  void add_topology(typename data::topology_slot<Topo> & slot) {
    add_index_spaces<Topo>(slot, typename Topo::index_spaces());
  }

  const std::vector<index_space_info_t> & get_index_space_info() const {
    return index_space_info_vector_;
  }

  /*--------------------------------------------------------------------------*
    Task Launch interface.
   *--------------------------------------------------------------------------*/

  /*!
    Return the count of executed tasks (\c const version).
   */

  unsigned const & flog_task_count() const {
    return flog_task_count_;
  } // flog_task_count

  /*!
    Return the count of executed tasks.
   */

  unsigned & flog_task_count() {
    return flog_task_count_;
  } // flog_task_count

  static std::optional<context_t> ctx;

protected:
  // Invoke initialization callbacks.
  // Call from hiding function in derived classes.
  void start() {
    for(auto ro : init_registry())
      ro();
  }

private:
  template<class Topo, typename Topo::index_space Index, typename Instance>
  void add_fields(Instance & slot) {
    index_space_info_vector_.push_back({&slot.template get_region<Index>(),
      &slot.template get_partition<Index>(),
      field_info_store<Topo, Index>(),
      util::type<Topo>() + '[' + std::to_string(Index) + ']'});
  } // add_fields

  template<class Topo, typename Topo::index_space... Index>
  void add_index_spaces(typename Topo::slot & slot,
    util::constants<Index...> /* to deduce pack */) {
    // global topology doesn't define get_partition, so skip it for now
    if constexpr(!std::is_same_v<topo::global_base, typename Topo::base>) {
      // register core fields
      (add_fields<Topo, Index>(slot.get()), ...);
      // if present, register ragged fields
      if constexpr(std::is_base_of_v<topo::with_ragged_base,
                     typename Topo::core>) {
        (
          [&] {
            for(const auto & fip :
              field_info_store<topo::ragged<Topo>, Index>()) {
              auto & t = slot->ragged.template get<Index>()[fip->fid];
              // Clang doesn't like "t.space" as a constant expression:
              constexpr auto space =
                std::remove_reference_t<decltype(t)>::space;
              index_space_info_vector_.push_back(
                {&t.template get_region<space>(),
                  &t.template get_partition<space>(),
                  {fip},
                  util::type<Topo>() + "::ragged[" + std::to_string(Index) +
                    ']'});
            }
          }(),
          ...);
      }
    }
  }

  /*--------------------------------------------------------------------------*
    Program options data members.
   *--------------------------------------------------------------------------*/

protected:
  // Option Descriptions
  static inline std::map<std::string,
    boost::program_options::options_description>
    descriptions_map_;

  // Positional options
  static inline boost::program_options::positional_options_description
    positional_desc_;
  static inline boost::program_options::options_description hidden_options_;
  static inline std::map<std::string, std::string> positional_help_;

  // Validation functions
  static inline std::map<std::string,
    std::pair<bool,
      std::function<bool(boost::any const &, std::stringstream & ss)>>>
    option_checks_;

  /*--------------------------------------------------------------------------*
    Basic runtime data members.
   *--------------------------------------------------------------------------*/

  Color process_, processes_, threads_per_process_, threads_;
  std::map<std::string, std::string> task_names_;

  /*--------------------------------------------------------------------------*
    Index space data members.
   *--------------------------------------------------------------------------*/

  std::vector<index_space_info_t> index_space_info_vector_;

  /*--------------------------------------------------------------------------*
    Task count.
   *--------------------------------------------------------------------------*/

  unsigned flog_task_count_ = 0;
}; // struct context

struct task_local_base {
  struct guard {
    guard() {
      if(!all.empty()) {
        all[0]->create_storage();
        for(auto * p : all)
          p->emplace();
      }
    }
    guard(guard &&) = delete;
    ~guard() {
      if(!all.empty()) {
        for(auto * p : all)
          p->reset();
        all[0]->reset_storage();
      }
    }
  };

  task_local_base() {
    all.push_back(this);
  }
  task_local_base(task_local_base &&) = delete;

protected:
  ~task_local_base() {
    all.erase(std::find(all.begin(), all.end(), this));
  }

private:
  static inline std::vector<task_local_base *> all;

  virtual void emplace() = 0;
  virtual void reset() noexcept = 0;
  virtual void create_storage() {}
  virtual void reset_storage() noexcept {}
};

/// \endcond

/// \}
} // namespace run
} // namespace flecsi

#endif
