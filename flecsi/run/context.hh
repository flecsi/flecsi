// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_CONTEXT_HH
#define FLECSI_RUN_CONTEXT_HH

#include <flecsi-config.h>

#include "flecsi/data/field_info.hh"
#include "flecsi/flog.hh"

#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <cstddef>
#include <cstdlib> // getenv
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {

inline log::devel_tag context_tag("context");

namespace run {
/// \addtogroup runtime
/// \{

struct context_t; // supplied by backend

/// exit status returned by initialization code
/// \see flecsi::initialize
/// \see flecsi::run::control::check_status
enum status : int {
  success, /// successful initialization
  help, /// user requested usage help
  control_model, /// print out control model graph in dot format
  control_model_sorted, /// print out sorted control model graph in dot format
  clean, /// any value greater than this implies an error
  command_line_error, /// error parsing command line
  error, // add specific error modes
}; // initialization_codes

/// \cond core

/// The results of parsing command-line options defined by FleCSI.
struct arguments {
  /// A command line.
  using argv = std::vector<std::string>;

  /// Arguments and options not recognized during parsing.
  argv unrecognized;

  /// Specification of operation to be performed.
  struct action {
    /// Program name.
    std::string program;
    status code; ///< Operation mode.
    std::string stderr; ///< Error text from initialization.
  } act; ///< Operation to perform.
  /// Specification for initializing underlying libraries.
  struct dependent {
#ifdef FLECSI_ENABLE_MPI
    argv mpi; ///< Command line for MPI.
#endif
#ifdef FLECSI_ENABLE_KOKKOS
    argv kokkos; ///< Command line for Kokkos.
#endif
  } dep; ///< Underlying initialization arguments.
  /// Specification of options for FleCSI.
  struct config {
#ifdef FLECSI_ENABLE_FLOG
    /// Specification for Flog operation.
    struct log {
      argv tags; ///< Tags to enable (perhaps including "all").
      int verbose, ///< Verbosity level.
        process; ///< Process from which to produce output, or -1 for all.
    } flog; ///< Flog options.
#endif
    /// Command line for FleCSI backend.  Some backends ignore it.
    argv backend;
  } cfg; ///< FleCSI options.

  /// Parse a command line.
  /// \note \c dep contains only the program name.
  arguments(int, char **);

  static std::vector<char *> pointers(argv & v) {
    std::vector<char *> ret;
    ret.reserve(v.size());
    for(auto & s : v)
      ret.push_back(s.data());
    return ret;
  }

private:
  status getopt(int, char **);
};

#ifdef DOXYGEN // implemented per-backend
/// RAII guard for initializing/finalizing FleCSI dependencies.
/// Which are included depends on configuration.
struct dependencies_guard {
  /// Construct the guard, possibly mutating argument values.
  dependencies_guard(arguments::dependent &);
};
#endif

/*!
  The context type provides a high-level execution context interface that
  is implemented by a specific backend.
 */

struct context {
  using field_info_store_t = data::fields;

  /*--------------------------------------------------------------------------*
    Deleted contructor and assignment interfaces.
   *--------------------------------------------------------------------------*/

  context(const context &) = delete;
  context & operator=(const context &) = delete;
  context(context &&) = delete;
  context & operator=(context &&) = delete;

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

  /*--------------------------------------------------------------------------*
    Runtime interface.
   *--------------------------------------------------------------------------*/
protected:
  context(const arguments::config & c,
    arguments::action & a,
    Color np,
    Color proc)
    : process_(proc), processes_(np) {
    if(const auto p = std::getenv("FLECSI_SLEEP")) {
      const auto n = std::atoi(p);
      std::cerr << getpid() << ": sleeping for " << n << " seconds...\n";
      sleep(n);
    }

#if defined(FLECSI_ENABLE_FLOG)
    if(c.flog.process + 1 && Color(c.flog.process) >= np) {
      std::ostringstream stderr;
      stderr << a.program << ": flog process " << c.flog.process
             << " does not exist with " << processes_ << " processes\n";
      a.stderr += std::move(stderr).str();
      a.code = error;
    }
    else
      log::state::instance.emplace(c.flog.tags, c.flog.verbose, c.flog.process);
#endif
  }

  ~context() {
#if defined(FLECSI_ENABLE_FLOG)
    log::state::instance.reset();
#endif
  }

public:
#ifdef DOXYGEN // these functions are implemented per-backend
  /*!
    Start the FleCSI runtime.

    @param action The top-level action FleCSI should execute.

    @return An integer with \em 0 being success, and any other value
            being failure.
   */

  int start(const std::function<int()> & action);
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
    running process that invokded the FleCSI runtime.
   */

  Color threads() const {
    return threads_;
  }

#ifdef DOXYGEN
  /*!
    Return the current task depth within the execution hierarchy. The
    top-level task has depth \em 0. This interface is primarily intended
    for FleCSI developers to use in enforcing runtime constraints.
   */

  static int task_depth();

  /*!
    Get the color of this process.
   */

  Color color() const;

  /*!
    Get the number of colors.
   */

  Color colors() const;
#endif

  static void register_init(void callback()) {
    init_registry.push_back(callback);
  }

  /*--------------------------------------------------------------------------*
    Field interface.
   *--------------------------------------------------------------------------*/

  /*!
    Register field information.

    \tparam Topo topology type
    \tparam Index topology-relative index space
    @param field_info               Field information.
   */
  template<class Topo, typename Topo::index_space Index>
  static void add_field_info(const data::field_info_t * field_info) {
    if(topology_ids_.count(Topo::id()))
      flog_fatal("Cannot add fields on an allocated topology");
    constexpr std::size_t NIndex = Topo::index_spaces::size;
    topology_field_info_map_.try_emplace(Topo::id(), NIndex)
      .first->second[Topo::index_spaces::template index<Index>]
      .push_back(field_info);
  } // add_field_information

  /*!
    Return the stored field info for the given topology type and layout.
    Const version.

    \tparam Topo topology type
    \tparam Index topology-relative index space
   */
  template<class Topo, typename Topo::index_space Index = Topo::default_space()>
  static field_info_store_t const & field_info_store() {
    static const field_info_store_t empty;
    topology_ids_.insert(Topo::id());

    auto const & tita = topology_field_info_map_.find(Topo::id());
    if(tita == topology_field_info_map_.end())
      return empty;

    return tita->second[Topo::index_spaces::template index<Index>];
  } // field_info_store

  /*--------------------------------------------------------------------------*
    Task Launch interface.
   *--------------------------------------------------------------------------*/

  /*!
    Return the count of executed tasks. Const version.
   */

  size_t const & flog_task_count() const {
    return flog_task_count_;
  } // flog_task_count

  /*!
    Return the count of executed tasks.
   */

  size_t & flog_task_count() {
    return flog_task_count_;
  } // flog_task_count

  static std::optional<context_t> ctx;

protected:
  // Invoke initialization callbacks.
  // Call from hiding function in derived classses.
  void start() {
    for(auto ro : init_registry)
      ro();
  }

  /*--------------------------------------------------------------------------*
    Program options data members.
   *--------------------------------------------------------------------------*/

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

  /*--------------------------------------------------------------------------*
    Field data members.
   *--------------------------------------------------------------------------*/

  /*
    This type allows storage of runtime field information per topology type.
   */

  static inline std::unordered_map<TopologyType,
    std::vector<field_info_store_t>>
    topology_field_info_map_;

  /// Set of topology types for which field definitions have been used
  static inline std::set<TopologyType> topology_ids_;

  /*--------------------------------------------------------------------------*
    Task count.
   *--------------------------------------------------------------------------*/

  size_t flog_task_count_ = 0;

private:
  static inline std::vector<void (*)()> init_registry;
}; // struct context

struct task_local_base {
  struct guard {
    guard() {
      for(auto * p : all)
        p->emplace();
    }
    ~guard() {
      for(auto * p : all)
        p->reset();
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
};

/// \endcond

/// \}
} // namespace run
} // namespace flecsi

#endif
