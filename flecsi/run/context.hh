// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_CONTEXT_HH
#define FLECSI_RUN_CONTEXT_HH

#include <flecsi-config.h>

#include "flecsi/data/field_info.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/demangle.hh"

#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#include <cstddef>
#include <cstdlib> // getenv
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
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

inline flog::devel_tag context_tag("context");

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

  /*--------------------------------------------------------------------------*
    Deleted contructor and assignment interfaces.
   *--------------------------------------------------------------------------*/

  context(const context &) = delete;
  context & operator=(const context &) = delete;
  context(context &&) = delete;
  context & operator=(context &&) = delete;

  /*
    Meyer's singleton instance.
   */

  static inline context_t & instance();

  bool initialized() {
    return initialized_;
  }

  /*--------------------------------------------------------------------------*
    Program options interface.
   *--------------------------------------------------------------------------*/

  boost::program_options::positional_options_description &
  positional_description() {
    return positional_desc_;
  }

  std::map<std::string, std::string> & positional_help() {
    return positional_help_;
  }

  boost::program_options::options_description & hidden_options() {
    return hidden_options_;
  }

  std::map<std::string,
    std::pair<bool,
      std::function<bool(boost::any const &, std::stringstream & ss)>>> &
  option_checks() {
    return option_checks_;
  }

  std::vector<char *> & argv() {
    return argv_;
  }

  std::string const & program() {
    return program_;
  }

  auto & descriptions_map() {
    return descriptions_map_;
  }

  std::vector<std::string> const & unrecognized_options() {
    flog_assert(initialized_,
      "unitialized program options -> "
      "invoke flecsi::initialize_program_options");
    return unrecognized_options_;
  }

  /*--------------------------------------------------------------------------*
    Runtime interface.
   *--------------------------------------------------------------------------*/

  int initialize_generic(int argc, char ** argv, bool dependent);

  inline void finalize_generic() {
#if defined(FLECSI_ENABLE_FLOG)
    flog::state::instance().finalize();
#endif

  } // finalize_generic

#ifdef DOXYGEN // these functions are implemented per-backend
  /*
    Documented in execution.hh
   */

  int initialize(int argc, char ** argv, bool dependent);

  /*!
    Perform FleCSI runtime finalization. If FleCSI was initialized with
    the \em dependent flag set to true, FleCSI will also finalize any runtimes
    on which it depends.
   */

  void finalize();

  /*!
    Start the FleCSI runtime.

    @param action The top-level action FleCSI should execute.

    @return An integer with \em 0 being success, and any other value
            being failure.
   */

  int start(const std::function<int()> & action);

  /*!
    Return the current process id.
   */

  Color process() const;

  /*!
    Return the number of processes.
   */

  Color processes() const;

  /*!
    Return the number of threads per process.
   */

  Color threads_per_process() const;

  /*!
    Return the number of execution instances with which the runtime was
    invoked. In this context a \em thread is defined as an instance of
    execution, and does not imply any other properties. This interface can be
    used to determine the full subscription of the execution instances of the
    running process that invokded the FleCSI runtime.
   */

  Color threads() const;

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

  /*!
    Return the exit status of the FleCSI runtime.
   */

  int & exit_status() {
    return exit_status_;
  }

  void register_init(void callback()) {
    init_registry.push_back(callback);
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
  void add_field_info(field_id_t id) {
    if(topology_ids_.count(Topo::id()))
      flog_fatal("Cannot add fields on an allocated topology");
    constexpr std::size_t NIndex = Topo::index_spaces::size;
    topology_field_info_map_.try_emplace(Topo::id(), NIndex)
      .first->second[Topo::index_spaces::template index<Index>]
      .push_back(std::make_shared<data::field_info_t>(
        data::field_info_t{id, sizeof(Field), util::type<Field>()}));
  } // add_field_information

  /*!
    Return the stored field info for the given topology type and layout.
    Const version.

    \tparam Topo topology type
    \tparam Index topology-relative index space
   */
  template<class Topo, typename Topo::index_space Index = Topo::default_space()>
  field_info_store_t const & field_info_store() {
    static const field_info_store_t empty;
    topology_ids_.insert(Topo::id());

    auto const & tita = topology_field_info_map_.find(Topo::id());
    if(tita == topology_field_info_map_.end())
      return empty;

    return tita->second[Topo::index_spaces::template index<Index>];
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

protected:
  context() = default;
  // Invoke initialization callbacks.
  // Call from hiding function in derived classses.
  void start() {
    for(auto ro : init_registry)
      ro();
  }

#ifdef DOXYGEN
  /*!
    Clear the runtime state of the context.

    Notes:
      - This does not clear objects that cannot be serialized, e.g.,
        std::function objects.
   */

  void clear();
#endif

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
              index_space_info_vector_.push_back({&t.get_region(),
                &t.get_partition(),
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
  std::string program_;
  std::vector<char *> argv_;
  std::vector<std::string> backend_args_;

  bool initialize_dependent_ = true;

  // Option Descriptions
  std::map<std::string, boost::program_options::options_description>
    descriptions_map_;

  // Positional options
  boost::program_options::positional_options_description positional_desc_;
  boost::program_options::options_description hidden_options_;
  std::map<std::string, std::string> positional_help_;

  // Validation functions
  std::map<std::string,
    std::pair<bool,
      std::function<bool(boost::any const &, std::stringstream & ss)>>>
    option_checks_;

  std::vector<std::string> unrecognized_options_;

  /*--------------------------------------------------------------------------*
    Basic runtime data members.
   *--------------------------------------------------------------------------*/

  bool initialized_ = false;
  Color process_, processes_, threads_per_process_, threads_;

  int exit_status_ = 0;

  /*--------------------------------------------------------------------------*
    Field data members.
   *--------------------------------------------------------------------------*/

  /*
    This type allows storage of runtime field information per topology type.
   */

  std::unordered_map<TopologyType, std::vector<field_info_store_t>>
    topology_field_info_map_;

  /// Set of topology types for which field definitions have been used
  std::set<TopologyType> topology_ids_;

  /*--------------------------------------------------------------------------*
    Index space data members.
   *--------------------------------------------------------------------------*/

  std::vector<index_space_info_t> index_space_info_vector_;

  /*--------------------------------------------------------------------------*
    Task count.
   *--------------------------------------------------------------------------*/

  size_t flog_task_count_ = 0;

private:
  struct backend_arg;

  std::vector<void (*)()> init_registry;
}; // struct context

/// \}
} // namespace run
} // namespace flecsi

#endif
