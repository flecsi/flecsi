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

  std::vector<char *> & argv() {
    return argv_;
  }

  std::string const & program() {
    return program_;
  }

  static auto & descriptions_map() {
    return descriptions_map_;
  }

  std::vector<std::string> const & unrecognized_options() {
    return unrecognized_options_;
  }

  /*--------------------------------------------------------------------------*
    Runtime interface.
   *--------------------------------------------------------------------------*/
protected:
  context(int argc, char ** argv, Color np, Color proc)
    : process_(proc), processes_(np) {
    if(const auto p = std::getenv("FLECSI_SLEEP")) {
      const auto n = std::atoi(p);
      std::cerr << getpid() << ": sleeping for " << n << " seconds...\n";
      sleep(n);
    }

    // Save command-line arguments
    for(auto i(0); i < argc; ++i) {
      argv_.push_back(argv[i]);
    } // for

    program_ = argv[0];
    program_ = program_.substr(program_.rfind('/') + 1);

    exit_status_ = getopt(argc, argv);
  }

  status getopt(int argc, char ** argv) {
    boost::program_options::options_description master("Basic Options");
    master.add_options()("help,h", "Print this message and exit.");

    // Add externally-defined descriptions to the main description
    for(auto & od : descriptions_map_) {
      if(od.first != "FleCSI Options") {
        master.add(od.second);
      } // if
    } // for

    boost::program_options::options_description flecsi_desc =
      descriptions_map_.count("FleCSI Options")
        ? descriptions_map_["FleCSI Options"]
        : boost::program_options::options_description("FleCSI Options");

    flecsi_desc.add_options()("backend-args",
      boost::program_options::value<std::vector<std::string>>(),
      "Pass arguments to the backend. The single argument is a quoted "
      "string of backend-specific options.");
#if defined(FLECSI_ENABLE_FLOG)
    std::string flog_tags_;
    int flog_verbose_;
    int64_t flog_output_process_;
    // Add FleCSI options
    flecsi_desc.add_options() // clang-format off
      (
        "flog-tags",
        boost::program_options::value(&flog_tags_)
          ->default_value("all"),
        "Enable the specified output tags, e.g., --flog-tags=tag1,tag2."
        " Use '--flog-tags=all' to show all output, and "
        " '--flog-tags=unscoped' to show only unguarded output."
      )
      (
        "flog-verbose",
        boost::program_options::value(&flog_verbose_)
          ->implicit_value(1)
          ->default_value(0),
        "Enable verbose output. Passing '-1' will strip any additional"
        " decorations added by flog and will only output the user's message."
      )
      (
        "flog-process",
        boost::program_options::value(&flog_output_process_)->default_value(0),
        "Restrict output to the specified process id. The default is process 0."
        " Use '--flog-process=-1' to enable all processes."
      ); // clang-format on
#endif

    // Make an options description to hold all options. This is useful
    // to exlude hidden options from help.
    boost::program_options::options_description all("All Options");
    all.add(master);
    all.add(flecsi_desc);
    all.add(hidden_options_);

    boost::program_options::parsed_options parsed =
      boost::program_options::command_line_parser(argc, argv)
        .options(all)
        .positional(positional_desc_)
        .allow_unregistered()
        .run();

    auto print_usage = [this](std::string program,
                         auto const & master,
                         auto const & flecsi) {
      if(process_ == 0) {
        std::cout << "Usage: " << program << " ";

        size_t positional_count = positional_desc_.max_total_count();
        size_t max_label_chars = std::numeric_limits<size_t>::min();

        for(size_t i{0}; i < positional_count; ++i) {
          std::cout << "<" << positional_desc_.name_for_position(i) << "> ";

          const size_t size = positional_desc_.name_for_position(i).size();
          max_label_chars = size > max_label_chars ? size : max_label_chars;
        } // for

        max_label_chars += 2;

        std::cout << std::endl << std::endl;

        if(positional_count) {
          std::cout << "Positional Options:" << std::endl;

          for(size_t i{0}; i < positional_desc_.max_total_count(); ++i) {
            auto const & name = positional_desc_.name_for_position(i);
            auto help = positional_help_.at(name);
            std::cout << "  " << name << " ";
            std::cout << std::string(max_label_chars - name.size() - 2, ' ');

            if(help.size() > 78 - max_label_chars) {
              std::string first = help.substr(
                0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
              std::cout << first << std::endl;
              help =
                help.substr(first.size() + 1, help.size() - first.size() + 1);

              while(help.size() > 78 - max_label_chars) {
                std::string part = help.substr(
                  0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
                std::cout << std::string(max_label_chars + 1, ' ') << part
                          << std::endl;
                help = help.substr(part.size() + 1, help.size() - part.size());
              } // while

              std::cout << std::string(max_label_chars + 1, ' ') << help
                        << std::endl;
            }
            else {
              std::cout << help << std::endl;
            } // if

          } // for

          std::cout << std::endl;
        } // if

        std::cout << master << std::endl;
        std::cout << flecsi << std::endl;

#if defined(FLECSI_ENABLE_FLOG)
        auto const & tm = log::state::tag_map();

        if(tm.size()) {
          std::cout << "Available FLOG Tags (FleCSI Logging Utility):"
                    << std::endl;
        } // if

        for(auto t : tm) {
          std::cout << "  " << t.first << std::endl;
        } // for
#endif
      } // if
    }; // print_usage

    try {
      boost::program_options::variables_map vm;
      boost::program_options::store(parsed, vm);

      if(vm.count("backend-args")) {
        auto args = vm["backend-args"].as<std::vector<std::string>>();
        using I = std::istream_iterator<std::string>;
        for(auto & a : args) {
          auto iss = std::istringstream(a);
          backend_args_.insert(backend_args_.end(), I(iss), I());
        }
      }
      if(vm.count("help")) {
        print_usage(program_, master, flecsi_desc);
        return status::help;
      } // if
      if(vm.count("control-model")) {
        return status::control_model;
      } // if
      if(vm.count("control-model-sorted")) {
        return status::control_model_sorted;
      } // if

      boost::program_options::notify(vm);

      // Call option check methods
      for(auto const & [name, boost_any] : vm) {
        auto const & ita = option_checks_.find(name);
        if(ita != option_checks_.end()) {
          auto [positional, check] = ita->second;

          std::stringstream ss;
          if(!check(boost_any.value(), ss)) {
            if(process_ == 0) {
              std::string dash = positional ? "" : "--";
              std::cerr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED
                        << "invalid argument for '" << dash << name
                        << "' option!!!" << std::endl
                        << FLOG_COLOR_LTRED << (ss.str().empty() ? "" : " => ")
                        << ss.str() << FLOG_COLOR_PLAIN << std::endl
                        << std::endl;
            } // if

            print_usage(program_, master, flecsi_desc);
            return status::help;
          } // if
        } // if
      } // for
    }
    catch(boost::program_options::error & e) {
      std::string error(e.what());

      auto pos = error.find("--");
      if(pos != std::string::npos) {
        error.replace(pos, 2, "");
      } // if

      if(!process_)
        std::cerr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED << error
                  << "!!!" << FLOG_COLOR_PLAIN << std::endl
                  << std::endl;
      print_usage(program_, master, flecsi_desc);
      return status::command_line_error;
    } // try

    unrecognized_options_ = boost::program_options::collect_unrecognized(
      parsed.options, boost::program_options::include_positional);

#if defined(FLECSI_ENABLE_FLOG)
    if(flog_output_process_ + 1 && flog_output_process_ >= processes_) {
      if(!process_)
        std::cerr << program_ << ": flog process " << flog_output_process_
                  << " does not exist with " << processes_ << " processes"
                  << std::endl;
      return status::error;
    }
    std::vector<std::string> tags;
    if(flog_tags_ != "none") {
      std::istringstream is(flog_tags_);
      std::string tag;
      while(std::getline(is, tag, ','))
        tags.push_back(tag);
    }
    log::state::instance.emplace(tags, flog_verbose_, flog_output_process_);
#endif

    return status::success;
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

  /*!
    Return the exit status of the FleCSI runtime.
   */

  int & exit_status() {
    return exit_status_;
  }

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

  std::string program_;
  std::vector<char *> argv_;
  std::vector<std::string> backend_args_;

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

  std::vector<std::string> unrecognized_options_;

  /*--------------------------------------------------------------------------*
    Basic runtime data members.
   *--------------------------------------------------------------------------*/

  Color process_, processes_, threads_per_process_, threads_;

  int exit_status_;

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

/// \}
} // namespace run
} // namespace flecsi

#endif
