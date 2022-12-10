#include "flecsi/run/backend.hh"

#include <chrono>
#include <thread>

namespace po = boost::program_options;

namespace flecsi::run {

struct context::backend_arg {
  static auto & args() {
    return instance().backend_args_;
  }
  static const std::string & get(const std::vector<std::string> & v) {
    flog_assert(v.size() == 1, "wrong token count");
    return v.front();
  }
  struct single {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, single *, int) {
      args().push_back(get(sv));
    }
  };
  struct space {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, space *, int) {
      using I = std::istream_iterator<std::string>;
      std::istringstream iss(get(sv));
      args().insert(args().end(), I(iss), I());
    }
  };
};

int
context::initialize_generic(int argc, char ** argv, bool dependent) {
  if(const auto p = std::getenv("FLECSI_SLEEP")) {
    const auto n = std::atoi(p);
    std::cerr << getpid() << ": sleeping for " << n << " seconds...\n";
    std::this_thread::sleep_for(std::chrono::seconds(n));
  }

  initialize_dependent_ = dependent;

  // Save command-line arguments
  for(auto i(0); i < argc; ++i) {
    argv_.push_back(argv[i]);
  } // for

  program_ = argv[0];
  program_ = program_.substr(program_.rfind('/') + 1);

  po::options_description master("Basic Options");
  master.add_options()("help,h", "Print this message and exit.");

  // Add externally-defined descriptions to the main description
  for(auto & od : descriptions_map_) {
    if(od.first != "FleCSI Options") {
      master.add(od.second);
    } // if
  } // for

  po::options_description flecsi_desc =
    descriptions_map_.count("FleCSI Options")
      ? descriptions_map_["FleCSI Options"]
      : po::options_description("FleCSI Options");

  // Together, these initialize backend_args_.
  flecsi_desc.add_options()("backend-args",
    po::value<backend_arg::space>(),
    "Pass arguments to the backend. The single argument is a quoted "
    "string of backend-specific options.");

  flecsi_desc.add_options()("Xbackend",
    po::value<backend_arg::single>(),
    "Pass single argument to the backend. This option can be passed "
    "multiple times.");
#if defined(FLECSI_ENABLE_FLOG)
  std::string flog_tags_;
  int flog_verbose_;
  int64_t flog_output_process_;
  // Add FleCSI options
  flecsi_desc.add_options() // clang-format off
      (
        "flog-tags",
        po::value(&flog_tags_)
          ->default_value("all"),
        "Enable the specified output tags, e.g., --flog-tags=tag1,tag2."
        " Use '--flog-tags=all' to show all output, and "
        " '--flog-tags=unscoped' to show only unguarded output."
      )
      (
        "flog-verbose",
        po::value(&flog_verbose_)
          ->implicit_value(1)
          ->default_value(0),
        "Enable verbose output. Passing '-1' will strip any additional"
        " decorations added by flog and will only output the user's message."
      )
      (
        "flog-process",
        po::value(&flog_output_process_)->default_value(0),
        "Restrict output to the specified process id. The default is process 0."
        " Use '--flog-process=-1' to enable all processes."
      ); // clang-format on
#endif

  // Make an options description to hold all options. This is useful
  // to exlude hidden options from help.
  po::options_description all("All Options");
  all.add(master);
  all.add(flecsi_desc);
  all.add(hidden_options_);

  po::parsed_options parsed = po::command_line_parser(argc, argv)
                                .options(all)
                                .positional(positional_desc_)
                                .allow_unregistered()
                                .run();

  auto print_usage =
    [this](std::string program, auto const & master, auto const & flecsi) {
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
        auto const & tm = flog::state::instance().tag_map();

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
    po::variables_map vm;
    po::store(parsed, vm);

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

    po::notify(vm);

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
  catch(po::error & e) {
    std::string error(e.what());

    auto pos = error.find("--");
    if(pos != std::string::npos) {
      error.replace(pos, 2, "");
    } // if

    std::cerr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED << error
              << "!!!" << FLOG_COLOR_PLAIN << std::endl
              << std::endl;
    print_usage(program_, master, flecsi_desc);
    return status::command_line_error;
  } // try

  unrecognized_options_ =
    po::collect_unrecognized(parsed.options, po::include_positional);

#if defined(FLECSI_ENABLE_FLOG)
  if(flog::state::instance().initialize(
       flog_tags_, flog_verbose_, flog_output_process_)) {
    return status::error;
  } // if
#endif

  initialized_ = true;

  return status::success;
}

} // namespace flecsi::run
