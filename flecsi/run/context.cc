#include "flecsi/run/backend.hh"

namespace po = boost::program_options;

namespace flecsi::run {

namespace {
struct backend_arg {
  static inline arguments::argv * argv;
  static const std::string & get(const std::vector<std::string> & v) {
    flog_assert(v.size() == 1, "wrong token count");
    return v.front();
  }
  struct single {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, single *, int) {
      argv->push_back(get(sv));
    }
  };
  struct space {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, space *, int) {
      using I = std::istream_iterator<std::string>;
      std::istringstream iss(get(sv));
      argv->insert(argv->end(), I(iss), I());
    }
  };
};
} // namespace

arguments::arguments(int argc, char ** argv) {
  act.program = argv[0] ? argv[0] : "";
  act.program = act.program.substr(act.program.rfind('/') + 1);
#ifdef FLECSI_ENABLE_MPI
  dep.mpi.push_back(act.program);
#endif
#ifdef FLECSI_ENABLE_KOKKOS
  dep.kokkos.push_back(act.program);
#endif
  act.code = getopt(argc, argv); // also populates act.stderr
}

status
arguments::getopt(int argc, char ** argv) {
  po::options_description master("Basic Options");
  master.add_options()("help,h", "Print this message and exit.");

  // Add externally-defined descriptions to the main description
  auto & dm = context::descriptions_map();
  for(auto & od : dm) {
    if(od.first != "FleCSI Options") {
      master.add(od.second);
    } // if
  } // for

  po::options_description flecsi_desc =
    dm.count("FleCSI Options") ? dm["FleCSI Options"]
                               : po::options_description("FleCSI Options");

  backend_arg::argv = &cfg.backend;
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
        po::value(&cfg.flog.verbose)
          ->implicit_value(1)
          ->default_value(0),
        "Enable verbose output. Passing '-1' will strip any additional"
        " decorations added by flog and will only output the user's message."
      )
      (
        "flog-process",
        po::value(&cfg.flog.process)->default_value(0),
        "Restrict output to the specified process id. The default is process 0."
        " Use '--flog-process=-1' to enable all processes."
      ); // clang-format on
#endif

  // Make an options description to hold all options. This is useful
  // to exlude hidden options from help.
  po::options_description all("All Options");
  all.add(master);
  all.add(flecsi_desc);
  all.add(context::hidden_options());

  auto & pd = context::positional_description();
  po::parsed_options parsed = po::command_line_parser(argc, argv)
                                .options(all)
                                .positional(pd)
                                .allow_unregistered()
                                .run();

  struct guard : std::ostringstream {
    guard(std::string & s) : out(s) {}
    ~guard() {
      out = std::move(*this).str();
    }
    std::string & out;
  } stderr(act.stderr);
  auto usage = [&] {
    stderr << "Usage: " << act.program << " ";

    size_t positional_count = pd.max_total_count();
    size_t max_label_chars = std::numeric_limits<size_t>::min();

    for(size_t i{0}; i < positional_count; ++i) {
      stderr << "<" << pd.name_for_position(i) << "> ";

      const size_t size = pd.name_for_position(i).size();
      max_label_chars = size > max_label_chars ? size : max_label_chars;
    } // for

    max_label_chars += 2;

    stderr << "\n\n";

    if(positional_count) {
      stderr << "Positional Options:" << std::endl;

      for(size_t i{0}; i < pd.max_total_count(); ++i) {
        auto const & name = pd.name_for_position(i);
        auto help = context::positional_help().at(name);
        stderr << "  " << name << " ";
        stderr << std::string(max_label_chars - name.size() - 2, ' ');

        if(help.size() > 78 - max_label_chars) {
          std::string first = help.substr(
            0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
          stderr << first << std::endl;
          help = help.substr(first.size() + 1, help.size() - first.size() + 1);

          while(help.size() > 78 - max_label_chars) {
            std::string part = help.substr(
              0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
            stderr << std::string(max_label_chars + 1, ' ') << part
                   << std::endl;
            help = help.substr(part.size() + 1, help.size() - part.size());
          } // while

          stderr << std::string(max_label_chars + 1, ' ') << help << std::endl;
        }
        else {
          stderr << help << std::endl;
        } // if

      } // for

      stderr << std::endl;
    } // if

    stderr << master << std::endl;
    stderr << flecsi_desc << std::endl;

#if defined(FLECSI_ENABLE_FLOG)
    auto const & tm = flog::state::tag_map();

    if(tm.size()) {
      stderr << "Available FLOG Tags (FleCSI Logging Utility):" << std::endl;
    } // if

    for(auto t : tm) {
      stderr << "  " << t.first << std::endl;
    } // for
#endif
  };

  try {
    po::variables_map vm;
    po::store(parsed, vm);

    if(vm.count("help")) {
      usage();
      return status::help;
    } // if
    if(vm.count("control-model")) {
      return status::control_model;
    } // if
    if(vm.count("control-model-sorted")) {
      return status::control_model_sorted;
    } // if

    po::notify(vm);

#ifdef FLECSI_ENABLE_FLOG
    if(flog_tags_ != "none") {
      std::istringstream is(flog_tags_);
      std::string tag;
      while(std::getline(is, tag, ','))
        cfg.flog.tags.push_back(tag);
    }
#endif

    // Call option check methods
    auto & oc = context::option_checks();
    for(auto const & [name, boost_any] : vm) {
      auto const & ita = oc.find(name);
      if(ita != oc.end()) {
        auto [positional, check] = ita->second;

        std::stringstream ss;
        if(!check(boost_any.value(), ss)) {
          std::string dash = positional ? "" : "--";
          stderr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED
                 << "invalid argument for '" << dash << name << "' option!!!"
                 << std::endl
                 << FLOG_COLOR_LTRED << (ss.str().empty() ? "" : " => ")
                 << ss.str() << FLOG_COLOR_PLAIN << std::endl
                 << std::endl;

          usage();
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

    stderr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED << error << "!!!"
           << FLOG_COLOR_PLAIN << std::endl
           << std::endl;
    usage();
    return status::command_line_error;
  } // try

  unrecognized =
    po::collect_unrecognized(parsed.options, po::include_positional);

  return status::success;
}

} // namespace flecsi::run
