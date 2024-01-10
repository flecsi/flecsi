#include "flecsi/run/init.hh"
#include "flecsi/run/options.hh"
#include "flecsi/runtime.hh"

#include <string_view>

namespace po = boost::program_options;

namespace flecsi {
namespace run {
namespace {
struct backend_arg {
  static inline run::argv * argv;
  static const std::string & get(const std::vector<std::string> & v) {
    flog_assert(v.size() == 1, "wrong token count");
    return v.front();
  }
  struct single {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, single *, int) {
      if(argv)
        argv->push_back(get(sv));
    }
  };
  struct space {
    friend void
    validate(boost::any &, const std::vector<std::string> & sv, space *, int) {
      using I = std::istream_iterator<std::string>;
      std::istringstream iss(get(sv));
      if(argv)
        argv->insert(argv->end(), I(iss), I());
    }
  };
};

void
do_call(call_policy & p) {
  throw control_base::exception{p()};
}
call::action<do_call, call_policy::single> phone;

void
finalize() {
  run::context::ctx.reset();
  run::dependent.reset();
}
} // namespace
} // namespace run

getopt::getopt(run::config * init) {
  using run::context;

  if(init)
    basic.add_options()("help,h", "Print this message and exit.");

  // Add externally-defined descriptions to the main description
  for(auto & [n, od] : context::descriptions_map())
    basic.add(od);

  all.add(basic);
  if(init) {
    run::backend_arg::argv = init->backend();

    auto & flecsi_desc = flecsi.emplace("FleCSI Options");
    flecsi_desc.add_options()("backend-args",
      po::value<run::backend_arg::space>(),
      "Pass arguments to the backend. The single argument is a quoted "
      "string of backend-specific options.");

    flecsi_desc.add_options()("Xbackend",
      po::value<run::backend_arg::single>(),
      "Pass single argument to the backend. This option can be passed "
      "multiple times.");
#if defined(FLECSI_ENABLE_FLOG)
    // Add FleCSI options
    flecsi_desc.add_options() // clang-format off
      (
        "flog-tags",
        po::value<std::string>()
          ->default_value("all"),
        "Enable the specified output tags, e.g., --flog-tags=tag1,tag2."
        " Use '--flog-tags=all' to show all output, and "
        " '--flog-tags=unscoped' to show only unguarded output."
      )
      (
        "flog-verbose",
        po::value(&init->flog.verbose)
          ->implicit_value(1)
          ->default_value(0),
        "Enable verbose output. Passing '-1' will strip any additional"
        " decorations added by flog and will only output the user's message."
      )
      (
        "flog-process",
        po::value(&init->flog.process)->default_value(0),
        "Restrict output to the specified process id. The default is process 0."
        " Use '--flog-process=-1' to enable all processes."
      ); // clang-format on
#endif
    all.add(flecsi_desc);
  }
  all.add(context::hidden_options());
}

auto
getopt::parse(int argc, char ** argv) const {
  using run::context;
  po::parsed_options parsed = po::command_line_parser(argc, argv)
                                .options(all)
                                .positional(context::positional_description())
                                .run();
  po::variables_map vm;
  po::store(parsed, vm);
  po::notify(vm);

  // Call option check methods
  auto & oc = context::option_checks();
  for(auto const & [name, boost_any] : vm) {
    auto const & ita = oc.find(name);
    if(ita != oc.end()) {
      auto [positional, check] = ita->second;

      std::stringstream ss;
      if(!check(boost_any.value(), ss)) {
        std::ostringstream err;
        err << "invalid argument for '" << (positional ? "" : "--") << name
            << "' option";
        if(ss.rdbuf()->in_avail())
          err << " (" << FLOG_COLOR_LTRED << ss.rdbuf() << FLOG_COLOR_RED
              << ')';
        throw po::invalid_option_value(std::move(err).str());
      } // if
    } // if
  } // for

  return vm;
}
void
getopt::operator()(int argc, char ** argv) const {
  parse(argc, argv);
}

std::string
getopt::usage(std::string_view p) const {
  std::ostringstream ret;
  ret << "Usage: " << p << ' ';

  auto & pd = run::context::positional_description();
  size_t positional_count = pd.max_total_count();
  size_t max_label_chars = std::numeric_limits<size_t>::min();

  for(size_t i{0}; i < positional_count; ++i) {
    ret << '<' << pd.name_for_position(i) << "> ";

    const size_t size = pd.name_for_position(i).size();
    max_label_chars = size > max_label_chars ? size : max_label_chars;
  } // for

  max_label_chars += 2;

  ret << "\n\n";

  if(positional_count) {
    ret << "Positional Options:\n";

    for(size_t i{0}; i < pd.max_total_count(); ++i) {
      auto const & name = pd.name_for_position(i);
      auto help = run::context::positional_help().at(name);
      ret << "  " << name << ' ';
      ret << std::string(max_label_chars - name.size() - 2, ' ');

      if(help.size() > 78 - max_label_chars) {
        std::string first = help.substr(
          0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
        ret << first << '\n';
        help = help.substr(first.size() + 1, help.size() - first.size() + 1);

        while(help.size() > 78 - max_label_chars) {
          std::string part = help.substr(
            0, help.substr(0, 78 - max_label_chars).find_last_of(' '));
          ret << std::string(max_label_chars + 1, ' ') << part << '\n';
          help = help.substr(part.size() + 1, help.size() - part.size());
        } // while

        ret << std::string(max_label_chars + 1, ' ') << help << '\n';
      }
      else {
        ret << help << '\n';
      } // if

    } // for

    ret << '\n';
  } // if

  ret << basic << '\n';
  if(flecsi) {
    ret << *flecsi << '\n';

#if defined(FLECSI_ENABLE_FLOG)
    auto const & tm = flog::state::tag_map();

    if(tm.size()) {
      ret << "Available FLOG Tags (FleCSI Logging Utility):\n";
    } // if

    for(auto t : tm) {
      ret << "  " << t.first << '\n';
    } // for
#endif
  }

  return std::move(ret).str();
}

int
initialize(int argc, char ** argv, bool dependent) {
  {
    std::string_view p;
    if(argv[0])
      p = argv[0];
    argv0 = p.substr(p.rfind('/') + 1);
  }

  run::config cfg{};
  if(auto * const p = cfg.backend())
    p->push_back(argv0);

  const getopt go(&cfg);

  auto ret = run::success;
  std::ostringstream stderr;
  const auto usage = [&](run::status s) {
    stderr << go.usage(argv0);
    ret = s;
  };

  try {
    const auto vm = go.parse(argc, argv);
#ifdef FLECSI_ENABLE_FLOG
    if(const auto flog_tags_ = vm["flog-tags"].as<std::string>();
       flog_tags_ != "none") {
      std::istringstream is(flog_tags_);
      std::string tag;
      while(std::getline(is, tag, ','))
        cfg.flog.tags.push_back(tag);
    }
#endif
    if(vm.count("help"))
      usage(run::help);
    else if(vm.count("control-model") && vm["control-model"].as<bool>())
      ret = run::control_model;
    else if(vm.count("control-model-sorted") &&
            vm["control-model-sorted"].as<bool>())
      ret = run::control_model_sorted;
  }
  catch(po::error & e) {
    std::string error(e.what());

    stderr << FLOG_COLOR_LTRED << "ERROR: " << FLOG_COLOR_RED << error << "!!!"
           << FLOG_COLOR_PLAIN << std::endl
           << std::endl;
    usage(run::command_line_error);
  } // try

  const auto make = [](auto & o, auto & x) -> auto & {
    flog_assert(!o, "already initialized");
    return o.emplace(x);
  };
  if(dependent) {
    run::dependencies_config dep;
    dep.mpi.push_back(argv0);
    make(run::dependent, dep);
  }
  auto & ctx = make(run::context::ctx, cfg);
#if defined(FLECSI_ENABLE_FLOG) && defined(FLOG_ENABLE_MPI)
  {
    const Color p = flog::state::instance().source_process();
    if(p != flog::state::all_processes && p >= ctx.processes()) {
      stderr << argv0 << ": flog process " << p << " does not exist with "
             << ctx.processes() << " processes\n";
      ret = run::command_line_error;
    }
  }
#endif
  if(ret) {
    if(!ctx.process())
      std::cerr << stderr.rdbuf();
    run::finalize();
  }
  return ret;
}

void
finalize() {
  run::finalize();
}

} // namespace flecsi
