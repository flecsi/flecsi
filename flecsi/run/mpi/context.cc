#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

using namespace boost::program_options;

namespace flecsi::run {

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action) {
  context::start();

  context::threads_per_process_ = 1;
  context::threads_ = context::processes_;

  std::vector<char *> largv;
  largv.push_back(argv_[0]);

  for(auto opt = unrecognized_options_.begin();
      opt != unrecognized_options_.end();
      ++opt) {
    largv.push_back(opt->data());
  } // for

  return detail::data_guard(), action(); // guard destroyed after action call
}

} // namespace flecsi::run
