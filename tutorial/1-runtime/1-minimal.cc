#include <flecsi/execution.hh>
#include <flecsi/runtime.hh>

/*
  The top-level action can be any C/C++ function that takes no arguments and
  returns an int.

  In this simple example, we only print a message to indicate that the
  top-level action was actually executed by FleCSI. However, in a real
  application, the top-level action would execute FleCSI tasks and other
  functions to implement the simulation.
 */

int
top_level_action() {
  std::cout << "Hello World" << std::endl;
  return 0;
} // top_level_action

/*
  The main function must create a FleCSI \c runtime object.  Otherwise, the
  implementation of main is left to the user.
 */

int
main() {
  const flecsi::run::dependencies_guard dg;
  /*
    flecsi::run::call means to call the single function given as an argument to
    run.main.  (Multiple functions will be covered in later examples.)
    It will be skipped if, say, --help was passed as an argument; FleCSI's
    command-line support is documented in the next example.
   */
  return flecsi::runtime().control<flecsi::run::call>(top_level_action);
} // main
