#include <flecsi/execution.hh>
#include <flecsi/runtime.hh>

/*
  The control function can be any C/C++ function that takes a scheduler and
  returns an int.

  In this simple example, we only print a message to indicate that the
  function was actually executed by FleCSI.  In a real application, it would
  execute FleCSI tasks to implement the simulation.
 */

int
simulation(flecsi::scheduler &) {
  std::cout << "Hello World" << std::endl;
  return 0;
} // simulation

/*
  The main function must create a FleCSI \c runtime object.  Otherwise, the
  implementation of main is left to the user.
 */

int
main() {
  const flecsi::run::dependencies_guard dg;
  /*
    flecsi::run::call means to call the single function given as an argument.
   */
  return flecsi::runtime().control<flecsi::run::call>(simulation);
} // main
