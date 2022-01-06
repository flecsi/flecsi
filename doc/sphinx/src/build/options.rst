.. |br| raw:: html

   <br />

CMake Configuration Options
===========================

The following set of options are available to control how FleCSI is
built.

* **BUILD_SHARED_LIBS [default: ON]** |br|
  Build shared library objects (as opposed to static).

* **CMAKE_BUILD_TYPE [default: Debug]** |br|
  Specify the build type (configuration) statically for this build tree.
  Possible choices are *Debug*, *Release*, *RelWithDebInfo*, and
  *MinSizeRel*.

* **CMAKE_INSTALL_PREFIX [default: /usr/local]** |br|
  Specify the installation path to use when *make install* is invoked.

* **CXX_CONFORMANCE_STANDARD [default: c++14]** |br|
  Specify to which C++ standard a compiler must conform. This is a
  developer option used to identify whether or not the selected C++
  compiler will be able to compile FleCSI, and which (if any) tests it
  fails to compile. This information can be shared with vendors to
  identify features that are required by FleCSI that are not
  standards-compliant in the vendor's compiler.

* **ENABLE_BOOST_PREPROCESSOR [default: ON]** |br|
  Boost.Preprocessor is a header-only Boost library that provides
  enhanced pre-processor options and manipulation, which are not
  supported by the standard C preprocessor. Currently, FleCSI uses the
  preprocessor to implement type reflection.

* **ENABLE_COLORING [default: OFF]** |br|
  This option controls whether or not various library dependencies and
  code sections are active that are required for graph partitioning
  (coloring) and distributed-memory parallelism. In general, if you have
  selected a runtime mode that requires this option, it will
  automatically be enabled.

* **ENABLE_COLOR_UNIT_TESTS [default: OFF]** |br|
  Enable coloraization of unit test output.

* **ENABLE_COVERAGE_BUILD [default: OFF]** |br|
  Enable build mode to determine the code coverage of the current set of
  unit tests. This is useful for continuous integration (CI) test analysis.

* **ENABLE_DEVEL_TARGETS [default: OFF]** |br|
  Development targets allow developers to add small programs to the
  FleCSI source code for testing code while it is being developed. These
  programs are not intended to be used as unit tests, and may be added
  or removed as the code evolves.

* **ENABLE_DOCUMENTATION [default: OFF]** |br|
  This option controls whether or not the FleCSI user and developer
  guide documentation is built. If enabled, CMake will generate these
  guides as PDFs in the *doc* subdirectory of the build.

* **ENABLE_DOXYGEN [default: OFF]** |br|
  If enabled, CMake will verify that a suitable *doxygen* binary is
  available on the system, and will add a target for generating
  Doxygen-style interface documentation from the FleCSI source code.
  **To build the doxygen documentation, users must explicitly invoke:**

.. code-block:: console

  $ make doxygen

* **ENABLE_DOXYGEN_WARN [default: OFF]** |br|
  Normal Doxygen output produces many pages worth of warnings. These are
  distracting and overly verbose. As such, they are disabled by default.
  This options allows the user to turn them back on.

* **ENABLE_EXODUS [default: OFF]** |br|
  If enabled, CMake will verify that a suitable Exodus library is
  available on the system, and will enable Exodus functionality in the
  FleCSI I/O interface.

* **ENABLE_FLOG [default: OFF]** |br|
  This option enables support for the FleCSI logging utility (flog).
  By default, it also activates the flog unit tests in the build
  system.

* **ENABLE_JENKINS_OUTPUT [default: OFF]** |br|
  If this options is on, extra meta data will be output during unit test
  invocation that may be used by the Jenkins CI system.

* **ENABLE_MPI_CXX_BINDINGS [default: OFF]** |br|
  This option is a fall-back for codes that actually require the MPI C++
  bindings. **This interface is deprecated and should only be used if it
  is impossible to get rid of the dependency.**

* **ENABLE_OPENMP [default: OFF]** |br|
  Enable OpenMP support. If enabled, the appropriate flags will be
  passed to the C++ compiler to enable language support for OpenMP
  pragmas.

* **ENABLE_OPENSSL [default: OFF]** |br|
  If enabled, CMake will verify that a suitable OpenSSL library is
  available on the system, and will enable the FleCSI checksum
  interface.

* **ENABLE_UNIT_TESTS [default: OFF]** |br|
  Enable FleCSI unit tests. If enabled, the unit test suite can be run
  by invoking:

.. code-block:: console

  $ make test

* **FLECSI_COUNTER_TYPE [default: int32_t]** |br|
  Specify the C++ type to use for the FleCSI counter interface.

* **FLECSI_DBC_ACTION [default: throw]** |br|
  Select the design-by-contract action.

* **FLECSI_DBC_REQUIRE [default: ON]** |br|
  Enable DBC pre/post condition assertions.

* **FLECSI_BACKEND [default: mpi]** |br|
  Specify the backend to use. Currently, *legion* and *mpi* are
  the only valid options.

* **FLOG_SERIALIZATION_INTERVAL [default: 100]** |br|
  The flog serialization interval specifies the number of task
  executions after which FleCSI should check for buffered output to
  process.  It should be set to a value that balances output
  timeliness (lower = more timely output) against performance (higher
  = less overhead from the requisite global reduction).

* **FLOG_SERIALIZATION_THRESHOLD [default: 1024]** |br|
  The flog serialization threshold specifies the number of messages
  that must have accumulated before output will be collected to a
  single process and written to the output streams.  It should be set
  to a value that balances output timeliness (lower = more timely
  output) against performance (higher = less overhead from the
  requisite global reduction and from writing the output).

* **VERSION_CREATION [default: git describe]** |br|
  This options allows the user to either directly specify a version by
  entering it here, or to let the build system provide a version using
  git describe.

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
