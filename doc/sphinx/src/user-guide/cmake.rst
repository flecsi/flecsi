CMake for FleCSI client applications
####################################

The FleCSI installation provides mutliple CMake files to support
writing new software.  While the ``FleCSI`` CMake package is used to
build software on top of FleCSI, the additional CMake files
provide common CMake code and macros for adding dependencies, building
documentation, unit-testing and other utilities that might be useful
for clients.

The CMake package is installed either when you build and install
FleCSI manually via CMake to ``CMAKE_INSTALL_PREFIX`` or when using a
Spack installation.

If you installed FleCSI in a Spack environment the necessary
environment variables will be automatically set.

After manual compilation and installation of FleCSI via CMake, you will
have to prepend the installation prefix to the ``CMAKE_PREFIX_PATH``
environment variable to make it available.

.. code-block:: bash

   export CMAKE_PREFIX_PATH=/path/to/your/flecsi/install:$CMAKE_PREFIX_PATH

---------------------------------------------------------------------------

FleCSI CMake files
==================

FleCSI-based applications often need common CMake code to enable
features and/or add dependencies. The ``FleCSI`` package also exposes
utility CMake files that are meant to be included in new projects. It
does so by adding its installation parent folder to the CMake's search
path when you add the package:

.. code-block:: cmake

   find_package(FleCSI REQUIRED)

Once added, you can ``include`` the CMake files provided by this
package.

For example, if you want to use the provided ``documentation.cmake``,
you can add it as follows:

.. code-block:: cmake

   include(FleCSI/documentation)

Documentation
-------------

FleCSI uses both Sphinx and Doxygen for its documentation system. The
following CMake files can be included to utilize the same
documentation system for your own projects.

``FleCSI/documentation``
   Adds the ``flecsi_set_doc_target_name(target)`` macro, which is
   used to define the name of a generic documentation target
   ``FLECSI_DOC_TARGET``. This target collects and runs all other
   later defined documentation targets. If no name is set, it defaults
   to ``doc``.  This target itself is created during the first call of
   either ``flecsi_add_doxygen_target`` or
   ``flecsi_add_sphinx_target``.

   This file also adds the ``flecsi_add_doc_deployment()`` function.
   It takes two parameters: the target name and ``GITHUB_PAGES_REPO``
   which is a Git repository URL. Running the target checks out the
   ``gh-pages`` branch of that repository, clears it and puts the
   result of all documentation targets into it.  Files are only added,
   but **not** commited or pushed. These are left as manual steps.

``FleCSI/sphinx``
  Adds the ``flecsi_set_sphinx_target_name`` macro, which is used to
  define the name of a generic Sphinx target ``FLECSI_SPHINX_TARGET``.
  This target collects and runs all other later defined Sphinx
  targets. If no name is set, it defaults to ``sphinx``. The target
  itself is created during the first call of
  ``flecsi_add_sphinx_target``.

  Sphinx builds are defined with the ``flecsi_add_sphinx_target``
  function. It takes three parameters: the target name, which will be
  prefixed with ``${FLECSI_SPHINX_TARGET}-``, a ``CONFIG`` directory, and
  ``OUTPUT`` directory. The provided ``CONFIG`` directory must contain
  a ``conf.py.in`` file. This template will be processed with
  ``configure_file`` to produce the final Sphinx configuration file
  ``conf.py``. This configuration file as well as the ``_static`` and
  ``_templates`` directories will be copied to the ``OUTPUT``
  directory, where the target will be built.

  Usage:

  .. code-block:: cmake

     include(FleCSI/documentation)
     include(FleCSI/sphinx)

     flecsi_set_doc_target_name(mydoc)
     flecsi_set_sphinx_target_name(mysphinx)

     flecsi_add_sphinx_target(main              # custom target name
       CONFIG ${CMAKE_SOURCE_DIR}/doc/sphinx    # folder containing conf.py.in
       OUTPUT ${CMAKE_BINARY_DIR}/doc           # output directory
     )

     flecsi_add_doc_deployment(deploy-docs
                               GITHUB_PAGES_REPO git@github.com:flecsi/flecsi.git)

     # the following targets will be defined:
     # - mydoc
     # - mysphinx
     # - mysphinx-main
     # - deploy-docs

``FleCSI/doxygen``
  Adds the ``flecsi_set_doxygen_target_name`` macro, which is used to
  define the name of a generic Doxygen target
  ``FLECSI_DOXYGEN_TARGET``. This target collects and runs all other
  later defined Doxygen targets. If no name is set, it defaults to
  ``doxygen``. The target itself is created during the first call of
  ``flecsi_add_doxygen_target``.

  Doxygen builds are defined with the ``flecsi_add_doxygen_target``
  function.  It takes two parameters: the target name, which will be
  prefixed with ``${FLECSI_DOXYGEN_TARGET}-``, and one or more file names in
  ``CONFIGS``. Each of the configuration files will be processed with
  ``configure_file``. The final output location is controlled by what
  is defined in these configuration files. E.g. by defining
  ``OUTPUT_DIRECTORY`` relative to the ``CMAKE_BINARY_DIR``:

  .. code-block:: ini

     OUTPUT_DIRECTORY = @CMAKE_BINARY_DIR@/doc/api

  See ``doc/doxygen/conf.in`` in the FleCSI source repository as an
  example configuration.

Coverage and Unit Testing
-------------------------

FleCSI uses its own unit-testing framework and installs the necessary
CMake files to allow using it in your own applications.

``FleCSI/coverage``
  Adds the ``flecsi_enable_coverage`` macro, which adds compiler and
  linker flags to enable capturing coverage information.

``FleCSI/unit``
  Adds the ``flecsi_enable_testing`` macro, which turns on CMake's
  testing capabilities through ``ctest`` and defines a ``test``
  target.

  While you can define your own test executables manually with
  `add_test
  <https://cmake.org/cmake/help/latest/command/add_test.html>`_, this
  CMake file also defines its own ``flecsi_add_test`` function for
  writing tests based on FleCSI Unit Test framework.

  .. code-block:: cmake

     flecsi_add_test(test-name                           # name of target
                     SOURCES src1 src2 ... srcN          # list of source files
	             INPUTS in1 in2 ... inN              # list of input files
	             LIBRARIES lib1 lib2 ... libN        # libraries linked to test target
	             DEFINES define1 define2 ... defineN # defines added to test target
	             ARGUMENTS  arg1 arg2 ... argN       # command arguments
	             TESTLABELS label1 label2 ... labelN # labels added to test target
	             PROCS nprocs1 nprocs2 ... nprocsN   # number(s) of MPI processes
	            )

  ``flecsi_add_test`` will take the sources files in ``SOURCES`` and
  compile them together with a predefined ``main()`` function. It will
  also link to any ``LIBRARIES`` and add ``DEFINES`` as compile
  definitions.

  If the test uses input files, they can be specified as
  ``INPUTS``. This ensures they are copied to the execution folder.

  Command-line arguments are passed via the ``ARGUMENTS`` option. You
  can also control the number of MPI processes with ``PROCS``. If you
  provide more than one value in ``PROCS``, this will define one
  target per value with a name ``<target-name>_<value>``.

  .. note::

     Targets added with ``flecsi_add_test`` will be run with GPU
     support if appropriate.


  ``TESTLABELS`` can be added to your test to allow filtering based on
  label when using ``ctest``.

  **Usage:**

  Here is a minimal unit test file ``mytest.cc``:

  .. code-block:: cpp

     #include <flecsi/util/unit.hh>

     int mytest_driver() {
       UNIT() {
         ASSERT_TRUE(true);
       };
     } // mytest_driver

     flecsi::unit::driver<mytest_driver> driver;

  Which can be compiled with the following ``CMakeLists.txt``:

  .. code-block:: cmake

     cmake_minimum_required(VERSION 3.20)
     project(myproject LANGUAGES CXX C)

     set(CXX_STANDARD_REQUIRED ON)
     set(CMAKE_CXX_STANDARD 17)

     find_package(FleCSI REQUIRED)

     include(FleCSI/unit)
     flecsi_enable_testing()

     flecsi_add_test(mytest
                     SOURCES mytest.cc)

  To configure and compile:

  .. code-block:: console

     mkdir build
     cd build
     cmake ..
     make

  Once compiled, you can run the tests with:

  .. code-block:: console

     make test
     # OR
     ctest


Code Formatting
---------------

``FleCSI/format``
  Adds the ``flecsi_enable_format`` macro, which takes a required
  ``clang-format`` version as parameter.

  It defines a ``format`` target that depends on both ``git`` and
  ``clang-format`` to be present. It also requires the source tree to
  be a Git checkout. Running this target will find all ``.hh`` and
  ``.cc`` files and apply the style defined in the project's
  ``.clang-format``.


Dependencies
------------

Some projects might want to explicitly link to dependencies that
FleCSI uses itself. External libraries used by FleCSI are added via
their own CMake file and the macros they define.

The general structure in these files is that they add a
``flecsi_enable_<PACKAGE>`` macro, which adds the necessary defines,
include folders and libraries to a given target.

* ``FleCSI/hdf5``
* ``FleCSI/hpx``
* ``FleCSI/kokkos``
* ``FleCSI/legion``
* ``FleCSI/mpi``
* ``FleCSI/openmp``
* ``FleCSI/parmetis``
* ``FleCSI/boost``
* ``FleCSI/caliper``

Other files
-----------

``FleCSI/colors``
  Defines several ASCII color codes for colored console output.

  .. hlist::
     :columns: 3

     * ``FLECSI_ColorReset``
     * ``FLECSI_ColorBold``
     * ``FLECSI_Red``
     * ``FLECSI_Green``
     * ``FLECSI_Yellow``
     * ``FLECSI_Brown``
     * ``FLECSI_Blue``
     * ``FLECSI_Magenta``
     * ``FLECSI_Cyan``
     * ``FLECSI_White``
     * ``FLECSI_BoldGrey``
     * ``FLECSI_BoldRed``
     * ``FLECSI_BoldGreen``
     * ``FLECSI_BoldYellow``
     * ``FLECSI_BoldBlue``
     * ``FLECSI_BoldMagenta``
     * ``FLECSI_BoldCyan``
     * ``FLECSI_BoldWhite``

``FleCSI/summary``
  Defines multiple macros to generate a (colored) configuration
  summary. Each of these macros appends to the global ``_summary``.
  At the end of your CMake file you can then print this summary using
  ``message(STATUS ${_summary})``.

  ``flecsi_summary_header`` will add a header.

  ``flecsi_summary_info(name info allow_split)`` will take a given
  ``name`` and add its value ``info`` next to it. If ``info`` is a
  space-separated list of values, ``allow_split`` controls if each
  value should be in its own line.

  ``flecsi_summary_option(name state extra)`` is used for adding
  Boolean values to the summary. If ``state`` evaluates to ``TRUE``
  the option state will be shown in a bright green color, followed by
  what is in ``extra``. Otherwise, the ``state`` will be shown in
  gray.
