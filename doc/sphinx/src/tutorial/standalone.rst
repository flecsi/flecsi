.. |br| raw:: html

   <br />

Standalone
**********

This example is designed to be used as a template for creating
FleCSI-based application codes and is intended as the last example in
the tutorial. As such, this example provides the following components:

* A basic CMake build system.

* A simple control policy.

* A standard FleCSI-based *main* function.

We discuss each of these individually. However, in general, to use this
example as a template for a real application, you should just change all
occurrances of ``poisson`` to whatever namespace name you would like to
use for your project.

Build System
++++++++++++

The build system uses standard CMake and is entirely defined in
CMakeLists.txt:

.. literalinclude:: ../../../../tutorial/standalone/poisson/CMakeLists.txt
   :language: cmake

To prepare this file for your project, you should do the following:

* Change ``poisson`` to the name of your project wherever it occurs.

* Update and add source files to the project.

* Add any dependencies.

For the most part, if you wish to extend this tutorial in any way, you
will need a working knowledge of CMake. Documentation for CMake is
`here`__.

__ https://cmake.org/documentation

Control Policy
++++++++++++++

The control policy for this example is located in
*specialization/control.hh*. This implementation is consistent with the
examples in :ref:`TUT-CM` Tutorial.

.. figure:: images/standalone.png
   :align: center

   Control Policy for Stand-Alone Application.

Main Function
+++++++++++++

The *main* function for this example is located in ``poisson.cc``.
Unless you need to initialize additional runtimes that are not handled
internally by FleCSI, you can likely use this file as-is (with a different namespace name).

.. literalinclude:: ../../../../tutorial/standalone/poisson/app/poisson.cc
   :language: cpp

Building the Stand-Alone Example
++++++++++++++++++++++++++++++++

:ref:`build` FleCSI somewhere on your system and make sure
that the location is set in your *CMAKE_PREFIX_PATH* environement
variable. Then, you can build this example like:

.. code-block:: console

  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
