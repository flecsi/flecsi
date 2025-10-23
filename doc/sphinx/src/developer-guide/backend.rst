Backend Development
+++++++++++++++++++

A FleCSI backend must implement a number of `entry points` (see :doc:`summary`) for invocation by the rest of FleCSI—and in a few cases, directly by application code.
Almost all of these entry points can be identified by a

.. code-block:: cpp

   #ifdef DOXYGEN // implemented per-backend

block in the source code that illustrates the interface required from each backend implementation.

For backend-developer convenience, the set of entry points is aggregated below, sorted by source file.

* ``data/topology.hh``

  * ``flecsi::data::region_base``

  * ``flecsi::data::partition``

  * ``flecsi::data::rows``

  * ``flecsi::data::borrow``

* ``data/copy.hh``

  * ``flecsi::data::prefixes``

  * ``flecsi::data::intervals``

  * ``flecsi::data::copy_engine``

* ``exec/bind_parameters.hh``

  * ``flecsi::exec::bind_accessors``

* ``exec/future.hh``

  * ``flecsi::future<R, single>`` (exposed to applications)

  * ``flecsi::future<R, index>`` (exposed to applications)

  * ``flecsi::make_future``

* ``exec/prolog.hh:``

  * ``flecsi::exec::task_prologue``

* ``execution.hh``

  * ``flecsi::exec::trace`` (exposed to applications)

  * ``flecsi::task_local`` (exposed to applications)

* ``io.hh``

  * ``flecsi::io::io_interface``

* ``run/context.hh``

  * ``flecsi::run::config`` (exposed to applications)

  * ``flecsi::run::dependencies_guard`` (exposed to applications)

  * ``flecsi::run::context_t`` (``#ifdef DOXYGEN`` lies within ``flecsi::run::context``)

* No ``#ifdef DOXYGEN``

  * ``flecsi::data::logical_size``

  * ``flecsi::exec::reduce_internal`` (supports ``flecsi::reduce``)

Definitions expected to be common to multiple backends can be found in the ``flecsi/data/local`` and ``flecsi/run/local`` directories.
Backends are free to include the header files found there but may reimplement the code entirely if meaningful to do so.
