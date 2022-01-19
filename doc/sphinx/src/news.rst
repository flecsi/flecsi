.. include:: <isopub.txt>

.. _news:

Release Notes
*************

``flecsi::`` qualification is omitted throughout.

.. Each release has as sections some subsequence of Removed, Other
   incompatibilities, Possible incompatibilities, Deprecated, New features,
   and Fixed.

   Each such section discusses some subsequence of Build, Runtime, Data,
   Execution, Topologies, Legion backend, MPI backend, On-node parallelism,
   Utilities, and Logging.

Changes in v2.1.1
+++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build

  * ``FLECSI_RUNTIME_MODEL`` is renamed to ``FLECSI_BACKEND`` (it never referred to :ref:`runtime`).

* Utilties

  * ``util::mdspan::operator()`` is removed (it had an inconsistent argument order).

Fixed
^^^^^
* Build

  * HDF5 is supported by the CMake configuration.
  * Caliper is supported by the MPI backend.
  * Building with the MPI backend properly links to the threading library.

* Runtime

  * Certain control-flow graphs compile with Graphviz support and are drawn correctly.

* Data

  * Mutators support write-only privileges (which, as with accessors, are necessary to properly initialize fields).

* Topologies

  * Topology accessors that provide metadata for direct access in a ``toc`` task work properly when writable (although without such direct access in that case).
  * ``narray`` topology accessors work properly when declared ``const``.

* Legion backend

  * Certain user-level fields are allocated properly.
  * Errors at process termination in certain configurations are avoided.

* MPI backend

  * ``exec::fold::min`` and ``exec::fold::max`` work with ``bool`` (but may be slower than using another type like ``int``).
  * Index futures work properly when declared ``const``.

* On-node parallelism

  * Several uses of "iterator" in documentation have been corrected to use "range".

* Logging

  * Messages are sorted by timestamp correctly.

* Testing

  * String-comparison macros (*e.g.*, ``ASSERT_STREQ``) handle runtime values properly.

Changes in v2.1.0 (April 16 2021)
+++++++++++++++++++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Topologies

  * ``narray`` topology accessors provide ``policy_meta_`` rather than ``meta()``.

New features
^^^^^^^^^^^^
* Topologies

  * ``topo::help`` is a convenient base class for specialization class templates that defines the non-dependent names in ``specialization``.

Fixed
^^^^^
* Topologies

  * ``narray`` topology accessors handle boundary conditions correctly.

Changes in v2.0.0 (March 30 2021)
+++++++++++++++++++++++++++++++++

The changes between versions 1 and 2 are extensive and so are merely summarized here.
The broadest change is that macros are used only for on-node parallelism constructs and for logging and unit tests.
Some macro arguments become template arguments.

Removed
^^^^^^^
* Data

  * ``flecsi_get_client_handle`` |mdash| use the topology slot directly
  * ``flecsi_get_handles``, ``flecsi_get_handles_all``, ``flecsi_is_at``, ``flecsi_has_attribute_at``, ``flecsi_has_attribute`` |mdash| will be provided differently by ``io``

* Execution

  * ``flecsi_register_task``, ``flecsi_register_task_simple``, ``flecsi_register_mpi_task``, ``flecsi_register_mpi_task_simple`` |mdash| unneeded
  * ``specialization_tlt_init``, ``specialization_spmd_init``, ``execution::driver`` |mdash| the application provides ``main``
  * ``flecsi_execute_task_simple`` |mdash| namespaces require no special treatment
  * ``flecsi_register_program`` |mdash| unneeded
  * ``flecsi_register_global_object``, ``flecsi_set_global_object``, ``flecsi_initialize_global_object``, ``flecsi_get_global_object`` |mdash| use non-local variables directly, with the same restrictions
  * ``flecsi_register_reduction_operation`` |mdash| unneeded

* Topologies

  * Topology element ``id()`` |mdash| iteration produces indices
  * Automatic dependent connectivity |mdash| will be provided as a separate utility

* I/O |mdash| will be reimplemented

Other incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^
* Data

  * ``flecsi_register_data_client`` |mdash| now topology slots
  * ``flecsi_register_field``, ``flecsi_register_global``, ``flecsi_register_color`` |mdash| now ``field::definition``, or a container of same to implement multiple versions
  * ``flecsi_get_handle``, ``flecsi_get_global``, ``flecsi_get_color``, ``flecsi_get_mutator`` |mdash| now ``definition::operator()``
  * ``data_client_handle_u<T,P>`` |mdash| now ``T::accessor<P>``
  * ``dense_accessor``, ``ragged_accessor``, ``ragged_mutator``, ``sparse_accessor``, ``sparse_mutator`` |mdash| now ``definition::accessor`` or ``definition::mutator``

    * ``ragged`` fields have a ``std::vector``-like interface at each index point
    * ``sparse`` fields have a ``std::map``-like interface at each index point

* Execution

  * ``flecsi_execute_task`` |mdash| now ``execute``
  * ``flecsi_execute_mpi_task``, ``flecsi_execute_mpi_task_simple`` |mdash| pass ``mpi`` to ``execute``
  * ``flecsi_execute_reduction_task`` |mdash| now ``reduce``
  * ``flecsi_color()``, ``flecsi_colors()`` |mdash| now ``color()`` and ``colors()``
  * ``flecsi_register_function``, ``flecsi_execute_function``, ``flecsi_function_handle``, ``flecsi_define_function_type`` |mdash| now ``exec::make_partial``
  * ``execution::flecsi_future`` |mdash| now ``future``, with a simpler interface

* Topologies

  * Specializations inherit from ``topo::specialization`` rather than from a core topology type directly.

New features
^^^^^^^^^^^^
* Runtime

  * Application command-line options can be specified using the ``program_option`` class template and associated helper functions.
  * The ``control`` class template is used to define an extensible :ref:`control-model` for an application.

* Data

  * The ``raw`` and ``single`` layouts support special cases of field usage.

* Execution

  * Tasks can be (specializations of) function templates.
  * Tasks can accept parameters of dynamic size (*e.g.*, ``std::vector<double>``), although passing large objects is inefficient.
  * Index launches can be of any number of point tasks (determined by the task's arguments, including ``exec::launch_domain``).
  * ``test`` conveniently executes tasks for unit testing.

* Topologies

  * Multiple topology categories are supported: ``unstructured`` and ``narray`` as well as the special cases of ``index`` and ``global``.
  * Multiple user-defined topology instances may exist sequentially or simultaneously.
