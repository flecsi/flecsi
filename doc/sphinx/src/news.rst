Release Notes
*************

``flecsi::`` qualification is omitted throughout.

Changes in v2.1.1
+++++++++++++++++

Possible incompatibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build

  * ``FLECSI_RUNTIME_MODEL`` is renamed to ``FLECSI_BACKEND`` (it never referred to :ref:`user-guide/runtime`).

* Utilties

  * ``util::mdspan::operator()`` is removed (it had an inconsistent argument order).

Fixed
^^^^^
* Build

  * HDF5 is supported by the CMake configuration.
  * Caliper is supported by the MPI backend.

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
