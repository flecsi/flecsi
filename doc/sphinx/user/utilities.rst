Utilities
*********
FleCSI provides a number of utilities that are frequently helpful in writing actions, tasks, and specializations but are not specific to FleCSI data or topologies.

.. _ranges:

Multi-dimensional arrays
++++++++++++++++++++++++

FleCSI provides utilities for structured memory access: ``mdspan`` and ``mdcolex``.

.. note::

   ``mdspan`` and ``mdcolex`` are GPU-compatible and can be used inside ``forall`` and ``reduceall`` constructs.

These types allow applications to express access patterns for multidimensional data in a portable and readable manner without sacrificing performance.

The ``mdspan`` type (based on a subset of ``std::mdspan`` in C++23) extends the concept of ``std::span`` to support multidimensional indexing.
It provides a way to treat flat memory as an N-dimensional array, by specifying the extents in each dimension at construction.
``mdspan`` supports access using parentheses, which makes it much easier to write readable and structured computational code.

For example, with a 3D buffer:

.. code-block:: c++

   void mdspan_task(field<double>::accessor<ro> p) noexcept {
     int is = 3, js = 4, ks = 5;
     flecsi::util::mdspan<double, 3> mds(p.span(), {ks, js, is});
     for(int i = 0; i < is; ++i)
       for(int j = 0; j < js; ++j)
         for(int k = 0; k < ks; ++k)
           mds(i, j, k) = 1.0;
   }

This abstraction removes the need to compute manual offsets and makes scientific applications with logically multidimensional data easier to write and debug.

The ``mdcolex`` type is a variant of ``mdspan`` with reversed indices.
While mdspan uses a row-major layout, mdcolex reverses the index traversal order, producing column-major order.

For example:

.. code-block:: c++

   void mdcolex_task(field<double>::accessor<ro> p) noexcept {
     int is = 3, js = 4, ks = 5;
     flecsi::util::mdcolex<double, 3> mds(p.span(), {ks, js, is});
     for(int i = 0; i < is; ++i)
       for(int j = 0; j < js; ++j)
         for(int k = 0; k < ks; ++k)
           mds(i, j, k) = 1.0;
   }

Renumbering
+++++++++++

It is common that some subset of a prefix of the whole numbers needs to be treated, comprising perhaps the global indices of a partition of a mesh or the local indices of ghost index points.
``util::transform`` can be applied to a range of such indices (or structures containing them) to gather them into a range (that already has the correct size).
The indices into the output range can serve as local (or packed) IDs for the subset.
``util::partition_point`` can be applied to a sorted range of such indices or structures to efficiently find the position or data for a particular index.
``util::binary_index`` (which does not reflect a standard-library feature) is a convenience for the case of finding the position of an index in a range of just indices (*e.g.*, finding the local ID for a global ID).
``util::permutation_view`` represents the subset of a range selected by indices stored in another range.
