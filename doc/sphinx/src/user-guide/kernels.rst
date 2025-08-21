On-node Parallelism
*******************


FleCSI tasks can launch *kernels* to exploit fine-grained, on-node parallelism.
These kernels operate inside the task body and are typically mapped to hardware threads by Kokkos.

FleCSI provides a unified API for launching kernels through an *executor*, using constructs like ``forall`` and ``reduceall``.

Simple Parallel Loop
++++++++++++++++++++

FleCSI provides a convenient API to iterate over data using the ``forall`` construct.
This construct is based on ``Kokkos::parallel_for`` and enables efficient, parallel execution over a range of elements.

Consider the example below, where each value in the accessor are incremented:

.. code-block:: c++

   void modify(exec::accelerator s,
     mesh::accessor<ro> t,
     field<double>::accessor<rw> p) noexcept {
       s.executor().forall(c, t.cells()) {
         p[c] += 1;
       };
   }

In this snippet, the loop index ``c`` is implicitly declared by the ``forall`` construct.
The iteration is over the range ``t.cells()``, which provides access to the mesh entities visible to the task.
The loop body is a lambda expression that must be followed by a semicolon.

Advanced Ranges
^^^^^^^^^^^^^^^

FleCSI also supports iterating over an :ref:`mdspan <ranges>` using ``mdiota_view``:

.. code-block:: c++

   s.executor().forall(
     mi,
     mdiota_view(
       md,
       exec::full_range,
       exec::prefix_range{2}
     )) { /* ... */ };

A subset of the array can be selected using parameters such as ``full_range``, ``prefix_range``, or ``sub_range``.

Reductions Inside Kernels
+++++++++++++++++++++++++

To obtain a reduced value from a parallel iteration a task uses the ``reduceall`` construct (based on ``Kokkos::parallel_reduce``).

For instance, the following kernel computes the maximum value over all cells:

.. code-block:: c++

   void reduce1(exec::accelerator s,
     mesh::accessor<ro> t,
     field<float>::accessor<ro> p) noexcept {
       auto res = s.executor().reduceall(
         c, up, t.cells(), exec::fold::prod, double) {
           up(p[c]);
       };
   }

Here, ``up`` is an additional variable declared by ``reduceall`` to collect values using the given type ``double`` from the kernel, while ``exec::fold::max`` defines the :ref:`fold operation <reduction-folds>`.

Further executor features
+++++++++++++++++++++++++

Executors support additional member functions for special cases.
Some of these specify optional, composable behavior for a kernel launch:

.. code-block:: c++

   void modify(exec::accelerator s,
               mesh::accessor<ro> t,
               field<double>::accessor<rw> p) noexcept {
     s.executor()
       .named("named forall")
       .threads<64, 1>()
       .for_each(t.cells(),
         [p] FLECSI_INLINE_TARGET (auto c) {
           p[c] += 2;
         });
   }

In this example, ``named("named forall")`` attaches a label to the kernel which can be used by Kokkos Tools for profiling and debugging.
``threads<64, 1>()`` specifies the number of threads and blocks for the construct, here one block of 64 threads.
``for_each`` is a function template equivalent of ``forall`` that directly accepts a function object.
``reduce`` is the equivalent for ``reduceall``; function objects used with these must be declared with ``FLECSI_INLINE_TARGET`` for compatibility with typical GPU compilers.
