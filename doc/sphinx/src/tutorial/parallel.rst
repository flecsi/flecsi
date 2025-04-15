.. |br| raw:: html

   <br />


Shared-memory parallelism
*************************

FleCSI provides two different levels of parallelism: distributed memory
parallelism and shared memory parallelism.

Distributed memory parallelism is provided through topology coloring and
distribution of the data between different processes (shards). 
FleCSI provides macros *forall* and *reduceall* for shared memory
parallelism. Currently, it uses Kokkos programing model.

----

Example 1: forall
+++++++++++++++++
This example is a modification of the data-dense tutorial example that replaces the data copy with a ``modify`` task that supports Kokkos.
The ``accelerator`` is an execution space that uses Kokkos parallelism on a GPU or via OpenMP if available.
Every execution space has an ``executor`` that implements its parallelism (if any) via a ``forall`` macro that can be used as a member function.

.. note::

   With the Legion backend, OpenMP task execution can be improved with the ``omp`` processor type.
   Legion knows to assign an entire node to such a task.

.. warning::

   With the MPI backend, running one process per node with ``toc`` tasks or one
   process per core with ``omp`` tasks likely leads to poor performance.

.. literalinclude:: ../../../../tutorial/5-parallel/1-forall.cc
  :language: cpp

Example 2: reduceall
++++++++++++++++++++
This example instead uses ``reduce1`` and ``reduce2`` tasks that use the ``reduceall`` macro interface and ``reduce`` function template interface respectively.
The former accepts two names declared for use in the body: the range element, as for ``forall``, and a function that accepts values for the reduction.
The latter supports further composition, such as client library interfaces that accept kernel functors; its analog for ``forall`` is called ``for_each``.
Any of these can have a name attached as illustrated with the ``named`` function.

.. literalinclude:: ../../../../tutorial/5-parallel/2-reduceall.cc
  :language: cpp

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
