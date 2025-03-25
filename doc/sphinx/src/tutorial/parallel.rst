.. |br| raw:: html

   <br />


Distributed and shared memory parallelism
*****************************************

FleCSI provides two different levels of parallelism: distributed memory
parallelism and shared memory parallelism.

Distributed memory parallelism is provided through topology coloring and
distribution of the data between different processes (shards). 
FleCSI provides macros *forall* and *reduceall* for shared memory
parallelism. Currently, it uses Kokkos programing model.

----

Shared memory
*************

Example 1: forall macro / parallel_for interface
++++++++++++++++++++++++++++++++++++++++++++++++
This example is a modification of the data-dense tutorial example that replaces the data copy with a ``modify`` task that uses the ``forall`` macro.
The task is executed with a second template parameter to ``execute``, which is a *processor_type*
with *loc* (latency optimized core) as a default value.
*default_accelerator* is a processor type that corresponds to Kokkos
default execution space. For example, if Kokkos is built with Cuda and
Serial, Cuda will be a default execution space or *toc* (throughput
optimized core) *processor type* in FleCSI.

.. note::

   With the Legion backend, OpenMP task execution can be improved with the ``omp`` processor type.
   Legion knows to assign an entire node to such a task.

.. warning::

   With the MPI backend, running one process per node with ``toc`` tasks or one
   process per core with ``omp`` tasks likely leads to poor performance.

.. literalinclude:: ../../../../tutorial/5-parallel/1-forall.cc
  :language: cpp

Example 2: reduceall macro / parallel_reduce interface
++++++++++++++++++++++++++++++++++++++++++++++++++++++
This example instead uses ``reduce1`` and ``reduce2`` tasks that use the ``reduceall`` macro interface and ``parallel_reduce`` function template interface respectively.
The former accepts two names declared for use in the body: the range element, as for ``forall``, and a function that accepts values for the reduction.
The latter supports further composition, such as client library interfaces that accept kernel functors.

.. literalinclude:: ../../../../tutorial/5-parallel/2-reduceall.cc
  :language: cpp

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
