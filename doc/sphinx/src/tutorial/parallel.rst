.. |br| raw:: html

   <br />

.. _TUT-DM:

Distributed and shared memory parallelism 
**********

----

FleCSI provides two different levels of parallelism: distributed memory
parallelism and shared memory parallelism.

Distributed memory parallelism is provided through topology coloring and
distribution of the data between different processes (shards). 
FleCSI provides macros *forall* and *reduceall* for shared memory
parallelism. Currently, it uses Kokkos programing model.

----
Shared memory
*****

Example 1: forall macro / parallel_for interface
++++++++++++++++++++++++++++++++++++++++++++++++

This example is an extension to the data-dense tutorial example with the
only difference of an additional "modify1" and "modify2" tasks that use
*forall* macro / *parallel_for* interface. Both "modify" tasks are executed on the FleCSI default accelerator.  

Second template parameter to the execute function is a *processor_type*
with *loc* (latency optimized core) as a default value.
*default_accelerator* is a processor type that corresponds to Kokkos
default execution space. For example, if Kokkos is bult with Cuda and
Serial, Cuda will be a default execution space or *toc* (throughput
optimized core) *processor type* in FleCSI. 

.. literalinclude:: ../../../../tutorial/5-parallel/1-forall.cc
  :language: cpp

Example 2: reduceall macro / parallel_reduce interface
++++++++++++++++++++++++++++++++++++++++++++++++

This example is an extension to the data-dense tutorial example with the
only difference of an additional "reduce1" and "reduce2" tasks that use
*reduceall* macro / *parallel_reduce* interface. Both "modify" tasks are
executed on the FleCSI default accelerator.

.. literalinclude:: ../../../../tutorial/5-parallel/2-reduceall.cc
  :language: cpp

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
