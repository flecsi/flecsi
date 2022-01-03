.. |br| raw:: html

   <br />

.. _TUT-EM:

Execution Model
***************

FleCSI has two mechanisms for expressing work:

* **Tasks** |br|
  Tasks operate on data distributed to one or more address spaces, and
  use data privileges to maintain memory consistency. FleCSI tasks are
  like a more flexible version of MPI that does not require the user to
  explicitly update dependencies between different ranks, and which does
  not use static process mappings, i.e., relocatable, distributed-memory
  data parallelism.

* **Kernels** |br|
  Kernels operate on data in a single address space, but require
  explicit barriers to ensure consistency. This is generally referred to
  as a relaxed-consistency memory model. The kernel interface in
  FleCSI is defined by three parallel operations: *forall*, *reduceall*,
  and *scan*. Each of these is a fine-grained, data-parallel operation.
  The use of the *kernel* nomenclature is derived from CUDA, and OpenCL,
  and is conceptually consistent with those models. Please see example
  of using *forall* kernels in *parallel* section of the tutorials. 

----

Tasks
*****

Example 1: Single Tasks
+++++++++++++++++++++++

Single task is a task that operates on entire data passed to it and is
executed by a single process. 
`execute` method will call a single task if no ‘launch domain` (see in
the next example) or a topology with a launch domain is passed to it.

In the example of `trivial` task, there is no arguments passed and a
default `launch domain` is used. Therefore, it is a single task

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 28-31

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 82-82

You can return a value from the task. And A `future` is a mechanism, to
access the result of an asynchronous task execution.

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 93-107

FleCSI can execute a task that takes an argument by-value.
FleCSI tasks can take any valid C++ type by value. However, because task
data must be relocatable, you cannot pass pointer arguments, or
arguments that contain pointers.

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 117-119

FleCSI tasks can also be templated:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 66-73

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 127-128

Example 2: Index Tasks
++++++++++++++++++++++

Index task is a task that is executed by several processes. It is often
used to operate on different parts of the input data (like partitioned
mesh) asynchronously. 

In this example we explicitly ask to execute `task` on 4 processes by
specifying `launch_domain` parameter

.. literalinclude:: ../../../../tutorial/3-execution/2-index-task.cc
  :language: cpp
  :lines: 26-43


Launch Domains
^^^^^^^^^^^^^^

Launch domain (`exec::launch_domain`) is used to define how many index
points should index task have. If no `launch_dmain` is passed to the
`execute` method, the default will be used. The default is `0` if no
topology instanced is passed or the one specified in the topology
instance passed to the task (the number of colors topology instance
have).


Example 3: MPI Tasks
++++++++++++++++++++

MPI task is an index task that has launch domain size equal to number of
MPI ranks and index points mapped to corresponding MPI ranks. Executing
MPI task adds synchronization between Legion and MPI and, therefore,
should only be used when one needs to call MPI library.  
To execute an MPI task, second template parameter to the `execute`
method should be set to `mpi`.

.. literalinclude:: ../../../../tutorial/3-execution/3-mpi-task.cc
  :language: cpp
  :lines: 26-40

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
