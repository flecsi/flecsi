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
  and is conceptually consistent with those models. Please see the
  example of using *forall* kernels in the *parallel* section of the
  tutorial. 

----

Tasks
*****

Example 1: Single Tasks
+++++++++++++++++++++++

A `single` task launches on a single process, i.e., only one instance of
the task is executed.
This is in contrast to an `index` launch, which executes a task as a
data-parallel operation, potentially across many processes.
FleCSI uses information about the arguments passed to a task to decide
how to launch the task: If no parameter is passed that defines a `launch
domain`, e.g., an explicit `launch domain`, a `topology` instance, or a
`future map`, FleCSI will launch the task as `single`.

The `trivial` task is an example of a `single` task.
Consider the following from `tutorial/3-execution/1-single-task.cc`:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 22-29

Execution of the task is trivial:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 76-80

A single task can return a value:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 31-40

The return value can be retrieved with a `future`:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 82-107

FleCSI tasks can take any valid C++ type as an argument `by-value`,
e.g., a ``std::vector``:

.. caution::

    FleCSI tasks can take any valid C++ type by value. However, because
    task data must be relocatable, you cannot pass pointer arguments, or
    arguments that contain pointers.  Modifications made to by-value
    data are local to the task and will not be reflected at the call
    site.

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 42-58

Execution of such a task is what you would expect:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 109-121

FleCSI tasks can also be templated:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 60-71

Again, execution is straightforward:

.. literalinclude:: ../../../../tutorial/3-execution/1-single-task.cc
  :language: cpp
  :lines: 123-131

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
