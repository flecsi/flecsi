.. |br| raw:: html

   <br />

.. _TUT-DM:

Data Model
**********

FleCSI provides a data model that integrates with the task and kernel
abstractions to provide easy registration and access to various data
types with automatic dependency tracking.

Example 1: Global data
++++++++++++++++++++++

Global fields are used to store global variables/objects that can be
accessed by any task.  As there is only one value for each field, it
is natural to use the ``data::single`` layout.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 13-15

Writing to a global field requires a single task launch.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 17-35

Example 2: Index data 
+++++++++++++++++++++

An index field is a field that is local to a color or process (MPI
rank or Legion shard). It is defined as a field on an `index`
topology.

.. literalinclude:: ../../../../tutorial/4-data/2-index.cc
  :language: cpp
  :lines: 10-43

Example 3: Dense data
+++++++++++++++++++++

A dense field is a field defined on a dense topology index space.  In
this example we allocate a `pressure` field on the `cells` index space
of the `canonical` topology.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 16-16

One can access the field inside of the FleCSI task by passing
topology and field accessors with `access permissions` (wo/rw/ro).  

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 18-45

----

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
