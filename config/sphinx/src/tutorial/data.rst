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

Global fields are used to store global variables/objects that can
be accessed by any task. They are created as a field on the
`data::single` index space.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 24-26

Global fields can be initialized only within a task that has size of the launch domain equal to 1 (default for `flecsi::single` topology). 

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 28-46

Example 2: Index data 
+++++++++++++++++++++

Index field is a field that is local to a color or process (MPI rank or
Legion shard). It is defined as a filed on an `index` topology.

.. literalinclude:: ../../../../tutorial/4-data/2-index.cc
  :language: cpp
  :lines: 21-54

Example 3: Dense data
+++++++++++++++++++++

Dense field is a field defined on a dense topology index space.
In this example we allocate `pressure` field on the `cells` index space
of `canonical` topology.

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 27-27

One can access the field inside of the FLeCSI task through passing
topology and field accessors with `access permissions` (wo/rw/ro).  

.. literalinclude:: ../../../../tutorial/4-data/3-dense.cc
  :language: cpp
  :lines: 29-56

----

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
