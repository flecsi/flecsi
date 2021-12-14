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
be accessed by any task.
For the common case of only one value for each field, it is natural to use the ``data::single`` layout.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 24-28

Writing to a global field requires a single task launch.

.. literalinclude:: ../../../../tutorial/4-data/1-global.cc
  :language: cpp
  :lines: 30-48

Example 2: Index data 
+++++++++++++++++++++

A field on an ``index`` topology stores one value for each color.

.. literalinclude:: ../../../../tutorial/4-data/2-index.cc
  :language: cpp
  :lines: 23-51

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
