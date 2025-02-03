.. |br| raw:: html

   <br />


Overview
********

The structure of applications built on top of the FleCSI programming
system assumes three basic types of users. Each of the user types has
their own set of responsibilities that are designed to separate
concerns and to make sure that development tasks are intuitive and
achievable by the associated user type.

.. tip::

  A single individual may play the role of more than one user type.

The user types are:

* **Core Developer** |br|
  These are users who design, implement, and maintain the core FleCSI
  library. Generally, these users are expert C++ developers who have a
  well-developed understanding of the the low-level design of the FleCSI
  software architecture. These users are generally computer scientists
  with expertise in generic programming techniques, data structure
  design, and optimization.
* **Specialization Developer** |br|
  These are users who adapt the core FleCSI data structures and runtime
  interfaces to create domain-specific interfaces for application
  developers.  These users are required to understand the components of
  the FleCSI interface that can be statically specialized and must have
  a solid understanding of the runtime interface. Additionally,
  specialization developers are assumed to understand the requirements
  of the application area for which they are designing an interface.
  These users are generally computational scientists with expertise in
  one or more numerical methods areas.
* **Application Developer** |br|
  These users are methods developers or physicists who use a particular
  FleCSI specialization layer to develop and maintain application codes.
  These are the FleCSI end-users, who have expertise in designing and
  implementing numerical methods to solve complicated, multiphysics
  simulation problems.

The source code implementing a FleCSI project will reflect this user
structure: the project will link to the core FleCSI library; the project will use one or more specializations (also usually from libraries); and the application
developers will use the core and specialization interfaces to write
their applications.

Documentation
+++++++++++++

The FleCSI documentation is structured as follows:

* :doc:`build`.
  This section explains how to download the FleCSI source code, lists its dependencies on other software, and specifies how to configure and build it.
* :doc:`tutorial`.
  The tutorial teaches FleCSI's core concepts through a sequence of annotated examples and associated discussion.
  All tutorial examples are provided in the FleCSI source distribution and can be compiled and run.
* :doc:`user-guide`.
  The user guide delves into more depth on the FleCSI programming model.
* :doc:`api`.
  This section provides links to the Doxygen-generated API documentation, both the public API intended for use by application and specialization developers and the internal API needed only by FleCSI core developers.
* :doc:`developer-guide`.
  The Developer Guide is meaningful only to FleCSI core developers.
  It describes how FleCSI is implemented and provides guidance on modifying, managing, and releasing FleCSI itself.

See also the :doc:`news` for advertisements of new features and bug fixes, announcements of deprecated APIs, and warnings about issues that may be encountered when upgrading applications to use new versions of FleCSI.

.. toctree::
   :hidden:

   news

Namespaces
++++++++++

FleCSI uses C++ namespaces to identify interfaces that are intended for
different user types:

* **flecsi** |br|
  The types and functions defined in the *flecsi* namespace are intended
  for all user types but are primarily targeted to application
  developers.

.. warning::

  Application developers should **never** use types or methods that are
  not defined in the top-level *flecsi* namespace.

* **flecsi::X** |br|
  The types and functions defined in *flecsi::X* namespaces, where *X*
  is nested within *flecsi*, e.g., *flecsi::topology*, should only be
  used by specialization and core developers.

* **flecsi::X::Y** |br|
  The types and functions defined in *flecsi::X::Y* namespaces are
  intended for internal FleCSI development only! **Use of any of these
  types or functions outside of the core library is undefined!!!**.

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
