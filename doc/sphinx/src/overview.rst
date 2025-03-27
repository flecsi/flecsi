.. |br| raw:: html

   <br />


Overview
********
FleCSI separates the concerns of simulation application development and efficient hardware utilization in part by supporting the construction of *specializations* of its generic topology data structures that provide domain-specific interfaces based on the relevant numerical methods.
Most sections of the documentation contain material intended for the developers of such specializations; application developers do not in general need to understand such details but do need to consult the documentation for the specialization(s) they use.
Only a few interfaces have the reverse position of being useful only to application developers (mainly because they concern program initialization); they are not specially marked.

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
FleCSI interfaces generally appear in namespaces that follow the header that provides them: ``flecsi::data`` for ``flecsi/data.hh``, ``flecsi::util`` for ``flecsi::utilities.hh``, and so forth.
The API Reference specifies the (abbreviated) name in each section as well as any member namespaces (*e.g.*, ``flecsi::exec::fold``) used to group closely related components.
However, a few fundamental facilities instead appear in the ``flecsi`` namespace itself to indicate their foundational status and make their frequent usage more convenient: ``flecsi::field``, ``flecsi::execute``, and ``flecsi::runtime``, for instance.
The API Reference also indicates such special cases.
