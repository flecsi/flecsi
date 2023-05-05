Specializations
***************
A *specialization* is an adaptation of one or more core FleCSI topologies to
create an interface that is suitable for developers of a family of 
domain-specific applications.

For example, in the Poisson tutorial a specialization is provided to define a
two-dimensional finite-difference domain interface required by applications
with nearest-neighbor ghost dependencies.

Utilities
+++++++++

id<S> objects exist to help prevent mistakes such as using a cell ID as if it
were a vertex ID. topo::make_ids<S>(r) is a convenience function to convert a
range r of ordinary integers into a range of id<S> objects. (See `API Reference:
Topologies <../../api/user/group__topology.html#make_ids>`__.)

.. code-block:: cpp

  template<auto S, class C>
  auto flecsi::topo::make_ids(C && r)
