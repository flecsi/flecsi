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
`Several class and function templates <../../api/user/group__topology.html>`__ are provided to assist in writing specializations.
The class template ``topo::id`` exists to help prevent mistakes such as using a cell ID as if it were a vertex ID.
``topo::make_ids`` is a convenience function template to convert a range of ordinary integers into a range of ``id`` objects.
