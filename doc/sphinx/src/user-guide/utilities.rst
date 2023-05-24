Utilities
*********
FleCSI provides a number of utilities that are frequently helpful in writing actions, tasks, and specializations but are not specific to FleCSI data or topologies.

Ranges
++++++
Several components of the C++20 and C++23 standard libraries are (at least in part) backported, especially for use with dense accessors.
These have been written to be usable on GPUs even where the standard-library components are not.

Renumbering
^^^^^^^^^^^
It is common that some subset of a prefix of the whole numbers needs to be treated, comprising perhaps the global indices of a partition of a mesh or the local indices of ghost index points.
``util::transform`` can be applied to a range of such indices (or structures containing them) to gather them into a range (that already has the correct size).
The indices into the output range can serve as local (or packed) IDs for the subset.
``util::partition_point`` can be applied to a sorted range of such indices or structures to efficiently find the position or data for a particular index.
``util::binary_index`` (which does not reflect a standard-library feature) is a convenience for the case of finding the position of an index in a range of just indices (*e.g.*, finding the local ID for a global ID).
