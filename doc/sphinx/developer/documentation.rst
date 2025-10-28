Documentation
*************

Published
+++++++++

The `project homepage`__ is a GitHub Pages site generated from `the current GitHub repository`__.
The ``deploy-docs`` Make target sets up a repository to update it by pushing to the appropriate branch.

  __ https://www.flecsi.org/
  __ https://github.com/flecsi/flecsi/tree/gh-pages

`Doxygen`__
+++++++++++
__ https://www.doxygen.nl/manual/

The API reference is organized exclusively using the groups feature; none of the files and namespaces are documented, since they have little relevance to the user.
Note that members of namespaces enclosed by the ``\{`` and ``\}`` of a grouping command are not included in the group.

The developers' version of the API reference includes a selection of internal interfaces for core developers (who of course must nonetheless consult the source in general).
Use ``\cond core`` and ``\endcond`` to mark material that should appear only there.
Developers must also check for such markers to identify internal interfaces; the output does not distinguish them, since users should consult only the users' version anyway.

`Sphinx`__
++++++++++
__ https://www.sphinx-doc.org/en/master

By convention, headings are underlined with characters in the order ``*+^=``.
Because our formatting tends to have many tasks begin with a line of just ``void``, some tutorial code has additional comments whose principal purpose is to be a ``:start-after:`` anchor.
