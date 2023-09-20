.. |br| raw:: html

   <br />

.. _performance:

Performance Do's and Don'ts
***************************

FleCSI's programming model enables asynchronous parallel performance.
This is a partial list of patterns (do's) and anti-patterns (don'ts)
that have significant performance consequences.

.. toctree::


----

Do Activate Tracing Around Your `flecsi::execute<>` Loops
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. toctree::

   tracing

Don't Block the Top-Level Task (TLT)
++++++++++++++++++++++++++++++++++++

.. toctree::

   blocking
