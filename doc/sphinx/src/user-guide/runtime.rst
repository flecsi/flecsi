.. |br| raw:: html

   <br />

.. _runtime:

Runtime Model
*************

With the following CMake options enabled:

.. code-block:: console

  $ cmake .. -DENABLE_FLOG=ON -DENABLE_GRAPHVIZ=ON

an executable compiled with FleCSI will have several command-line options available. For example, running the *task* unit test from its location in *test/exec* with the *-h* flag will produce the following output:

.. code-block:: console

  $ ./task -h
    task:
    -h [ --help ]            Print this message and exit.
    -t [ --tags ] [=arg(=0)] Enable the specified output tags, e.g.,
                             --tags=tag1,tag2. Passing --tags by itself will
                             print the available tags.
    --control-model          Output the current control model and exit.

The *--tags* option allows users to control logging output, e.g., by
turning on or off certain *guarded* outputs. This is a feature provided
by the FleCSI :ref:`flog`.  The *--control-model* option instructs the
executable to output a *.dot* file of the control-flow graph of the
control model. The FleCSI :ref:`control-model` allows users to define
the structure of execution of a program. Additional options may be added
in the future and will be documented in this guide.  

.. vim: set tabstop=2 shiftwidth=2 expandtab fo=cqt tw=72 :
