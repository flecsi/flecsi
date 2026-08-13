---
title: 'FleCSI: Flexible Computational Science Infrastructure'
tags:
  - programming models
  - task-based runtimes
  - parallel computing
  - distributed computing
  - performance portability
  - computational science
  - multiphysics
  - GPU
  - C++
authors:
# Authors are sorted in decreasing order of the total number of lines
# inserted into the FleCSI Git repository.
  - given-names: Benjamin
    surname: Bergen
    affiliation: 1
    orcid: 0009-0008-1802-4285
  - given-names: Nick
    surname: Moss
    affiliation: 2
  - given-names: Irina
    surname: Demeshko
    affiliation: 3
    orcid: 0009-0001-1035-7260
  - given-names: Davis
    surname: Herring
    affiliation: 1
    orcid: 0009-0000-2467-5537
  - given-names: Marc
    surname: Charest
    affiliation: 4
    orcid: 0000-0002-2818-1232
  - given-names: Julien
    surname: Loiseau
    affiliation: 1
    orcid: 0000-0002-2116-2493
  - given-names: Navamita
    surname: Ray
    affiliation: 1
    orcid: 0000-0002-8235-1706
  - given-names: Jonathan
    surname: Graham
    affiliation: 1
    orcid: 0000-0003-1862-0526
  - given-names: Hartmut
    surname: Kaiser
    affiliation: 5
    orcid: 0000-0002-8712-2806
  - given-names: Li-Ta
    surname: Lo
    affiliation: 1
    orcid: 0000-0001-6244-9696
  - given-names: Karen
    surname: Tsai
    affiliation: 1
    orcid: 0000-0003-2848-832X
  - given-names: Charles
    surname: Ferenbaugh
    affiliation: 1
    orcid: 0000-0001-7908-8567
  - given-names: Richard
    surname: Berger
    affiliation: 1
    orcid: 0000-0002-3044-8266
  - given-names: John
    surname: Wohlbier
    affiliation: 6
    orcid: 0009-0000-8749-2762
  - given-names: Jonas
    surname: Lippuner
    affiliation: 2
    orcid: 0000-0002-5936-3485
  - given-names: Wei
    surname: Wu
    affiliation: 3
    orcid: 0000-0002-2750-6365
  - given-names: Andrew
    surname: Reisner
    affiliation: 1
    orcid: 0000-0002-5325-2266
  - given-names: Scott
    surname: Pakin
    affiliation: 1
    orcid: 0000-0002-5220-1985
  - given-names: Brendan K.
    surname: Krueger
    affiliation: 1
    orcid: 0000-0002-8275-9277
  - given-names: Lukas
    surname: Spies
    affiliation: 7
    orcid: 0000-0002-2063-7637
  - given-names: Sumathi
    surname: Lakshmiranganatha
    affiliation: 1
    orcid: 0000-0001-8369-9387
  - given-names: Max
    surname: Ortner
    affiliation: 2
    orcid: 0000-0001-9016-3708
  - given-names: Pascal
    surname: Grosset
    affiliation: 1
    orcid: 0000-0003-2192-3843
  - given-names: David
    surname: Gunter
    affiliation: 2
  - given-names: Maxim
    surname: Moraru
    affiliation: 1
    orcid: 0000-0001-7213-089X
  - given-names: Galen
    surname: Shipman
    orcid: 0000-0001-6297-2145
    affiliation: 1
  - given-names: Jiajia
    surname: Waters
    affiliation: 1
    orcid: 0000-0002-6517-4445
  - given-names: Scot A.
    surname: Halverson
    orcid: 0009-0005-1017-4682
    affiliation: 3
  - given-names: Onur
    surname: Çaylak
    affiliation: 12
    orcid: 0000-0003-2410-7411
  - given-names: Peter
    surname: Brady
    affiliation: 1
    orcid: 0000-0002-4906-2195
  - given-names: Philipp V. F.
    surname: Edelmann
    affiliation: 1
    orcid: 0000-0001-7019-9578
  - given-names: Mason
    surname: Delan
    affiliation: 2
  - given-names: Brandon
    surname: Keim
    affiliation: 8
    orcid: 0009-0006-8688-3642
  - given-names: Christopher M.
    surname: Malone
    affiliation: 1
    orcid: 0000-0002-4045-7932
  - given-names: Alex
    surname: Villa
    affiliation: 9
    orcid: 0009-0000-3422-8626
  - given-names: Daniel
    surname: Holladay
    affiliation: 1
    orcid: 0000-0002-0673-9741
  - given-names: Dani
    surname: Barrack
    affiliation: 2
    orcid: 0000-0002-4881-0921
  - given-names: Nikunj
    surname: Gupta
    affiliation: 10
    orcid: 0000-0003-0525-3667
  - given-names: Ondřej
    surname: Čertík
    affiliation: 4
    orcid: 0000-0003-3968-3614
  - given-names: Robert
    surname: Bird
    affiliation: 2
    orcid: 0000-0003-1228-498X
  - given-names: Melissa
    surname: Rasmussen
    affiliation: 11
    orcid: 0000-0002-0297-0313
  - given-names: Christoph
    surname: Junghans
    affiliation: 1
    orcid: 0000-0003-0925-1458
affiliations:
  - name: Los Alamos National Laboratory, USA
    index: 1
  - name: Independent researcher, USA
    index: 2
  - name: NVIDIA, USA
    index: 3
  - name: Microsoft, USA
    index: 4
  - name: Louisiana State University, USA
    index: 5
  - name: Software Engineering Institute, USA
    index: 6
  - name: INRIA, France
    index: 7
  - name: University at Buffalo, USA
    index: 8
  - name: University of California, Merced, USA
    index: 9
  - name: Databricks, USA
    index: 10
  - name: Stony Brook University, USA
    index: 11
  - name: Independent researcher, UK
    index: 12
date: 6 June 2025
bibliography: "flecsi_joss.bib"
header-includes:
- |
  ```{=latex}
  \usepackage{nameref}
  ```
---

<!-- ![FleCSI logo.](flecsi.png){ width=50% }

---
-->

# Summary

<!-- (A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.) -->

**FleCSI** [@bergen2021flecsi] is a modern C++ framework designed to support the development of multiphysics simulations. It provides a task-based programming model that unifies shared- and distributed-memory programming. FleCSI provides high performance, flexibility, and portability across heterogeneous computing architectures.

![The FleCSI software ecosystem\label{fig:ecosystem}](flecsi_diagram.pdf){ width=60% }

# Statement of need
<!-- (Section that clearly illustrates the research purpose of the software and places it in the context of related work.) -->

FleCSI is designed to support the development of multiphysics simulations through a flexible, task-based programming model that enables performance portability across distributed, heterogeneous systems. It advances prior work by integrating dynamic task scheduling, data abstraction, and backend interoperability within a unified C++ framework. Compared to related systems like Uintah [@meng2012uintah] and MPC [@perache2008mpc], FleCSI offers greater extensibility and finer runtime control, placing it at the intersection of portability, scalability, and modern software design for scientific computing.

# Software description

FleCSI is designed to abstract away complexity while offering fine control for high-performance computing.  The FleCSI runtime system manages initialization, execution, and shutdown. As presented in \autoref{fig:ecosystem}, the FleCSI runtime supports backends such as Legion [@bauer2012legion], HPX [@kaiser2009parallex; @Kaiser2020hpx], MPI [@mpi50], and Kokkos [@edwards2014kokkos], enabling code to remain portable across a variety of systems without manually handling the execution environment.

FleCSI’s programming model is based on a hierarchy of parallelism: sequential, task-parallel, and data-parallel.  The relationships among these is illustrated in \autoref{fig:model}:

* **Control points** (CP) define an application's sequential backbone.
* **Actions** (A) specify a directed acyclic graph of high-level operations and their dependencies.
* **Tasks** are functions that operate on data distributed across address spaces.
* **Point tasks** (PT) are individual instances of a task that operate on a local fragment of a distributed data structure.
* **Kernels** (K) process a block of local data in a data-parallel fashion on a CPU or GPU.

![FleCSI control and execution models\label{fig:model}.](flecsi_model.pdf){ width=100% }

FleCSI's _control model_ comprises control points and actions and determines what work is performed and in what order.  FleCSI's _execution model_, comprising tasks, point tasks, and kernels, governs where and how that work actually runs.  FleCSI's _data model_, not shown in \autoref{fig:model}, governs how data are distributed and accessed.

## Control model

_Control points_ specify an application's sequential control flow and can include conditional branches.  For example, one control point may represent "initialization", another "repetition until convergence", and a third "finalization".

Control points provide hooks for a directed acyclic graph (DAG) of _actions_ to be attached.  An action is a sequential function that defines an application's core numerics or physics routines such as "hydrodynamics", "radiative diffusion", or "reaction network".  A new action can be incorporated into an application by specifying its direct dependents and dependencies (control points or other actions).  For example, if an existing application defines a "solver iteration" action, a new developer later can create a "visualization" action and insert it after "solver iteration" in the DAG without having to modify any other code or interfaces.  If other actions depend on "solver iteration", these will run concurrently with "visualization".  By walking the DAG in topological order, FleCSI ensures a valid program execution.

## Execution model

Actions spawn _tasks_, which are functions that are distributed within and across the nodes of the compute cluster and that complete asynchronously.
In a computational-science application, a task typically represents updates to a data structure, such as to perform mesh operations (e.g., relaxation).
Because there exist run-time costs in launching tasks and moving data across a large-scale, hybrid CPU/GPU cluster, task granularity should be large enough to amortize these costs.
A rule of thumb is for tasks to execute in no less than about 10 ms to keep the relative overhead manageable.

A task declaration includes the _fields_ of a distributed data structure that it will access (defined on, say, the cells, edges, and vertices of an unstructured mesh) and the access rights it requires on each field: read only, write only, or read/write.
Tasks are run concurrently according to field data dependencies.
For example, if task A reads $x$ and writes $y$, task B reads $x$ and writes $z$, and task C reads $y$ and writes $w$, then the FleCSI runtime will execute tasks A and B concurrently but require that task A finish before task C can start.

Distributed parallelism is achieved by decomposing field domains into _colors_.
For any one color, memory for a field is allocated contiguously, but colors do not need to map 1:1 to processes.
Rather, the application chooses an appropriate number of colors for each task launch.
If colors outnumber processes then some processes simply handle more than one color.
Each color is handled by exactly one _point task_—an individual instance of a task.
All point tasks are executed on CPUs, but the data for readable fields are preloaded into a specified memory space (CPU NUMA domain or GPU device memory), and the data for writable fields are automatically communicated to dependent tasks.

Point tasks process their data by launching data-parallel _kernels_ that operate on the memory space in which the field data was placed.  In a computational-science application, these typically perform element updates such as incrementing position, momentum, energy, etc.  Kernels can execute in parallel on GPUs, in parallel on CPUs (using OpenMP threads), or serially on CPUs.  Kernel code is portable across these three forms of execution; no code modifications are needed to dispatch a kernel to a CPU versus a GPU.  This is because FleCSI arranges for a kernel's data to be available locally before the kernel is launched---in either a CPU or GPU execution space---and because FleCSI provides `forall` and `reduceall` constructs that operate consistently across execution spaces, supporting data-parallel code execution.  In cases where these constructs are too limiting, FleCSI supports _task variants_, whereby a program can provide different implementations for different execution spaces (e.g., using arbitrary Kokkos calls in the GPU execution space and explicit OpenMP pragmas in the OpenMP execution space) and let FleCSI select the appropriate variant at run time.

## Data model

FleCSI provides several topology types---skeletons of distributed data structures---that applications use to represent physical quantities and their relationships:

* `topo::unstructured` supports graph-based meshes and is suitable for finite element or finite volume methods.
* `topo::narray` provides structured $n$-dimensional grids with support for boundary conditions and periodicity, making it ideal for Eulerian hydrodynamics.
* `topo::ntree` organizes data in a hashed tree structure that enables fast neighbor searches and is appropriate for particle-based simulations and adaptive mesh refinement.

Although topology data are distributed, all communication and synchronization is implicit and is based on the access rights associated with each field.  (See \nameref{execution-model} above.)  Fields can be defined with several layouts such as dense (arrays), ragged (vectors), sparse (maps), or particle (buffers).

# Acknowledgments

The FleCSI project is supported by the U.S. Department of Energy through Los Alamos National Laboratory (LANL).  Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of the U.S. Department of Energy (contract no. 89233218CNA000001). This paper has been assigned a Los Alamos Unlimited Release number of LA-UR-25-25479.

The work reported in this paper would not have been possible without close collaborations with the Legion and HPX teams and LANL's Ristra project, FleCSI's initial "customer".

# References
