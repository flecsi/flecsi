# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *
import os

class Flecsi(CMakePackage):
    '''FleCSI is a compile-time configurable framework designed to support
       multi-physics application development. As such, FleCSI attempts to
       provide a very general set of infrastructure design patterns that can
       be specialized and extended to suit the needs of a broad variety of
       solver and data requirements. Current support includes multi-dimensional
       mesh topology, mesh geometry, and mesh adjacency information,
       n-dimensional hashed-tree data structures, graph partitioning
       interfaces,and dependency closures.
    '''
    homepage = 'http://flecsi.org/'
    git      = 'https://github.com/flecsi/flecsi.git'

    version('2.2', branch='2', submodules=False, preferred=False)

    #--------------------------------------------------------------------------#
    # Variants
    #--------------------------------------------------------------------------#

    variant('backend', default='legion',
            values=('legion', 'mpi', 'hpx'),
            description='Distributed-Memory Backend', multi=False)

    variant('caliper_detail', default='none',
            values=('none', 'low', 'medium', 'high'),
            description='Set Caliper Profiling Detail', multi=False)

    variant('flog', default=False,
            description='Enable FLOG Logging Utility')

    variant('graphviz', default=False,
            description='Enable GraphViz Support')

    variant('hdf5', default=False,
            description='Enable HDF5 Support')

    variant('kokkos', default=False,
            description='Enable Kokkos Support')

    variant('openmp', default=False,
            description='Enable OpenMP Support')

    variant('unit', default=False,
            description='Enable Unit Tests (Requires +flog)')

    # Spack-specific variants

    variant('shared', default=True,
            description='Build Shared Libraries')

    #--------------------------------------------------------------------------#
    # Dependencies
    #--------------------------------------------------------------------------#

    # Boost

    depends_on('boost@1.70.0: cxxstd=17 +program_options +atomic '
        '+filesystem +regex +system')

    # Caliper

    for level in ('low', 'medium', 'high'):
        depends_on('caliper', when='caliper_detail=%s' % level)

    # CMake

    depends_on('cmake@3.12:')

    # Graphviz

    depends_on('graphviz', when='+graphviz')

    # HDF5

    depends_on('hdf5+mpi+hl', when='+hdf5')

    # Kokkos

    depends_on('kokkos', when='+kokkos')

    # Legion

    depends_on('legion@ctrl-rep-11:ctrl-rep-99',when='backend=legion')
    depends_on('legion+hdf5',when='backend=legion +hdf5')
    depends_on('legion+shared',when='backend=legion +shared')

    # Metis

    depends_on('metis')
    depends_on('parmetis')

    # MPI

    depends_on('mpi')

    # HPX

    depends_on('hpx@1.7.1 cxxstd=17 malloc=system max_cpu_count=128 networking=mpi',
         when='backend=hpx')

    #--------------------------------------------------------------------------#
    # Conflicts
    #--------------------------------------------------------------------------#

    conflicts('~flog', when='+unit', msg='Unit tests require +flog')

    #--------------------------------------------------------------------------#
    # CMake Configuration
    #--------------------------------------------------------------------------#

    def cmake_args(self):
        spec = self.spec
        options = []

        options.append('-DFLECSI_RUNTIME_MODEL=%s' %
            spec.variants['backend'].value)

        options.append('-DCALIPER_DETAIL=%s' %
            spec.variants['caliper_detail'].value)

        if ('+flog' in spec):
            options.append('-DENABLE_FLOG=ON')

        if '+graphviz' in spec:
            options.append('-DENABLE_GRAPHVIZ=ON')

        if '+hdf5' in spec and spec.variants['backend'].value != 'hpx':
            options.append('-DENABLE_HDF5=ON')

        if '+kokkos' in spec:
            options.append('-DENABLE_KOKKOS=ON')

        if '+openmp' in spec:
            options.append('-DENABLE_OPENMP=ON')

        if '~shared' in spec:
            options.append('-DBUILD_SHARED_LIBS=OFF')

        if '~unit' in spec:
            options.append('-DENABLE_UNIT_TESTS=OFF')

        if '+kokkos' in spec:
            options.append('-DENABLE_KOKKOS=ON')
        else:
            options.append('-DENABLE_KOKKOS=OFF')

        return options
