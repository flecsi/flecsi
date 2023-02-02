# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Flecsi(CMakePackage, CudaPackage, ROCmPackage):
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

    depends_on('boost@1.79.0: cxxstd=17 +program_options +atomic '
        '+filesystem +regex +system')

    # Caliper

    for level in ('low', 'medium', 'high'):
        depends_on('caliper', when='caliper_detail=%s' % level)
        conflicts('caliper@2.6', when='caliper_detail=%s' % level)
        conflicts('caliper@2.7', when='caliper_detail=%s' % level)

    # CMake

    depends_on('cmake@3.19:')

    # Graphviz

    depends_on('graphviz', when='+graphviz')

    # HDF5

    depends_on('hdf5+mpi+hl', when='+hdf5')

    # Kokkos

    depends_on('kokkos@3.2.00:', when='+kokkos')
    depends_on('kokkos +cuda +cuda_constexpr +cuda_lambda', when='+kokkos +cuda')
    depends_on('kokkos +rocm', when='+kokkos +rocm')

    # Legion

    depends_on('legion@ctrl-rep-13:ctrl-rep-99',when='backend=legion')
    depends_on('legion+hdf5',when='backend=legion +hdf5')
    depends_on('legion+shared',when='backend=legion +shared')
    depends_on('legion network=gasnet', when='backend=legion')
    depends_on('legion +kokkos +cuda', when='backend=legion +kokkos +cuda')
    conflicts('legion@ctrl-rep-14',when='backend=legion')

    # Metis

    depends_on('metis')
    depends_on('parmetis')

    # MPI

    depends_on('mpi')

    # HPX

    depends_on('hpx@1.7.1: cxxstd=17 malloc=system max_cpu_count=128 '
        'networking=mpi', when='backend=hpx')

    # Propagate cuda_arch requirement to dependencies
    cuda_arch_list = ('60', '70', '75', '80')
    for _flag in cuda_arch_list:
        depends_on("kokkos cuda_arch=" + _flag, when="+cuda+kokkos cuda_arch=" + _flag)
        depends_on("legion cuda_arch=" + _flag, when="backend=legion +cuda cuda_arch=" + _flag)

    # Propagate amdgpu_target requirement to dependencies
    for _flag in ROCmPackage.amdgpu_targets:
        depends_on("kokkos amdgpu_target=" + _flag, when="+kokkos +rocm amdgpu_target=" + _flag)

    #--------------------------------------------------------------------------#
    # Conflicts
    #--------------------------------------------------------------------------#

    conflicts('~flog', when='+unit', msg='Unit tests require +flog')

    #--------------------------------------------------------------------------#
    # CMake Configuration
    #--------------------------------------------------------------------------#

    def cmake_args(self):
        spec = self.spec
        options = [
            self.define_from_variant('FLECSI_BACKEND', 'backend'),
            self.define_from_variant('CALIPER_DETAIL', 'caliper_detail'),
            self.define_from_variant('ENABLE_FLOG', 'flog'),
            self.define_from_variant('ENABLE_GRAPHVIZ', 'graphviz'),
            self.define('ENABLE_HDF5', '+hdf5' in spec and spec.variants['backend'].value != 'hpx'),
            self.define_from_variant('ENABLE_KOKKOS', 'kokkos'),
            self.define_from_variant('ENABLE_OPENMP', 'openmp'),
            self.define_from_variant('BUILD_SHARED_LIBS', 'shared'),
            self.define_from_variant('ENABLE_UNIT_TESTS', 'unit')
        ]

        if "+rocm" in self.spec:
            options.append(self.define("CMAKE_CXX_COMPILER", self.spec["hip"].hipcc))
            options.append(self.define("CMAKE_C_COMPILER", self.spec["hip"].hipcc))
        return options
