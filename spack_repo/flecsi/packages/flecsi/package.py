from spack.package import *
from spack_repo.builtin.packages.flecsi.package import Flecsi
from pathlib import Path

with open(Path(__file__).parent / "../../../../.version") as f:
  dot_version = f.read().strip()

flecsi_selector = {"preferred": True}

if dot_version == "develop":
    flecsi_selector["branch"] = "develop"
    flecsi_version = "develop"
elif dot_version.startswith("v"):
    flecsi_selector["tag"] = dot_version[1:]
    flecsi_version = dot_version[1:]
else:
    flecsi_selector["branch"] = dot_version[1:]
    flecsi_version = dot_version[1:] + ".develop"

class Flecsi(Flecsi):
    version(flecsi_version, **flecsi_selector)

    # local development / CI changes (not intended for public Spack package)
    conflicts('^hpx networking=tcp', when='backend=hpx')

    variant("format", default=False, description="Enable Formatting")
    depends_on("llvm@13 +clang", type="build", when="+format")
    depends_on("git", type="build", when="+format")

    def setup_build_environment(self, env):
        # build environment changes are needed for CI testing
        # compiler env vars get put into the generated build-env, which allows
        # standalones to pick up the right compiler
        if self.run_tests and self.spec.satisfies("^ucx"):
            # UCX workaround to avoid misdetecting GPU memory as host memory
            env.set("UCX_MEMTYPE_CACHE", "n")
        if self.spec.satisfies("^kokkos +rocm"):
            env.set("CC", self.spec["hip"].hipcc)
            env.set("CXX", self.spec["hip"].hipcc)
        if self.run_tests and self.spec.satisfies("^[virtuals=mpi]openmpi@5:"):
            # OpenMPI 5.x uses bind-to core by default, limiting us to a single core
            env.set("OMPI_MCA_hwloc_base_binding_policy", "none")

    def cmake_args(self):
        args = super().cmake_args()
        args.append(self.define_from_variant("ENABLE_FORMAT", "format"))
        if self.spec.satisfies("+format"):
            args.append(self.define("ClangFormat_EXECUTABLE", Path(self.spec["llvm"].prefix.bin) / "clang-format"))
        return args
