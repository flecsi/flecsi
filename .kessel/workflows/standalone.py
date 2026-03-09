from kessel.workflows import environment, default_ci_message
from kessel.workflows.base.cmake import CMake
from pathlib import Path

class Standalone(CMake):
    steps = ["env", "configure", "build", "test"]

    spack_env = environment("default")
    source_dir = environment(Path.cwd())
    build_dir = environment(Path.cwd() / "build")
    build_env = environment()
    flecsi_install_dir = environment(variable="FLECSI_INSTALL_DIR", default=Path.cwd() / "build" / "install")

    def ci_message(self, args):
        init_msg = ("source .gitlab/kessel.sh\n"
                   f"kessel run --env {args.spack_env}")
        system = self.environ.get("KESSEL_SYSTEM", "local")
        return default_ci_message("flecsi", system=system, workflow="standalone", post_alloc_init=init_msg)

    def env_args(self, parser):
        parser.add_argument("-e", "--env", metavar="ENVIRONMENT", default=self.spack_env, dest="spack_env")
        parser.add_argument("-F", "--flecsi-install-dir", default=self.flecsi_install_dir)

    def env(self, args):
        """Prepare Environment"""
        self.exec(f"spack env activate {self.spack_env}")
        self.exec("spack repo add spack_repo/flecsi")
        self.build_dir.mkdir(exist_ok=True)
        self.build_env = self.build_dir / "build_env.sh"
        self.environ['CMAKE_PREFIX_PATH'] = self.flecsi_install_dir
        self.exec(f"{self.kessel_root}/lib/kessel/workflows/base/spack/build_environment/gen-build-env", self.build_env, "flecsi")

    def configure_args(self, parser):
        parser.add_argument("-S", "--source-dir", default=self.source_dir)
        parser.add_argument("-B", "--build-dir", default=self.build_dir)

    def configure(self, args):
        """Configure Standalone"""
        cmake_args = [
            self.define("CMAKE_BUILD_TYPE", "Debug"),
            self.define("ENABLE_UNIT_TESTS", True)
        ]
        super().configure(args, cmake_args)
