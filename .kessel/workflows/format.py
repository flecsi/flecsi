from kessel.workflows import environment, collapsed
from kessel.workflows.base.spack import BuildEnvironment
from kessel.workflows.base.cmake import CMake
from pathlib import Path

class Format(BuildEnvironment, CMake):
    steps = ["env", "configure", "check_format"]

    build_dir = environment(Path.cwd() / "build_format")
    spack_env = environment("format")
    project_spec = environment("flecsi+format")
    tests = False

    def ci_message(self, args):
        return super().ci_message(args, post_alloc_init="source .gitlab/kessel.sh")

    @collapsed
    def configure(self, args):
        """Configure"""
        cmake_args = [self.define("ENABLE_LIBRARY", False)]
        super().configure(args, cmake_args)

    def check_format(self, args):
        """Clang-Format"""
        super().build(args, targets=["format"])
        self.exec(f"git -C '{self.source_dir}' diff --exit-code --compact-summary")
