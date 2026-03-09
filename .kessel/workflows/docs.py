from kessel.workflows import environment, collapsed
from kessel.workflows.base.spack import BuildEnvironment
from kessel.workflows.base.cmake import CMake
from pathlib import Path

class Docs(BuildEnvironment, CMake):
    steps = ["env", "configure", "build"]

    build_dir = environment(Path.cwd() / "build_docs")
    spack_env = environment("docs")
    project_spec = environment("flecsi+doc")

    def ci_message(self, args):
        return super().ci_message(args, post_alloc_init="source .gitlab/kessel.sh")

    @collapsed
    def configure(self, args):
        """Configure"""
        cmake_args = [self.define("ENABLE_LIBRARY", False)]
        super().configure(args, cmake_args)

    def build(self, args):
        """Build Documentation"""
        super().build(args, targets=["doc"])
