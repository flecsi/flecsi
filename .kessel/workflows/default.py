from kessel.workflows import environment, collapsed
from kessel.workflows.base.spack import BuildEnvironment
from kessel.workflows.base.cmake import CMake

class Default(BuildEnvironment, CMake):
    steps = ["env", "configure", "build_noflog", "build", "test", "install"]

    project_spec = environment("flecsi+flog")
    tests = True

    def ci_message(self, args):
        return super().ci_message(args, post_alloc_init="source .gitlab/kessel.sh")

    def build_noflog(self, args):
        """Build (without FLOG)"""
        cmake_args = [
            self.define("ENABLE_FLOG", False),
            self.define("ENABLE_UNIT_TESTS", False),
            self.define("ENABLE_DEVELOPER_WARNINGS", True)
        ]
        super().build(args, cmake_args)

    def build(self, args):
        """Build (with FLOG)"""
        cmake_args = [
            self.define("ENABLE_FLOG", True),
            self.define("ENABLE_UNIT_TESTS", True),
            self.define("ENABLE_DEVELOPER_WARNINGS", True)
        ]
        super().build(args, cmake_args)

    @collapsed
    def install(self, args):
        super().install(args)
