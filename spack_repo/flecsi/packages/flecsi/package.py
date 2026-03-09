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

    depends_on("texlive", when="+doc")
    depends_on("pdf2svg", when="+doc")
    variant("format", default=False, description="Enable Formatting")
    depends_on("llvm@13 +clang", type="build", when="+format")
    depends_on("git", type="build", when="+format")

    def cmake_args(self):
        args = super().cmake_args()
        args.append(self.define_from_variant("ENABLE_FORMAT", "format"))
        if self.spec.satisfies("+format"):
            args.append(self.define("ClangFormat_EXECUTABLE", Path(self.spec["llvm"].prefix.bin) / "clang-format"))
        return args
