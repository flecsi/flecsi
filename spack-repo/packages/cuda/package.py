import platform
from spack.package import *
from spack.pkg.builtin.cuda import Cuda

preferred_ver = "11.8.0"
_myversions = {
    "12.3.2": {
        "Linux-aarch64": (
            "7deb8a60af6407caf04692639d8779d3",
            "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.sbsa.run"
        ),
        "Linux-x86_64": (
            "9d3585b651f4909f72c4db379beb3e01",
            "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run"
        ),
        "Linux-ppc64le": (
            "e4e0683d03a481ba71555df62383559b",
            "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux_ppc64le.run"
        ),
    }
}


class Cuda(Cuda):
    for ver, packages in _myversions.items():
        key = "{0}-{1}".format(platform.system(), platform.machine())
        pkg = packages.get(key)
        if pkg:
            if ver == preferred_ver:
                version(ver, sha256=pkg[0], url=pkg[1], expand=False, preferred=True)
            else:
                version(ver, sha256=pkg[0], url=pkg[1], expand=False)
