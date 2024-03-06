from spack.package import *
from spack.pkg.builtin.flecsi import Flecsi

class Flecsi(Flecsi):
    depends_on('cmake@3.23:')
    depends_on("legion@cr-16:cr-99", when="backend=legion")
