#include <cassert>
#include <ostream>
#include <string>

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/io.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

int
index_topology() {
  UNIT() {
    using F = io::hdf5::file;

    int my_rank = process();
    const std::string file_name{"io_metadata.dat" + std::to_string(my_rank)};

    // create hdf5 file and checkpoint
    F checkpoint_file = F::create(file_name);

    checkpoint_file.write_string("control", "ds1", "control_ds1");
    checkpoint_file.write_string("control", "ds2", "control_ds2");
    checkpoint_file.write_string("topology", "ds1", "topology_ds1");

    checkpoint_file.close();

    // recover
    checkpoint_file = F::open(file_name);

    std::string str1_recover;
    checkpoint_file.read_string("control", "ds1", str1_recover);

    flog(info) << "str1 recover " << str1_recover << std::endl;
    ASSERT_EQ(str1_recover, "control_ds1");

    std::string str2_recover;
    checkpoint_file.read_string("control", "ds2", str2_recover);

    flog(info) << "str2 recover " << str2_recover << std::endl;
    ASSERT_EQ(str2_recover, "control_ds2");

    std::string str3_recover;
    checkpoint_file.read_string("topology", "ds1", str3_recover);

    flog(info) << "str3 recover " << str3_recover << std::endl;
    ASSERT_EQ(str3_recover, "topology_ds1");

    checkpoint_file.close();
  };
} // index_topology

util::unit::driver<index_topology> driver;
