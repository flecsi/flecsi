// Low-level HDF5 interface.

#ifndef FLECSI_IO_HDF5_HH
#define FLECSI_IO_HDF5_HH

#include <string>

#include "flecsi/flog.hh"

#include <hdf5.h>

namespace flecsi {
inline log::devel_tag io_tag("io");

namespace io {
namespace detail {
struct hdf5 {
  hdf5() noexcept : id(-1) {}
  hdf5(const char * f, bool create)
    : id(create ? H5Fcreate(f, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
                : H5Fopen(f, H5F_ACC_RDWR, H5P_DEFAULT)) {
    const auto v = create ? "create" : "open";
    if(*this) {
      log::devel_guard guard(io_tag);
      flog_devel(info) << v << " HDF5 file " << f << " file_id " << id
                       << std::endl;
    }
    else {
      flog(error) << "H5F" << v << " failed: " << id << std::endl;
    }
  }
  hdf5(hdf5 && h) noexcept {
    id = std::exchange(h.id, -1);
  }
  ~hdf5() {
    close();
  }

  bool close() { // true if successfully closed
    if(*this) {
      H5Fflush(id, H5F_SCOPE_LOCAL);
      if(const herr_t e = H5Fclose(id); e >= 0) {
        log::devel_guard guard(io_tag);
        flog_devel(info) << "Close HDF5 file_id " << id << std::endl;
        id = -1;
        return true;
      }
      else
        flog(error) << "H5Fclose failed: " << e << std::endl;
    }
    return false;
  }

  hdf5 & operator=(hdf5 && h) noexcept {
    hdf5(std::move(h)).swap(*this);
    return *this;
  }

  void swap(hdf5 & h) noexcept {
    std::swap(id, h.id);
  }

  explicit operator bool() const {
    return id >= 0;
  }
  operator hid_t() const {
    assert(*this);
    return id;
  }

private:
  hid_t id;
};
} // namespace detail

hid_t
get_hdf5_type(int item_size) {
  hsize_t type_size = (item_size + 3) / 4;
  return H5Tarray_create2(H5T_NATIVE_B32, 1, &type_size);
}

struct hdf5 {
  static hdf5 create(const std::string & file_name) {
    return {{file_name.c_str(), true}};
  }
  static hdf5 open(const std::string & file_name) {
    return {{file_name.c_str(), false}};
  }

  /// Must not be called repeatedly.
  /// \return whether the file was successfully closed
  bool close() {
    assert(hdf5_file_id);
    return hdf5_file_id.close();
  }

  bool write_string(const std::string & group_name,
    const std::string & dataset_name,
    const std::string & str) {

    [[maybe_unused]] herr_t status; // FIXME: report errors
    // TODO:FIXME
    // status = H5Eset_auto(NULL, NULL);
    // status = H5Gget_objinfo (hdf5_file_id, group_name, 0, NULL);

    const bool add = hdf5_groups.insert(group_name).second;
    const hid_t group_id =
      add ? H5Gcreate2(hdf5_file_id,
              group_name.c_str(),
              H5P_DEFAULT,
              H5P_DEFAULT,
              H5P_DEFAULT)
          : H5Gopen2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT);
    if(group_id < 0) {
      flog(error) << (add ? "H5Gcreate2" : "H5Gopen2")
                  << " failed: " << group_id << std::endl;
      close();
      return false;
    }

    hid_t filetype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(filetype, H5T_VARIABLE);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(memtype, H5T_VARIABLE);

    const hsize_t dim = 1;
    hid_t dataspace_id = H5Screate_simple(1, &dim, NULL);

    const auto data = str.c_str();
    hid_t dset = H5Dcreate2(group_id,
      dataset_name.c_str(),
      filetype,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);
    status = H5Dwrite(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);

    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    status = H5Dclose(dset);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(filetype);
    status = H5Tclose(memtype);
    status = H5Gclose(group_id);
    return true;
  }

  bool read_string(const std::string & group_name,
    const std::string & dataset_name,
    std::string & str) {

    [[maybe_unused]] herr_t status; // FIXME: report errors
    // TODO:FIXME
    // status = H5Eset_auto(NULL, NULL);
    // status = H5Gget_objinfo (hdf5_file_id, group_name, 0, NULL);

    hid_t group_id;
    group_id = H5Gopen2(hdf5_file_id, group_name.c_str(), H5P_DEFAULT);

    if(group_id < 0) {
      flog(error) << "H5Gopen2 failed: " << group_id << std::endl;
      close();
      return false;
    }

    hid_t dset = H5Dopen2(group_id, dataset_name.c_str(), H5P_DEFAULT);

    hid_t filetype = H5Dget_type(dset);
    hid_t memtype = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(memtype, H5T_VARIABLE);

    char * data;
    status = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);

    str += data;
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);

    hid_t space = H5Dget_space(dset);
    status = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, &data);
    status = H5Dclose(dset);
    status = H5Tclose(filetype);
    status = H5Tclose(memtype);
    status = H5Gclose(group_id);
    return true;
  }

  bool create_dataset(const std::string & field_name,
    hsize_t nitems,
    hsize_t item_size) {
    const hsize_t dims[2] = {nitems, 1};
    const hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    if(dataspace_id < 0) {
      flog(error) << "H5Screate_simple failed: " << dataspace_id << std::endl;
      close();
      return false;
    }

    hid_t datatype_id = get_hdf5_type(item_size);
    if(datatype_id < 0) {
      flog(error) << "H5Tarray_create2 failed: " << datatype_id << std::endl;
      close();
      return false;
    }

#if 0
    hid_t group_id = H5Gcreate2(file_id, (*lr_it).logical_region_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
      printf("H5Gcreate2 failed: %lld\n", (long long)group_id);
      H5Sclose(dataspace_id);
      close();
      return false;
    }
#endif
    hid_t dataset = H5Dcreate2(hdf5_file_id,
      field_name.c_str(),
      datatype_id,
      dataspace_id,
      H5P_DEFAULT,
      H5P_DEFAULT,
      H5P_DEFAULT);
    if(dataset < 0) {
      flog(error) << "H5Dcreate2 failed: " << dataset << std::endl;
      //    H5Gclose(group_id);
      H5Sclose(dataspace_id);
      close();
      return false;
    }
    H5Dclose(dataset);
    //   H5Gclose(group_id);
    H5Sclose(dataspace_id);
    H5Tclose(datatype_id);
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    return true;
  }

  hdf5(detail::hdf5 h) : hdf5_file_id(std::move(h)) {}

  detail::hdf5 hdf5_file_id;
  std::set<std::string> hdf5_groups;
};

} // namespace io
} // namespace flecsi

#endif
