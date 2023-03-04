// Low-level HDF5 interface.

#ifndef FLECSI_IO_HDF5_HH
#define FLECSI_IO_HDF5_HH

#include "flecsi/flog.hh"

#include <hdf5.h>

/// \cond core
namespace flecsi {
inline log::devel_tag io_tag("io");

namespace io::hdf5 {
/// \addtogroup io
/// \{
namespace detail {
struct id {
  explicit id(hid_t i = H5I_INVALID_HID) noexcept : h(i) {}
  id(id && h) noexcept : h(h.release()) {}

  explicit operator bool() const noexcept {
    return h >= 0;
  }
  operator hid_t() const noexcept {
    flog_assert(*this, "invalid HDF5 ID");
    return h;
  }

protected:
  hid_t release() noexcept {
    return std::exchange(h, H5I_INVALID_HID);
  }
  void swap(id & i) noexcept {
    std::swap(h, i.h);
  }

private:
  hid_t h;
};

template<herr_t (&C)(hid_t)>
struct unique : id {
protected:
  using id::id;
  unique(unique &&) = default;
  ~unique() {
    close();
  }

  unique & operator=(unique && h) & noexcept {
    unique(std::move(h)).swap(*this);
    return *this;
  }

public:
  bool close() { // true if successfully closed
    return *this && C(release()) >= 0;
  }

  void swap(unique & h) noexcept {
    id::swap(h);
  }
};

// An RAII HDF5 file handle.
struct file : unique<H5Fclose> {
  file() = default;
  file(const char * f, bool create)
    : unique(create ? H5Fcreate(f, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
                    : H5Fopen(f, H5F_ACC_RDWR, H5P_DEFAULT)) {
    const auto v = create ? "create" : "open";
    if(*this) {
      log::devel_guard guard(io_tag);
      flog_devel(info) << v << " HDF5 file " << f << " file_id " << *this
                       << std::endl;
    }
    else {
      flog(error) << "H5F" << v << " failed: " << *this << std::endl;
    }
  }
};
} // namespace detail

struct group : detail::unique<H5Gclose> {
  group(hid_t f, const char * n, bool create)
    : unique(create ? H5Gcreate2(f, n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
                    : H5Gopen2(f, n, H5P_DEFAULT)) {}
};

struct dataset : detail::unique<H5Dclose> {
  dataset(hid_t l, const char * n, hid_t file, hid_t space)
    : unique(
        H5Dcreate2(l, n, file, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) {}
  dataset(hid_t l, const char * n) : unique(H5Dopen2(l, n, H5P_DEFAULT)) {}
};

struct dataspace : detail::unique<H5Sclose> {
  template<std::size_t N>
  explicit dataspace(const hsize_t (&s)[N])
    : unique(H5Screate_simple(N, s, NULL)) {}
  explicit dataspace(const dataset & d) : unique(H5Dget_space(d)) {}
};

struct datatype : detail::unique<H5Tclose> {
  using unique::unique;
  explicit datatype(const dataset & d) : unique(H5Dget_type(d)) {}

  static datatype copy(hid_t t) {
    return datatype(H5Tcopy(t));
  }
  static datatype string() {
    auto ret = copy(H5T_C_S1);
    H5Tset_size(ret, H5T_VARIABLE);
    return ret;
  }
};

struct string {
  explicit string(const dataset & d) : dset(d) { // lifetime-bound
    H5Dread(d, str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
  }
  ~string() {
    H5Dvlen_reclaim(str, dataspace(dset), H5P_DEFAULT, &data);
  }

  operator char *() const {
    return data;
  }

private:
  datatype str = datatype::string();
  const dataset & dset;
  char * data;
};

// Higher-level interface with group support.
struct file {
  static file create(const std::string & file_name) {
    return {{file_name.c_str(), true}};
  }
  static file open(const std::string & file_name) {
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
    const group group_id(hdf5_file_id, group_name.c_str(), add);
    if(group_id < 0) {
      flog(error) << (add ? "H5Gcreate2" : "H5Gopen2")
                  << " failed: " << group_id << std::endl;
      close();
      return false;
    }

    const auto filetype = datatype::string();

    const auto data = str.c_str();
    const dataset dset(
      group_id, dataset_name.c_str(), filetype, dataspace({1}));
    status = H5Dwrite(dset, filetype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);

    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    return true;
  }

  bool read_string(const std::string & group_name,
    const std::string & dataset_name,
    std::string & str) {

    [[maybe_unused]] herr_t status; // FIXME: report errors
    // TODO:FIXME
    // status = H5Eset_auto(NULL, NULL);
    // status = H5Gget_objinfo (hdf5_file_id, group_name, 0, NULL);

    const group group_id(hdf5_file_id, group_name.c_str(), false);

    if(group_id < 0) {
      flog(error) << "H5Gopen2 failed: " << group_id << std::endl;
      close();
      return false;
    }

    str += string(dataset(group_id, dataset_name.c_str()));
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);

    return true;
  }

  bool create_dataset(const std::string & field_name, hsize_t size) {
    const hsize_t nsize = util::ceil_div(size, {sizeof(double)});

    const dataspace dataspace_id({nsize, 1});
    if(dataspace_id < 0) {
      flog(error) << "H5Screate_simple failed: " << dataspace_id << std::endl;
      close();
      return false;
    }
#if 0
    const group group_id(file_id, (*lr_it).logical_region_name.c_str(), true);
    if (group_id < 0) {
      printf("H5Gcreate2 failed: %lld\n", (long long)group_id);
      close();
      return false;
    }
#endif
    const dataset dataset(
      hdf5_file_id, field_name.c_str(), H5T_IEEE_F64LE, dataspace_id);
    if(dataset < 0) {
      flog(error) << "H5Dcreate2 failed: " << dataset << std::endl;
      close();
      return false;
    }
    H5Fflush(hdf5_file_id, H5F_SCOPE_LOCAL);
    return true;
  }

  file(detail::file f) : hdf5_file_id(std::move(f)) {}

  detail::file hdf5_file_id;
  std::set<std::string> hdf5_groups;
};
/// \}
} // namespace io::hdf5
} // namespace flecsi
/// \endcond

#endif
