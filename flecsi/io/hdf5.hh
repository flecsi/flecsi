// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

// Low-level HDF5 interface.

#ifndef FLECSI_IO_HDF5_HH
#define FLECSI_IO_HDF5_HH

#include <hdf5.h>
#include <stdexcept>
#include <string>
#ifdef H5_HAVE_PARALLEL
#include <mpi.h>
#endif

#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

/// \cond core
namespace flecsi {
inline flog::devel_tag io_tag("io");

namespace io::hdf5 {
/// \addtogroup io
/// \{
struct exception : std::runtime_error {
  using runtime_error::runtime_error;
};
template<auto & F, class... TT>
auto // hid_t or herr_t
test(TT &&... tt) {
  const auto ret = F(std::forward<TT>(tt)...);
  if(ret < 0)
    throw exception(util::symbol<F>());
  return ret;
}

namespace detail {
struct id {
  explicit id(hid_t i = H5I_INVALID_HID) noexcept : h(i) {}
  id(id && h) noexcept : h(h.release()) {}

  explicit operator bool() const noexcept {
    return h >= 0;
  }
  operator hid_t() const noexcept {
    flog_assert(*this, "empty HDF5 ID");
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
    if(*this)
      C(release()); // NB: error lost
  }

  unique & operator=(unique && h) & noexcept {
    unique(std::move(h)).swap(*this);
    return *this;
  }

public:
  void close() {
    if(*this)
      test<C>(release());
  }

  void swap(unique & h) noexcept {
    id::swap(h);
  }
};
} // namespace detail

struct plist : detail::unique<H5Pclose> {
  explicit plist(hid_t cls) : unique(test<H5Pcreate>(cls)) {}
};

namespace detail {
// An RAII HDF5 file handle.
struct file : unique<H5Fclose> {
  file() = default;
  file(const char * f, bool create) : file(f, create, H5P_DEFAULT) {}
#ifdef H5_HAVE_PARALLEL
  file(const char * f, MPI_Comm comm, bool create)
    : file(f, create, [&] {
        plist ret(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(ret, comm, MPI_INFO_NULL);
        return ret;
      }()) {}
#endif

private:
  file(const char * f, bool create, hid_t pl)
    : unique(create ? test<H5Fcreate>(f, H5F_ACC_TRUNC, H5P_DEFAULT, pl)
                    : test<H5Fopen>(f, H5F_ACC_RDWR, pl)) {
    flog::devel_guard guard(io_tag);
    flog_devel(info) << (create ? "create" : "open") << " HDF5 file " << f
                     << " file_id " << *this << std::endl;
  }
};
} // namespace detail

struct group : detail::unique<H5Gclose> {
  group(hid_t f, const char * n, bool create)
    : unique(create
               ? test<H5Gcreate2>(f, n, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
               : test<H5Gopen2>(f, n, H5P_DEFAULT)) {}
};

struct dataset : detail::unique<H5Dclose> {
  dataset(hid_t l, const char * n, hid_t file, hid_t space)
    : unique(test<H5Dcreate2>(l,
        n,
        file,
        space,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT)) {}
  dataset(hid_t l, const char * n)
    : unique(test<H5Dopen2>(l, n, H5P_DEFAULT)) {}
};

struct dataspace : detail::unique<H5Sclose> {
  template<std::size_t N>
  explicit dataspace(const hsize_t (&s)[N])
    : unique(test<H5Screate_simple>(N, s, nullptr)) {}
  explicit dataspace(const dataset & d) : unique(test<H5Dget_space>(d)) {}
};

struct datatype : detail::unique<H5Tclose> {
  using unique::unique;
  explicit datatype(const dataset & d) : unique(test<H5Dget_type>(d)) {}

  static datatype bytes(hsize_t item_size) {
    return datatype(test<H5Tarray_create2>(H5T_NATIVE_B8, 1, &item_size));
  }
  static datatype copy(hid_t t) {
    return datatype(test<H5Tcopy>(t));
  }
  static datatype string() {
    auto ret = copy(H5T_C_S1);
    H5Tset_size(ret, H5T_VARIABLE);
    return ret;
  }
};

struct string {
  explicit string(const dataset & d) : dset(d) { // lifetime-bound
    test<H5Dread>(d, str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
  }
  ~string() {
    test<H5Dvlen_reclaim>(str, dataspace(dset), H5P_DEFAULT, &data);
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

#ifdef H5_HAVE_PARALLEL
  static file pcreate(const std::string & file_name, MPI_Comm comm) {
    return {{file_name.c_str(), comm, true}};
  }
  static file popen(const std::string & file_name, MPI_Comm comm) {
    return {{file_name.c_str(), comm, false}};
  }
#endif

  /// Must not be called repeatedly.
  void close() {
    assert(hdf5_file_id);
    hdf5_file_id.close();
  }

  void write_string(const std::string & group_name,
    const std::string & dataset_name,
    const std::string & str) {
    const auto filetype = datatype::string();
    const auto data = str.c_str();
    test<H5Dwrite>(dataset(group(hdf5_file_id,
                             group_name.c_str(),
                             hdf5_groups.insert(group_name).second),
                     dataset_name.c_str(),
                     filetype,
                     dataspace({1})),
      filetype,
      H5S_ALL,
      H5S_ALL,
      H5P_DEFAULT,
      &data);
    test<H5Fflush>(hdf5_file_id, H5F_SCOPE_LOCAL);
  }

  void read_string(const std::string & group_name,
    const std::string & dataset_name,
    std::string & str) {
    str += string(dataset(
      group(hdf5_file_id, group_name.c_str(), false), dataset_name.c_str()));
    test<H5Fflush>(hdf5_file_id, H5F_SCOPE_LOCAL);
  }

  void create_dataset(const std::string & field_name,
    hsize_t nitems,
    hsize_t item_size) {
#if 0
    const group group_id(file_id, (*lr_it).logical_region_name.c_str(), true);
#endif
    dataset(hdf5_file_id,
      field_name.c_str(),
      datatype::bytes(item_size),
      dataspace({nitems, 1}));

    test<H5Fflush>(hdf5_file_id, H5F_SCOPE_LOCAL);
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
