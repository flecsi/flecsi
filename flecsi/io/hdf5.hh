// Low-level HDF5 interface.

#ifndef FLECSI_IO_HDF5_HH
#define FLECSI_IO_HDF5_HH

#include "flecsi/flog.hh"

#include <hdf5.h>

namespace flecsi {
inline log::devel_tag io_tag("io");

namespace io {

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

} // namespace io
} // namespace flecsi

#endif
