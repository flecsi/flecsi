#ifndef FLECSI_IO_HH
#define FLECSI_IO_HH

#include "flecsi/io/backend.hh"

/// \cond core
namespace flecsi::io {
/// \defgroup io I/O
/// Checkpointing.
/// \{

#ifdef DOXYGEN // implemented per-backend
struct io_interface {
  explicit io_interface(int files);

  void checkpoint_all_fields(const std::string & name);
  void recover_all_fields(const std::string & name);
};
#endif

// currently these methods don't do anything unless topologies and
// index spaces have been registered manually first
// TODO:  add automatic registration
void
checkpoint_all_fields(const std::string & file_name, int num_files) {
  io_interface(num_files).checkpoint_all_fields(file_name);
}
void
recover_all_fields(const std::string & file_name, int num_files) {
  io_interface(num_files).recover_all_fields(file_name);
}

/// \}
} // namespace flecsi::io
/// \endcond

#endif
