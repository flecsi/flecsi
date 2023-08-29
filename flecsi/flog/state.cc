#include <flecsi-config.h>

#include "flecsi/execution.hh"
#include "flecsi/flog/packet.hh"
#include "flecsi/flog/state.hh"
#include "flecsi/flog/utils.hh"
#include "flecsi/util/mpi.hh"

#if defined(FLECSI_ENABLE_FLOG)

namespace flecsi {
namespace flog {

#ifdef FLOG_ENABLE_TAGS
task_local<std::size_t> state::cur_tag;
#endif

std::size_t &
state::active_tag() {
#ifdef FLOG_ENABLE_TAGS
  return *cur_tag;
#else
  static std::size_t t; // modifications disabled here
  return t;
#endif
}

#if defined(FLOG_ENABLE_MPI)

void
state::send_to_one(bool last) {
  using util::mpi::test;

  std::lock_guard guard(packets_mutex_);

  std::vector<int> sizes(process_ ? 0 : processes_), offsets(sizes);
  std::vector<std::byte> data, buffer;

  if(active_process())
    data = util::serial::put_tuple(packets_);

  int bytes = data.size();

  if(source_process_ == 0) {
    buffer = std::move(data);
  }
  else if(source_process_ == all_processes) {
    test(MPI_Gather(
      &bytes, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
    int sum{0};

    if(process_ == 0) {
      for(Color p = 0; p < processes_; ++p) {
        offsets[p] = sum;
        sum += sizes[p];
      } // for

      buffer.resize(sum);
    } // if

    test(MPI_Gatherv(data.data(),
      bytes,
      MPI_CHAR,
      buffer.data(),
      sizes.data(),
      offsets.data(),
      MPI_CHAR,
      0,
      MPI_COMM_WORLD));
  }
  else {
    if(process_ == 0) {
      test(MPI_Recv(&bytes,
        1,
        MPI_INT,
        source_process_,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));
      buffer.resize(bytes);
      test(MPI_Recv(buffer.data(),
        bytes,
        MPI_CHAR,
        source_process_,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));
    }
    else if(process_ == source_process_) {
      test(MPI_Send(&bytes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
      test(MPI_Send(data.data(), bytes, MPI_CHAR, 0, 0, MPI_COMM_WORLD));
    }
  }

  packets_.clear();

  if(process_ == 0) {

    for(Color p = 0; p < processes_; ++p) {

      if(source_process_ == all_processes || p == source_process_) {
        auto remote_packets =
          util::serial::get1<std::vector<packet_t>>(buffer.data() + offsets[p]);

        packets_.reserve(packets_.size() + remote_packets.size());
        packets_.insert(
          packets_.end(), remote_packets.begin(), remote_packets.end());
      } // if
    } // for

    stop = last;
    avail.notify_one();
  } // if

} // send_to_one

void
state::flush_packets() {
  std::unique_lock lk(packets_mutex_);
  while(true) {
    std::sort(packets_.begin(), packets_.end());

    for(auto & p : packets_) {
      stream_ << p.message();
    } // for

    packets_.clear();
    if(stop)
      break;
    avail.wait(lk); // spurious wakeups have no effect
  } // while
} // flush_packets

#endif // FLOG_ENABLE_MPI

} // namespace flog
} // namespace flecsi

#endif // FLECSI_ENABLE_FLOG
