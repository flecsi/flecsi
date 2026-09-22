#include "flecsi/flog/state.hh"
#include "flecsi/execution.hh"
#include "flecsi/flog/utils.hh"
#include "flecsi/util/mpi.hh"

#if defined(FLECSI_ENABLE_FLOG)

namespace flecsi {
namespace flog {

task_local<state::Tag> state::cur_tag;

state::Tag &
state::active_tag() {
  return *cur_tag;
}

void
state::flush() {
  if(!source_process_ || processes_ == 1)
    gather(*this); // no MPI communication needed
  else
    exec::reduce_internal<gather, void, flecsi::mpi>(*this);
  tasks = 0;
}

void
state::send_to_one(bool last) {
  using util::mpi::test;

  std::unique_lock lk(packets_mutex_);

  if(source_process_ != 0 && processes_ > 1) {
    std::vector<int> offsets(process_ ? 0 : processes_);
    std::vector<std::byte> data, buffer;

    if(process_ != 0 && active_process())
      data = util::serial::put_tuple(packets_);

    int bytes = data.size();

    if(source_process_ == all_processes) {
      std::vector<int> sizes = offsets;
      test(MPI_Gather(
        &bytes, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));

      if(process_ == 0) {
        int sum{0};
        for(Color p = 0; p < processes_; ++p) {
          offsets[p] = sum;
          sum += sizes[p];
        } // for

        buffer.resize(sum);
      } // if

      test(MPI_Gatherv(data.data(),
        bytes,
        MPI_BYTE,
        buffer.data(),
        sizes.data(),
        offsets.data(),
        MPI_BYTE,
        0,
        MPI_COMM_WORLD));
    }
    else {
      if(process_ == 0) {
        util::mpi::recv(bytes, source_process_, 0);
        buffer.resize(bytes);
        util::mpi::recv(std::span(buffer), source_process_, 0);
      }
      else if(process_ == source_process_) {
        util::mpi::send(bytes, 0, 0, MPI_COMM_WORLD);
        util::mpi::send(std::span(data), 0, 0, MPI_COMM_WORLD);
      }
    }

    if(process_ == 0) {
      for(Color p = 1; p < processes_; ++p) {

        if(source_process_ == all_processes || p == source_process_) {
          auto remote_packets =
            util::serial::get1<decltype(packets_)>(buffer.data() + offsets[p]);

          packets_.insert(packets_.end(),
            std::move_iterator(remote_packets.begin()),
            std::move_iterator(remote_packets.end()));
        } // if
      } // for
    }
    else {
      packets_.clear();
    }
  }

  if(process_ == 0) {
    stop = last;
    lk.unlock();
    avail.notify_one();
  } // if
} // send_to_one

void
state::flush_packets() {
  decltype(packets_) work;
  bool running = true;
  while(running) {
    {
      std::unique_lock lk(packets_mutex_);
      if(packets_.empty() && !stop)
        avail.wait(lk); // spurious wakeups have no effect
      running = !stop;
      work.swap(packets_);
    }
    std::sort(work.begin(), work.end());

    for(auto & p : work) {
      stream_ << p.second;
    } // for

    work.clear();
  } // while
} // flush_packets

} // namespace flog
} // namespace flecsi

#endif // FLECSI_ENABLE_FLOG
