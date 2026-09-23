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

state::state(const config & cfg)
  : verb(cfg.verbose), serialization_interval_(cfg.serialization_interval),
    color_output_(cfg.color), strip_level_(cfg.strip_level),
    source_process_(cfg.process), process_(util::mpi::rank()),
    processes_(util::mpi::size()),
    commp(communicate() ? new comm(comm::world()) : nullptr) {
#if defined(FLOG_ENABLE_DEBUG)
  std::cerr << FLOG_COLOR_LTGRAY << "Flog: initializing runtime"
            << FLOG_COLOR_PLAIN << std::endl;
#endif

  // Because active tags are specified at runtime, it is
  // necessary to maintain a map of the compile-time registered
  // tag names to the id that they get assigned after the state
  // initialization (register_tag). This map will be used to populate
  // the tag_bitset_ for fast runtime comparisons of enabled tag groups.

  for(auto & tag : cfg.tags) {
#ifdef FLOG_ENABLE_DEBUG
    std::cerr << "Flog: active tag " << std::quoted(tag) << '\n';
#endif
    if(tag == "all")
      tag_bitset_.set();
    else if(const auto it = tag_map_.find(tag); it != tag_map_.end()) {
      tag_bitset_.set(it->second);
    }
    else {
      std::cerr << "FLOG WARNING: tag " << tag
                << " has not been registered. Ignoring this group..."
                << std::endl;
    }
  }

#if defined(FLOG_ENABLE_DEBUG)
  std::cerr << FLOG_COLOR_LTGRAY << "Flog: initializing mpi state"
            << FLOG_COLOR_PLAIN << std::endl;
#endif

  if(process_ == 0) {
    flusher_thread_ = std::thread(&state::flush_packets, std::ref(*this));
  } // if
}

state::~state() {
#if defined(FLOG_ENABLE_DEBUG)
  std::cerr << FLOG_COLOR_LTGRAY << "Flog: state destructor" << std::endl;
#endif
  send_to_one(true, MPI_COMM_WORLD);

  if(process_ == 0) {
    flusher_thread_.join();
  } // if
}

struct state::gather {
  static void task(state * s, comm::ref c) noexcept {
    s->send_to_one(false, c);
  }
};

void
state::flush() {
  if(communicate())
    exec::reduce_internal<gather::task, void, 0>(this, *commp);
  else
    send_to_one(false, MPI_COMM_NULL);
  tasks = 0;
}

void
state::send_to_one(bool last, MPI_Comm c) {
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
      test(MPI_Gather(&bytes, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, c));

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
        c));
    }
    else {
      if(process_ == 0) {
        util::mpi::recv(bytes, source_process_, 0, c);
        buffer.resize(bytes);
        util::mpi::recv(std::span(buffer), source_process_, 0, c);
      }
      else if(process_ == source_process_) {
        util::mpi::send(bytes, 0, 0, c);
        util::mpi::send(std::span(data), 0, 0, c);
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
