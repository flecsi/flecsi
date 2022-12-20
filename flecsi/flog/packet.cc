#include <flecsi-config.h>

#include "flecsi/flog/packet.hh"
#include "flecsi/flog/state.hh"
#include "flecsi/flog/types.hh"

#include <chrono>
#include <thread>

#if defined(FLECSI_ENABLE_FLOG)

#if defined(_WIN32)
#include <sys/timeb.h>
int
gettimeofday(struct timeval * t, void * timezone) {
  struct _timeb timebuffer;
  _ftime(&timebuffer);
  t->tv_sec = timebuffer.time;
  t->tv_usec = 1000 * timebuffer.millitm;
  return 0;
}
#else
#include <sys/time.h>
#endif

namespace flecsi {
namespace flog {

#if defined(FLOG_ENABLE_MPI)
void
flush_packets() {
  while(state::instance().run_flusher()) {
    std::this_thread::sleep_for(
      std::chrono::microseconds(FLOG_PACKET_FLUSH_INTERVAL));
    std::lock_guard<std::mutex> guard(state::instance().packets_mutex());

    if(state::instance().serialized()) {
      if(state::instance().packets().size()) {
        std::sort(state::instance().packets().begin(),
          state::instance().packets().end());

        for(auto & p : state::instance().packets()) {
          state::instance().stream() << p.message();
        } // for

        state::instance().packets().clear();
      } // if
    } // if
  } // while
} // flush_packets
#endif // FLOG_ENABLE_MPI

} // namespace flog
} // namespace flecsi

#endif // FLECSI_ENABLE_FLOG
