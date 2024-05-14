// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_STATE_HH
#define FLECSI_LOG_STATE_HH

#include "flecsi/config.hh"

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/data/field_info.hh"
#include "flecsi/flog/types.hh"
#include "flecsi/flog/utils.hh"

#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(FLOG_ENABLE_MPI)
#include <mpi.h>
#endif

/// \cond core
namespace flecsi {
template<class>
struct task_local;

namespace flog {
/// \addtogroup flog
/// \{

/*!
  The state type provides access to logging parameters and configuration.

  This type provides access to the underlying logging parameters for
  configuration and information. The FleCSI logging functions provide
  basic logging with an interface that is similar to Google's GLOG
  and the Boost logging utilities.

  @note We may want to consider adopting one of these packages
  in the future.
 */

class state
{
public:
  static constexpr std::size_t tag_bits = 1024;
  static constexpr Color all_processes = -1;

  state(const std::vector<std::string> & active,
    int verbose,
    Color source_process)
    : verb(verbose) {
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: initializing runtime"
              << FLOG_COLOR_PLAIN << std::endl;
#endif

#if defined(FLOG_ENABLE_TAGS)
    // Because active tags are specified at runtime, it is
    // necessary to maintain a map of the compile-time registered
    // tag names to the id that they get assigned after the state
    // initialization (register_tag). This map will be used to populate
    // the tag_bitset_ for fast runtime comparisons of enabled tag groups.

    // Note: For the time being, the map uses actual strings rather than
    // hashes. We should consider creating a const_string_t type for
    // constexpr string creation.

    for(auto & tag : active) {
#ifdef FLOG_ENABLE_DEBUG
      std::cerr << "FLOG: active tag " << std::quoted(tag) << '\n';
#endif
      if(tag == "all")
        tag_bitset_.set();
      else if(tag_map_.find(tag) != tag_map_.end()) {
        tag_bitset_.set(tag_map_[tag]);
      }
      else {
        std::cerr << "FLOG WARNING: tag " << tag
                  << " has not been registered. Ignoring this group..."
                  << std::endl;
      }
    }
#else
    (void)active;
#endif // FLOG_ENABLE_TAGS

#if defined(FLOG_ENABLE_MPI)

#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: initializing mpi state"
              << FLOG_COLOR_PLAIN << std::endl;
#endif

    {
      int p, np;
      MPI_Comm_rank(MPI_COMM_WORLD, &p);
      MPI_Comm_size(MPI_COMM_WORLD, &np);
      process_ = p;
      processes_ = np;
    }

    source_process_ = source_process;

    if(process_ == 0) {
      flusher_thread_ = std::thread(&state::flush_packets, std::ref(*this));
    } // if
#endif // FLOG_ENABLE_MPI
  }
  state(state &&) = delete; // address is known to the thread

  ~state() {
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: state destructor" << std::endl;
#endif
#if defined(FLOG_ENABLE_MPI)
    send_to_one(true);

    if(process_ == 0) {
      flusher_thread_.join();
    } // if
#endif // FLOG_ENABLE_MPI
  } // finalize

  int verbose() {
    return verb;
  }

  /*!
    Return the tag map.
   */

  static const std::unordered_map<std::string, size_t> & tag_map() {
    return tag_map_;
  }

  /*!
    Return the log stream.
   */

  std::ostream & stream() {
    return stream_;
  }

  /*!
    Return the tee stream to allow the user to set configuration options.
    FIXME: Need a better interface for this...
   */

  tee_stream_t & config_stream() {
    return stream_;
  }

  /*!
    Return the next tag id.
   */

  static std::size_t register_tag(const char * tag) {
    // If the tag is already registered, just return the previously
    // assigned id. This allows tags to be registered in headers.
    if(tag_map_.find(tag) != tag_map_.end()) {
      return tag_map_[tag];
    } // if

    const size_t id = tag_names.size();
    assert(id < tag_bits && "Tag bits overflow! Increase state::tag_bits");
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: registering tag " << tag << ": "
              << id << FLOG_COLOR_PLAIN << std::endl;
#endif
    tag_map_[tag] = id;
    tag_names.push_back(tag);
    return id;
  } // next_tag

  /*!
    Return a reference to the active tag.
   */

  static std::size_t & active_tag();

  /*!
    Return the tag name associated with a tag id.
   */

  static std::string tag_name(size_t id) {
    return tag_names.at(id);
  }

  /*!
    Return the tag name associated with the active tag.
   */

  static std::string active_tag_name() {
    return tag_name(active_tag());
  }

  static bool tag_enabled() {
#if defined(FLOG_ENABLE_TAGS)
    const std::size_t t = active_tag();
    const bool ret = instance().tag_bitset_.test(t);

#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: tag " << t << " is "
              << (ret ? "true" : "false") << FLOG_COLOR_PLAIN << std::endl;
#endif
    return ret;
#else
    return true;
#endif // FLOG_ENABLE_TAGS
  } // tag_enabled

  static std::size_t lookup_tag(const char * tag) {
    if(tag_map_.find(tag) == tag_map_.end()) {
      std::cerr << FLOG_COLOR_YELLOW << "FLOG: !!!WARNING " << tag
                << " has not been registered. Ignoring this group..."
                << FLOG_COLOR_PLAIN << std::endl;
      return 0;
    } // if

    return tag_map_[tag];
  }

#if defined(FLOG_ENABLE_MPI)
  using clock = std::chrono::system_clock;
  using packet_t = std::pair<std::chrono::time_point<clock>, std::string>;

  bool active_process() const {
    return source_process_ == all_processes || source_process_ == process_;
  }

  Color source_process() const {
    return source_process_;
  }

  Color process() const {
    return process_;
  }

  Color processes() const {
    return processes_;
  }

  void buffer_output(std::string message) {
    std::lock_guard<std::mutex> guard(packets_mutex_);
    packets_.emplace_back(clock::now(), std::move(message));
  }

  std::vector<packet_t> & packets() {
    return packets_;
  }

  void flush_packets();

  // Can be used as MPI tasks:

  /// Return number of buffered packets.
  static std::size_t log_size(const state & s) {
    return s.packets_.size();
  }
  /// Gather log output on the root.
  static void gather(state & s) {
    s.send_to_one(false);
  }
#endif

  static state & instance() {
    return instance_.value();
  }

  static void reset_instance() {
    instance_.reset();
  }

  static void set_instance(const std::vector<std::string> & active,
    int verbose,
    Color source_process) {
    instance_.emplace(active, verbose, source_process);
  }

private:
  int verb;

  tee_stream_t stream_;

  static std::optional<state> instance_;

#ifdef FLOG_ENABLE_TAGS
  static task_local<std::size_t> cur_tag;
  std::bitset<tag_bits> tag_bitset_;
#endif
  static inline std::unordered_map<std::string, size_t> tag_map_;
  static inline std::vector<std::string> tag_names;

#if defined(FLOG_ENABLE_MPI)
  void send_to_one(bool last);

  Color source_process_, process_, processes_;
  std::thread flusher_thread_;
  std::mutex packets_mutex_;
  std::condition_variable avail;
  std::vector<packet_t> packets_;
  bool stop = false;
#endif

}; // class state
inline std::optional<state> state::instance_;

/// \}
} // namespace flog
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
