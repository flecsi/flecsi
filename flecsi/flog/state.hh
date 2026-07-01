// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_STATE_HH
#define FLECSI_LOG_STATE_HH

#include "flecsi/config.hh"

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/flog/types.hh"
#include "flecsi/flog/utils.hh"
#include "flecsi/util/common.hh" // convert
#include "flecsi/util/types.hh" // Color

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

#include <mpi.h>

/// \cond core
namespace flecsi {
template<class>
struct task_local;
struct comm;

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

class state {
public:
  static constexpr std::size_t tag_bits = 1024;
  static constexpr Color all_processes = -1;

  explicit state(const config &);
  state(state &&) = delete; // address is known to the thread
  ~state();

  int verbose() const {
    return verb;
  }

  /*!
    Return \c true if ANSI color support for messages was requested.  Note
    that this is a \c static method.  If the (singleton) \ref state has not
    yet been constructed, return the compile-time default colorization flag
    \c FLOG_ENABLE_COLOR_OUTPUT.  Otherwise, return the value specified
    at run time by \ref config.color.
  */
  static bool color_output() {
    return instance_.has_value() ? instance_.value().color_output_
                                 : bool(FLOG_ENABLE_COLOR_OUTPUT);
  }

  int & strip_level() {
    return strip_level_;
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
    return tag_map_
      .try_emplace(tag, util::convert{[&] {
        const size_t id = tag_names.size();
        assert(id < tag_bits && "Tag bits overflow! Increase state::tag_bits");
#if defined(FLOG_ENABLE_DEBUG)
        std::cerr << FLOG_COLOR_LTGRAY << "Flog: registering tag " << tag
                  << ": " << id << FLOG_COLOR_PLAIN << std::endl;
#endif
        tag_names.push_back(tag);
        return id;
      }})
      .first->second;
  }

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
    const std::size_t t = active_tag();
    const bool ret = instance().tag_bitset_.test(t);

#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "Flog: tag " << t << " is "
              << (ret ? "true" : "false") << FLOG_COLOR_PLAIN << std::endl;
#endif
    return ret;
  } // tag_enabled

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

  void flush();
  void count_tasks(unsigned n) {
    if((tasks += n) >= serialization_interval_)
      flush();
  }
  [[nodiscard]] unsigned restart_count() {
    return std::exchange(tasks, 0);
  }

  static state & instance() {
    return instance_.value();
  }

  static void reset_instance() {
    instance_.reset();
  }

  static void set_instance(const config & c) {
    instance_.emplace(c);
  }

private:
  int verb;
  unsigned serialization_interval_, tasks = 0;
  bool color_output_;
  int strip_level_;

  tee_stream_t stream_;

  static std::optional<state> instance_;

  static task_local<std::size_t> cur_tag;
  std::bitset<tag_bits> tag_bitset_;
  static inline std::unordered_map<std::string, size_t> tag_map_;
  static inline std::vector<std::string> tag_names;

  using clock = std::chrono::system_clock;
  using packet_t = std::pair<std::chrono::time_point<clock>, std::string>;

  bool communicate() const {
    return source_process_ && processes_ > 1;
  }
  void flush_packets();
  void send_to_one(bool last, MPI_Comm);

  Color source_process_, process_, processes_;
  std::thread flusher_thread_;
  std::mutex packets_mutex_;
  std::condition_variable avail;
  std::vector<packet_t> packets_;
  bool stop = false;

  // To avoid circularity:
  struct gather;
  std::unique_ptr<comm> commp;
}; // class state
inline std::optional<state> state::instance_;

/// \}

namespace detail {

inline const char *
use_color(const char * c) {
  return state::color_output() ? c : "";
}

} // namespace detail

} // namespace flog
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
