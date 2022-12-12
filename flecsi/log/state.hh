// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_STATE_HH
#define FLECSI_LOG_STATE_HH

#include <flecsi-config.h>

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/data/field_info.hh"
#include "flecsi/log/packet.hh"
#include "flecsi/log/types.hh"
#include "flecsi/log/utils.hh"

#include <atomic>
#include <bitset>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

/// \cond core
namespace flecsi {
namespace log {
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
  state(std::string active, int verbose, Color one_process) : verb(verbose) {
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

    if(active == "all") {
      // Turn on all of the bits for "all".
      tag_bitset_.set();
    }
    else if(active == "unscoped") {
      tag_bitset_.set(0);
    }
    else if(active != "none") {
      // Turn on the bits for the selected groups.
      std::istringstream is(active);
      std::string tag;
      while(std::getline(is, tag, ',')) {
        if(tag_map_.find(tag) != tag_map_.end()) {
          tag_bitset_.set(tag_map_[tag]);
        }
        else {
          std::cerr << "FLOG WARNING: tag " << tag
                    << " has not been registered. Ignoring this group..."
                    << std::endl;
        } // if
      } // while
    } // if

#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: active tags (" << active << ")"
              << FLOG_COLOR_PLAIN << std::endl;
#endif

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

    one_process_ = one_process;

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

  static int verbose() {
    return instance ? instance->verb : 0;
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
    assert(id < FLOG_TAG_BITS && "Tag bits overflow! Increase FLOG_TAG_BITS");
#if defined(FLOG_ENABLE_DEBUG)
    std::cerr << FLOG_COLOR_LTGRAY << "FLOG: registering tag " << tag << ": "
              << id << FLOG_COLOR_PLAIN << std::endl;
#endif
    tag_map_[tag] = id;
    tag_names.push_back(tag);
    return id;
  } // next_tag

  /*!
    Return a reference to the active tag (const version).
   */

  const std::atomic<size_t> & active_tag() const {
    return active_tag_;
  }

  /*!
    Return a reference to the active tag (mutable version).
   */

  std::atomic<size_t> & active_tag() {
    return active_tag_;
  }

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
    if(!instance)
      return "external";
    return tag_name(instance->active_tag_);
  }

  static bool tag_enabled() {
#if defined(FLOG_ENABLE_TAGS)
    // If the runtime context hasn't been initialized, return true only
    // if the user has enabled externally-scoped messages.
    if(!instance) {
#if defined(FLOG_ENABLE_EXTERNAL)
      return true;
#else
      return false;
#endif
    } // if

    const std::size_t t = instance->active_tag_;
    const bool ret = instance->tag_bitset_.test(t);

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
  bool one_process() const {
    return one_process_ < processes_;
  }

  Color process() const {
    return process_;
  }

  void buffer_output(std::string const & message) {
    std::string tmp = message;

    // Make sure that the string fits within the packet size.
    if(message.size() > FLOG_MAX_MESSAGE_SIZE) {
      tmp.resize(FLOG_MAX_MESSAGE_SIZE - 100);
      std::stringstream stream;
      stream << tmp << FLOG_COLOR_LTRED << " OUTPUT BUFFER TRUNCATED TO "
             << FLOG_MAX_MESSAGE_SIZE << " BYTES (" << message.size() << ")"
             << FLOG_COLOR_PLAIN << std::endl;
      tmp = stream.str();
    } // if

    std::lock_guard<std::mutex> guard(packets_mutex_);
    packets_.push_back({tmp.c_str()});
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

  static std::optional<state> instance;

private:
  int verb;

  tee_stream_t stream_;

  std::atomic<std::size_t> active_tag_ = 0;
  std::bitset<FLOG_TAG_BITS> tag_bitset_;
  static inline std::unordered_map<std::string, size_t> tag_map_;
  static inline std::vector<std::string> tag_names{"unscoped"};

#if defined(FLOG_ENABLE_MPI)
  void send_to_one(bool last);

  Color one_process_, process_, processes_;
  std::thread flusher_thread_;
  std::mutex packets_mutex_;
  std::condition_variable avail;
  std::vector<packet_t> packets_;
  bool stop = false;
#endif

}; // class state
inline std::optional<state> state::instance;

/// \}
} // namespace log
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
