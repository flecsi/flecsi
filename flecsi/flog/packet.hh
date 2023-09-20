// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_PACKET_HH
#define FLECSI_LOG_PACKET_HH

#include "flecsi/config.hh"

#if defined(FLECSI_ENABLE_FLOG)

#if defined(FLOG_ENABLE_MPI)
#include <mpi.h>
#endif

#if defined(_MSC_VER)
#error "Need implementation for Windows"
#endif

#include <sys/time.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

/// \cond core
namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

/*!
  Packet type for serializing output from distributed-memory tasks.
 */

struct packet_t {
  static constexpr size_t sec_bytes = sizeof(time_t);
  static constexpr size_t usec_bytes = sizeof(suseconds_t);
  static constexpr size_t max_message_size = 4096;
  static constexpr size_t packet_bytes =
    sec_bytes + usec_bytes + max_message_size;

  packet_t(const char * msg = nullptr) {
    timeval stamp;
    if(gettimeofday(&stamp, NULL)) {
      std::cerr << "FLOG: call to gettimeofday failed!!! " << __FILE__
                << __LINE__ << std::endl;
      std::exit(1);
    } // if

    std::memcpy(bytes_.data(), &stamp.tv_sec, sec_bytes);
    std::memcpy(bytes_.data() + sec_bytes, &stamp.tv_usec, usec_bytes);

    std::ostringstream oss;
    if(msg)
      oss << msg;

    strcpy(bytes_.data() + sec_bytes + usec_bytes, oss.str().c_str());
  } // packet_t

  time_t seconds() const {
    return get<time_t>(bytes_.data());
  } // seconds

  suseconds_t useconds() const {
    return get<suseconds_t>(bytes_.data() + sec_bytes);
  } // seconds

  const char * message() {
    return bytes_.data() + sec_bytes + usec_bytes;
  } // message

  const char * data() const {
    return bytes_.data();
  } // data

  bool operator<(packet_t const & b) const {
    return this->seconds() == b.seconds() ? this->useconds() < b.useconds()
                                          : this->seconds() < b.seconds();
  } // operator <

private:
  template<class T>
  static T get(const void * p) {
    T ret;
    std::memcpy(&ret, p, sizeof ret);
    return ret;
  }

  std::array<char, packet_bytes> bytes_;

}; // packet_t

/// \}
} // namespace flog
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
