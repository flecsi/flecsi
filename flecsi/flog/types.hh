// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_TYPES_HH
#define FLECSI_LOG_TYPES_HH

#include "flecsi/config.hh"

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/flog/utils.hh"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

/// Specification for Flog operation.
struct config {
  /// Tags to enable (perhaps including "all").
  /// \showinitializer
  std::vector<std::string> tags{"all"};
  /// Verbosity level (suppresses decorations if negative).
  /// \showinitializer
  int verbose = 0,
      /// Process from which to produce output, or -1 for all.
      /// \showinitializer
    process = 0;
  /// Frequency of message serialization in number of tasks.
  /// \showinitializer
  unsigned serialization_interval = FLOG_SERIALIZATION_INTERVAL;
  /// Enable color output.
  /// \showinitializer
  bool color = FLOG_ENABLE_COLOR_OUTPUT;
  /// Flog strip level (0-4).
  /// \showinitializer
  int strip_level = FLOG_STRIP_LEVEL;
};

/// \cond core

/*!
  The tee_buffer_t type provides a stream buffer that allows output to
  multiple targets.
 */

class tee_buffer_t : public std::streambuf {
public:
  /*!
    The buffer_data_t type is used to hold state and the actual low-level
    stream buffer pointer.
   */

  struct buffer_data_t {
    bool colorized;
    std::streambuf * buffer;
  }; // struct buffer_data_t

  /*!
    Add a buffer to which output should be written.
   */

  void add_buffer(std::streambuf * sb, bool colorized) {
    buffers_.push_back({colorized, sb});
  } // add_buffer

protected:
  /*!
    Override the overflow method. This streambuf has no buffer, so overflow
    happens for every character that is written to the string, allowing
    us to write to multiple output streams. This method also detects
    colorization strings embedded in the character stream and removes
    them from output that is going to non-colorized buffers.

    \param c The character to write. This is passed in as an int so that
             non-characters like EOF can be written to the stream.
   */

  virtual int overflow(int c) {
    if(c == EOF) {
      return !EOF;
    }
    else {
      // Get the size before we add the current character
      const size_t tbsize = test_buffer_.size();

      // Buffer the output for now...
      test_buffer_.append(1, char(c)); // takes char

      switch(tbsize) {

        case 0:
          if(c == '\033') {
            // This could be a color string, start buffering
            return c;
          }
          else {
            // No match, go ahead and write the character
            return flush_buffer();
          } // if

        case 1:
          if(c == '[') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if

        case 2:
          if(c == '0' || c == '1') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if

        case 3:
          if(c == ';') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else if(c == 'm') {
            // This is a plain color termination. Write the
            // buffered output to the color buffers.
            return flush_buffer(color_buffers);
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if

        case 4:
          if(c == '3') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if

        case 5:
          if(isdigit(c) && (c - '0') < 8) {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if

        case 6:
          if(c == 'm') {
            // This is a color string termination. Write the
            // buffered output to the color buffers.
            return flush_buffer(color_buffers);
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer();
          } // if
      } // switch

      return c;
    } // if
  } // overflow

  /*!
    Override the sync method so that we sync all of the output buffers.
   */

  virtual int sync() {
    bool fail = false;

    for(const auto & b : buffers_) {
      if(b.buffer->pubsync())
        fail = true;
    } // for

    // Return -1 if one of the buffers had an error
    return -fail;
  } // sync

private:
  // Default predicate: flush unconditionally.
  int flush_buffer() {
    return flush_buffer([](const buffer_data_t &) { return true; });
  }

  // Predicate to select color buffers.
  static bool color_buffers(const buffer_data_t & bd) {
    return bd.colorized;
  } // any_buffer

  // Flush buffered output to buffers that satisfy the predicate function.
  template<typename P>
  int flush_buffer(P && predicate) {
    int eof = !EOF;

    // Put test buffer characters to each buffer
    for(const auto & b : buffers_) {
      if(predicate(b)) {
        for(auto bc : test_buffer_) {
          const int w = b.buffer->sputc(bc);
          if(eof != EOF)
            eof = w;
        } // for
      } // if
    } // for

    // Clear the test buffer
    test_buffer_.clear();

    // Return EOF if one of the buffers hit the end
    return eof == EOF ? EOF : !EOF;
  } // flush_buffer

  std::vector<buffer_data_t> buffers_;
  std::string test_buffer_;

}; // class tee_buffer_t

/*!
  The tee_stream_t type provides a stream class that writes to multiple
  output buffers.
 */

struct tee_stream_t : public std::ostream {

  tee_stream_t() : std::ostream(&tee_) {
    // Allow users to turn std::clog output on and off from
    // their environment.
    if(std::getenv("FLOG_ENABLE_STDLOG")) {
      tee_.add_buffer(std::clog.rdbuf(), true);
    } // if
  } // tee_stream_t

  /*!
    Add a new buffer to the output.
   */

  void add_buffer(std::ostream & s, bool colorized = false) {
    tee_.add_buffer(s.rdbuf(), colorized);
  } // add_buffer

private:
  tee_buffer_t tee_;

}; // struct tee_stream_t

/// \endcond
/// \}
} // namespace flog
} // namespace flecsi

#endif // FLECSI_ENABLE_FLOG

#endif
