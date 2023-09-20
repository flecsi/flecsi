// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_LOG_TYPES_HH
#define FLECSI_LOG_TYPES_HH

#include "flecsi/config.hh"

#if defined(FLECSI_ENABLE_FLOG)

#include "flecsi/flog/packet.hh"
#include "flecsi/flog/utils.hh"

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

/// \cond core
namespace flecsi {
namespace flog {
/// \addtogroup flog
/// \{

/// Specification for Flog operation.
struct config {
  static inline unsigned default_serialization_interval =
    FLOG_SERIALIZATION_INTERVAL;
  static inline bool default_color_output = FLOG_ENABLE_COLOR_OUTPUT;
  static inline int default_strip_level = FLOG_STRIP_LEVEL;

  /// Tags to enable (perhaps including "all").
  /// Populated from \c \--flog-tags option.
  /// Empty if that option's argument is "none".
  std::vector<std::string> tags;
  /// Verbosity level (suppresses decorations if negative).  Populated
  /// from \c \--flog-verbose option.
  int verbose,
    /// Process from which to produce output, or -1 for all.
    /// Populated from \c \--flog-process option.
    process;
  /// Frequency of message serialization in number of tasks.
  unsigned serialization_interval = default_serialization_interval;
  /// Enable color output.
  bool color = default_color_output;
  /// FLOG strip level (0-4).
  int strip_level = default_strip_level;
};

/*!
  The tee_buffer_t type provides a stream buffer that allows output to
  multiple targets.
 */

class tee_buffer_t : public std::streambuf
{
public:
  /*!
    The buffer_data_t type is used to hold state and the actual low-level
    stream buffer pointer.
   */

  struct buffer_data_t {
    bool enabled;
    bool colorized;
    std::streambuf * buffer;
  }; // struct buffer_data_t

  /*!
    Add a buffer to which output should be written. This also enables
    the buffer,i.e., output will be written to it. For a given key,
    only the first call to this method will have an effect.
   */

  void add_buffer(std::string key, std::streambuf * sb, bool colorized) {
    buffers_.insert({key, {true, colorized, sb}});
  } // add_buffer

  /*!
    Enable a buffer so that output is written to it. This is mainly
    for buffers that have been disabled and need to be re-enabled.
   */

  void enable_buffer(std::string key) {
    buffers_.at(key).enabled = true;
  } // enable_buffer

  /*!
    Disable a buffer so that output is not written to it.
   */

  void disable_buffer(std::string key) {
    buffers_.at(key).enabled = false;
  } // disable_buffer

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
            return flush_buffer(all_buffers);
          } // if

        case 1:
          if(c == '[') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer(all_buffers);
          } // if

        case 2:
          if(c == '0' || c == '1') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer(all_buffers);
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
            return flush_buffer(all_buffers);
          } // if

        case 4:
          if(c == '3') {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer(all_buffers);
          } // if

        case 5:
          if(isdigit(c) && (c - '0') < 8) {
            // This still looks like a color string, keep buffering
            return c;
          }
          else {
            // This is some other kind of escape. Write the
            // buffered output to all buffers.
            return flush_buffer(all_buffers);
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
            return flush_buffer(all_buffers);
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
      if(b.second.buffer->pubsync())
        fail = true;
    } // for

    // Return -1 if one of the buffers had an error
    return -fail;
  } // sync

private:
  // Predicate to select all buffers.
  static bool all_buffers(const buffer_data_t & bd) {
    return bd.enabled;
  } // any_buffer

  // Predicate to select color buffers.
  static bool color_buffers(const buffer_data_t & bd) {
    return bd.enabled && bd.colorized;
  } // any_buffer

  // Flush buffered output to buffers that satisfy the predicate function.
  template<typename P>
  int flush_buffer(P && predicate = all_buffers) {
    int eof = !EOF;

    // Put test buffer characters to each buffer
    for(const auto & b : buffers_) {
      if(predicate(b.second)) {
        for(auto bc : test_buffer_) {
          const int w = b.second.buffer->sputc(bc);
          eof = (eof == EOF) ? eof : w;
        } // for
      } // if
    } // for

    // Clear the test buffer
    test_buffer_.clear();

    // Return EOF if one of the buffers hit the end
    return eof == EOF ? EOF : !EOF;
  } // flush_buffer

  std::unordered_map<std::string, buffer_data_t> buffers_;
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
      tee_.add_buffer("flog", std::clog.rdbuf(), true);
    } // if
  } // tee_stream_t

  /*!
    Add a new buffer to the output.
   */

  void add_buffer(std::string const & key,
    std::ostream & s,
    bool colorized = false) {
    tee_.add_buffer(key, s.rdbuf(), colorized);
  } // add_buffer

  /*!
    Enable an existing buffer.

    \param key The string identifier of the streambuf.
   */

  void enable_buffer(std::string const & key) {
    tee_.enable_buffer(key);
  } // enable_buffer

  /*!
    Disable an existing buffer.

    \param key The string identifier of the streambuf.
   */

  void disable_buffer(std::string const & key) {
    tee_.disable_buffer(key);
  } // disable_buffer

private:
  tee_buffer_t tee_;

}; // struct tee_stream_t

/// \}
} // namespace flog
} // namespace flecsi
/// \endcond

#endif // FLECSI_ENABLE_FLOG

#endif
