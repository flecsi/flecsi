#include "flecsi/util/demangle.hh"

#include <memory>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace flecsi {
namespace util {

std::string
demangle(const char * const name) {
#if defined(__GNUG__)
  int status = -4;
  std::unique_ptr<char, void (*)(void *)> res{
    abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
  if(status == 0)
    return res.get();
#endif
  // does nothing if not __GNUG__, or if abi::__cxa_demangle failed
  return name;
} // demangle

std::string
strip_parameter_list(const std::string & sig) {
  int level = 1;
  std::size_t pos = sig.length() - 2;

  while(pos > 0 && level > 0) {
    char c = sig[pos--];
    if(c == '(')
      --level;
    else if(c == ')')
      ++level;
  }

  if(level == 0) {
    return sig.substr(0, pos + 1);
  }
  return sig;
}

} // namespace util
} // namespace flecsi
