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
  if(sig.empty() || sig[sig.length() - 1] != ')') {
    return sig; // empty or nothing to remove
  }

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

std::string
strip_return_type(const std::string & sig) {
  std::string s = strip_parameter_list(sig);
  if(s.empty() || s[s.length() - 1] != '>') {
    return sig; // empty or not a template
  }

  int tlevel = 1, blevel = 0;
  std::size_t pos = s.length() - 2;

  while(pos > 0) {
    char c = s[pos--];
    switch(c) {
      case '<':
        if(!blevel)
          --tlevel;
        break;
      case '>':
        if(!blevel)
          ++tlevel;
        break;
      case '{':
      case '[':
      case '(':
        --blevel;
        break;
      case '}':
      case ']':
      case ')':
        ++blevel;
        break;
      default:
        if(tlevel == 0 && std::isspace(c))
          return sig.substr(pos + 2);
    }
  }
  return sig;
}

} // namespace util
} // namespace flecsi
