#include "flecsi/util/unit.hh"

using namespace flecsi;

int
log_driver() {
  UNIT() {
    {
      std::vector<std::size_t> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.emplace_back(i);
      }
      flog(info) << flog::container{v} << std::endl;
    }

    {
      std::vector<std::vector<std::size_t>> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.push_back({0, 1, 2});
      }
      flog(info) << flog::container{v} << std::endl;
    }

    {
      std::map<std::size_t, std::size_t> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = i;
      }
      flog(info) << flog::container{m} << std::endl;
    }

    {
      std::map<std::size_t, std::vector<std::size_t>> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = {0, 1, 2};
      }
      flog(info) << flog::container{m} << std::endl;
    }
  };
} // flog

util::unit::driver<log_driver> driver;
