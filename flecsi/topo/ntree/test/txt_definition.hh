#ifndef FLECSI_TOPO_NTREE_TEST_TXT_DEFINITION_HH
#define FLECSI_TOPO_NTREE_TEST_TXT_DEFINITION_HH

#include <fstream>
#include <set>
#include <vector>

#include "flecsi/util/geometry/point.hh"

// This structure is used to store temporary information between a file and the
// N-Tree topology. It depends of the specialization or the use case. A similar
// structure is used in the N-Tree tutorial due to the similarity of the
// specializations.
template<flecsi::Dimension DIM, typename T, class KEY>
struct sort_entity {
  using point_t = flecsi::util::point<T, DIM>;
  using key_t = KEY;
  using type_t = T;

  sort_entity() {}
  bool operator<(const sort_entity & s) const {
    return std::tie(key_, id_) < std::tie(s.key_, s.id_);
  }

  key_t key_;
  int64_t id_;
  point_t coordinates_;
  type_t mass_;
  type_t radius_;
}; // class sort_entity

template<flecsi::Dimension DIM, typename T, class KEY>
std::ostream &
operator<<(std::ostream & os, const sort_entity<DIM, T, KEY> & e) {
  os << " Key: " << e.key_ << " Id: " << e.id_;
  return os;
}

template<typename KEY, flecsi::Dimension DIM>
class txt_definition {
public:
  static constexpr flecsi::Dimension dim = DIM;
  using key_t = KEY;
  using point_t = flecsi::util::point<double, DIM>;
  using ent_t = sort_entity<DIM, double, key_t>;
  using range_t = std::array<point_t, 2>;

  txt_definition(const std::string & filename, const int size) {
    read_sizes_(filename, size);
  }

  void read_entities(int c) {
    nlocal_entities_ = distribution_[c];
    entities_.resize(nlocal_entities_);

    const int lineC = 6; // 3 digits, 2 spaces, and newline
    const int lineR = 2; // 1 digit, and newline
    int position = 2 + lineC * offset_[c];
    myfile_.seekg(position);

    // Coordinates, ignore the other colors
    for(flecsi::util::id i = 0; i < nlocal_entities_; ++i) {
      for(flecsi::Dimension j = 0; j < dim; ++j)
        myfile_ >> entities_[i].coordinates_[j];
    }

    position = 2 + nglobal_entities_ * lineC + lineR * offset_[c];
    myfile_.seekg(position);

    // Radius
    for(flecsi::util::id i = 0; i < nlocal_entities_; ++i)
      myfile_ >> entities_[i].radius_;

    position = 2 + nglobal_entities_ * lineC + nglobal_entities_ * lineR +
               lineR * offset_[c];
    myfile_.seekg(position);

    // Mass
    for(flecsi::util::id i = 0; i < nlocal_entities_; ++i)
      myfile_ >> entities_[i].mass_;

    // Ids
    for(flecsi::util::gid i = offset_[c], k = 0; i < offset_[c + 1]; ++i, ++k)
      entities_[k].id_ = i;

    // Generate the keys
    for(flecsi::util::id i = 0; i < nlocal_entities_; ++i)
      entities_[i].key_ = key_t(range_, entities_[i].coordinates_);
  }

  flecsi::util::gid global_num_entities() const {
    return nglobal_entities_;
  }

  size_t distribution() const {
    return entities_.size();
  }

  std::pair<flecsi::util::gid, flecsi::util::gid> offset(const int & i) const {
    return std::pair(offset_[i], offset_[i + 1]);
  }

  std::vector<ent_t> & entities() {
    return entities_;
  }

  ent_t & entities(const int & i) {
    return entities_[i];
  }

private:
  void compute_range() {

    int position = 2;
    myfile_.seekg(position);

    point_t p;
    for(flecsi::Dimension j = 0; j < dim; ++j) {
      myfile_ >> p[j];
    }

    range_[0] = range_[1] = p;
    for(flecsi::util::gid i = 1; i < nglobal_entities_; ++i) {
      for(flecsi::Dimension j = 0; j < dim; ++j) {
        myfile_ >> p[j];
      }
      for(flecsi::Dimension d = 0; d < dim; ++d) {
        range_[1][d] = std::max(range_[1][d], p[d] + 1);
        range_[0][d] = std::min(range_[0][d], p[d] - 1);
      }
    }
  } // compute_range

  void read_sizes_(const std::string & filename, const int size) {
    // For now read all particles?
    myfile_ = std::ifstream(filename);
    if(myfile_.fail()) {
      std::cerr << "Cannot open file: " << filename << std::endl;
    }
    nglobal_entities_ = 0;
    myfile_ >> nglobal_entities_;

    offset_.resize(size + 1, 0);
    distribution_.resize(size, 0);
    const flecsi::util::id nlocal_entities = nglobal_entities_ / size,
                           lm = nglobal_entities_ % size;
    for(int i = 0; i < size; ++i) {
      distribution_[i] = nlocal_entities;
      if(i < lm)
        ++distribution_[i];
    }

    for(int i = 1; i < size + 1; ++i) {
      offset_[i] = distribution_[i - 1] + offset_[i - 1];
    }

    compute_range();
  }

  std::ifstream myfile_;
  range_t range_;
  std::vector<ent_t> entities_;
  flecsi::util::gid nglobal_entities_;
  flecsi::util::id nlocal_entities_;
  std::vector<flecsi::util::id> distribution_;
  std::vector<flecsi::util::gid> offset_;

}; // class txt_definition

#endif
