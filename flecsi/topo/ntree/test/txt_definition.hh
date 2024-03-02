#ifndef FLECSI_TOPO_NTREE_TEST_TXT_DEFINITION_HH
#define FLECSI_TOPO_NTREE_TEST_TXT_DEFINITION_HH

#include <fstream>
#include <set>
#include <vector>

#include "flecsi/util/geometry/point.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/sort.hh"

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

template<typename KEY, int DIM>
class txt_definition
{
public:
  const int dim = DIM;
  using key_t = KEY;
  using point_t = flecsi::util::point<double, DIM>;
  using ent_t = sort_entity<DIM, double, key_t>;
  using range_t = std::array<point_t, 2>;

  txt_definition(const std::string & filename) {
    const auto [rank, size] = flecsi::util::mpi::info();
    read_entities_(filename);
    // Compute the range
    mpi_compute_range(entities_, range_);
    // Generate the keys
    for(size_t i = 0; i < entities_.size(); ++i) {
      entities_[i].key_ = key_t(range_, entities_[i].coordinates_);
    }

    nlocal_entities_ = entities_.size();
    if(rank == 0)
      flog(info) << rank << ": Range: " << range_[0] << ";" << range_[1]
                 << std::endl;
  }

  size_t global_num_entities() const {
    return nglobal_entities_;
  }

  size_t distribution() const {
    return entities_.size();
  }

  std::pair<size_t, size_t> offset(const int & i) const {
    return std::pair(offset_[i], offset_[i + 1]);
  }

  std::vector<ent_t> & entities() {
    return entities_;
  }

  ent_t & entities(const int & i) {
    return entities_[i];
  }

private:
  void mpi_compute_range(const std::vector<ent_t> & ents, range_t & range) {

    // Compute the local range
    range[0] = range[1] = ents.front().coordinates_;

    for(size_t i = 1; i < ents.size(); ++i) {
      for(int d = 0; d < dim; ++d) {
        range[1][d] =
          std::max(range[1][d], ents[i].coordinates_[d] + ents[i].radius_);
        range[0][d] =
          std::min(range[0][d], ents[i].coordinates_[d] - ents[i].radius_);
      }
    }
    // Do the MPI Reduction
    flecsi::util::mpi::test(MPI_Allreduce(
      MPI_IN_PLACE, &(range[1][0]), dim, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    flecsi::util::mpi::test(MPI_Allreduce(
      MPI_IN_PLACE, &(range[0][0]), dim, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));

  } // mpi_compute_range

  void read_entities_(const std::string & filename) {
    const auto [rank, size] = flecsi::util::mpi::info();
    // For now read all particles?
    std::ifstream myfile(filename);
    if(myfile.fail()) {
      std::cerr << "Cannot open file: " << filename << std::endl;
    }
    nglobal_entities_ = 0;
    myfile >> nglobal_entities_;

    offset_.resize(size + 1, 0);
    distribution_.resize(size, 0);
    // Entities per ranks
    int nlocal_entities = nglobal_entities_ / size;
    int lm = nglobal_entities_ % size;
    for(int i = 0; i < size; ++i) {
      distribution_[i] = nlocal_entities;
      if(i < lm)
        ++distribution_[i];
    }

    for(int i = 1; i < size + 1; ++i) {
      offset_[i] = distribution_[i - 1] + offset_[i - 1];
    }

    if(rank == 0) {
      flog(info) << "Global entities: " << nglobal_entities_ << std::endl;
      std::ostringstream oss;
      oss << "Distribution:";
      for(int i = 0; i < size; ++i) {
        oss << " " << i << ":" << distribution_[i];
      }
      flog(info) << oss.str() << std::endl;
      oss.str("");
      oss.clear();
      oss << "Offset:";
      for(int i = 0; i < size + 1; ++i) {
        oss << " " << i << ":" << offset_[i];
      }
      flog(info) << oss.str() << std::endl;
    }

    nlocal_entities_ = distribution_[rank];

    entities_.resize(nlocal_entities_);

    // Coordinates, ignore the other ranks
    int k = 0;
    for(size_t i = 0; i < nglobal_entities_; ++i) {
      point_t p;
      for(int j = 0; j < dim; ++j) {
        myfile >> p[j];
      }
      if(i >= offset_[rank] && i < offset_[rank + 1])
        entities_[k++].coordinates_ = p;
    }

    // Radius
    k = 0;
    for(size_t i = 0; i < nglobal_entities_; ++i) {
      double r;
      myfile >> r;
      if(i >= offset_[rank] && i < offset_[rank + 1])
        entities_[k++].radius_ = r;
    }

    // Mass
    k = 0;
    for(size_t i = 0; i < nglobal_entities_; ++i) {
      double m;
      myfile >> m;
      if(i >= offset_[rank] && i < offset_[rank + 1])
        entities_[k++].mass_ = m;
    }

    k = 0;
    for(size_t i = 0; i < nglobal_entities_; ++i) {
      if(i >= offset_[rank] && i < offset_[rank + 1])
        entities_[k++].id_ = i;
    }

    myfile.close();
  }

  range_t range_;
  std::vector<ent_t> entities_;
  size_t nglobal_entities_;
  size_t nlocal_entities_;
  std::vector<flecsi::util::id> distribution_;
  std::vector<flecsi::util::id> offset_;

}; // class txt_definition

#endif
