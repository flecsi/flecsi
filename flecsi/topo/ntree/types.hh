/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/topo/index.hh"
#include "flecsi/util/geometry/point.hh"

/// \cond core
namespace flecsi {
namespace topo {
/// \addtogroup ntree
/// \{

/// Mandatory information for entity to create the basic structure of the ntree.
template<Dimension DIM, typename T, class KEY>
class sort_entity
{
  /// Point type to represent coordinates
  using point_t = util::point<T, DIM>;
  /// Key used in the hashtable, filling curve.
  using key_t = KEY;
  using type_t = T;

public:
  /// Default entity constructor
  sort_entity() {}

  /// Get coordinates of the entity
  point_t coordinates() const {
    return coordinates_;
  }
  /// Get key of the entity
  key_t key() const {
    return key_;
  }
  /// Get id of the entity
  int64_t id() const {
    return id_;
  }
  /// Get mass of the entity
  type_t mass() const {
    return mass_;
  }
  /// Get the radius of thee entity
  type_t radius() const {
    return radius_;
  }
  /// Set the coordinates of the entity
  void set_coordinates(const point_t & coordinates) {
    coordinates_ = coordinates;
  }
  /// Set the key of the entity
  void set_key(const key_t & key) {
    key_ = key;
  }
  /// Set the id of the entity
  void set_id(const int64_t & id) {
    id_ = id;
  }
  /// Set the mass of the entity
  void set_mass(const type_t & mass) {
    mass_ = mass;
  }
  /// Set the radius of the entity
  void set_radius(const type_t & radius) {
    radius_ = radius;
  }
  /// Compare entities using key and id
  bool operator<(const sort_entity & s) const {
    return std::tie(key_, id_) < std::tie(s.key_, s.id_);
  }
  template<Dimension D, typename TY, class K>
  friend std::ostream & operator<<(std::ostream & os,
    const sort_entity<D, TY, K> & e);

private:
  point_t coordinates_;
  key_t key_;
  int64_t id_;
  type_t mass_;
  type_t radius_;
}; // class sort_entity

template<Dimension DIM, typename T, class KEY>
std::ostream &
operator<<(std::ostream & os, const sort_entity<DIM, T, KEY> & e) {
  os << "Coords: " << e.coordinates() << " Mass: " << e.mass()
     << " Radius: " << e.radius() << " Key: " << e.key() << " Id: " << e.id();
  return os;
}

/// Base type for an entry in the hashtable
template<Dimension DIM, typename T, class KEY>
class hcell_base_t
{

  const static Dimension dimension = DIM;
  using type_t = T;
  using key_t = KEY;

  enum type_displ : int {
    CHILD_DISPL = 0,
    LOCALITY_DISPL = 1 << dimension,
    REQUESTED_DISPL = (1 << dimension) + 2,
    NCHILD_RECV_DISPL = (1 << dimension) + 3
  };
  enum type_mask : int {
    CHILD_MASK = 0b11111111,
    LOCALITY_MASK = 0b11 << LOCALITY_DISPL,
    REQUESTED_MASK = 0b1 << REQUESTED_DISPL,
    NCHILD_RECV_MASK = 0b1111 << NCHILD_RECV_DISPL
  };
  enum type_locality : int { LOCAL = 0, NONLOCAL = 1, SHARED = 2 };

public:
  hcell_base_t() = default;

  hcell_base_t(const key_t & key) : key_(key) {}

  key_t key() const {
    return key_;
  }

  size_t ent_idx() const {
    assert(is_ent_);
    assert(!is_node_);
    return idx_;
  }
  size_t node_idx() const {
    assert(!is_ent_);
    assert(is_node_);
    return idx_;
  }
  void set_key(const key_t & key) {
    key_ = key;
  }
  void set_ent_idx(const int & idx) {
    is_ent_ = true;
    is_node_ = false;
    idx_ = idx;
  }
  void set_node_idx(const int & idx) {
    is_ent_ = false;
    is_node_ = true;
    idx_ = idx;
  }
  void set_node() {
    is_ent_ = false;
    is_node_ = true;
    idx_ = 0;
  }
  void unset() {
    is_ent_ = false;
    is_node_ = false;
    idx_ = 0;
  }
  void add_child(const int & c) {
    type_ |= (1 << c);
  }
  bool is_ent() const {
    return is_ent_;
  }

  bool is_node() const {
    return is_node_;
  }

  bool is_unset() {
    return !is_node_ && !is_ent_;
  }
  unsigned int type() {
    return type_;
  }

  bool has_child(const std::size_t & c) const {
    return type_ & (1 << c);
  }

  template<Dimension DD, typename TT, class KK>
  friend std::ostream & operator<<(std::ostream & os,
    const hcell_base_t<DD, TT, KK> & hb);

private:
  key_t key_;
  size_t idx_ = 0;
  bool is_ent_ = false;
  bool is_node_ = false;
  unsigned int type_ = 0;
};

template<Dimension D, typename T, class K>
std::ostream &
operator<<(std::ostream & os, const hcell_base_t<D, T, K> & hb) {
  hb.is_node() ? os << "hb: node " : os << "hb: ent ";
  os << hb.key_ << "-" << hb.idx_;
  return os;
}

template<Dimension, typename T, class KEY>
class node
{
  node() = default;
};
/// \}

} // namespace topo
} // namespace flecsi
/// \endcond