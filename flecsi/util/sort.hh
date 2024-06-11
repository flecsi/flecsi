#ifndef FLECSI_UTIL_SORT_HH
#define FLECSI_UTIL_SORT_HH

#include "flecsi/data/copy_plan.hh"
#include "flecsi/data/map.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/global.hh"

#include <algorithm>
#include <random>

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

namespace heap {

// The head value was changed
// Update the Heap
template<typename Iterator,
  typename Compare = std::less<typename Iterator::value_type>>
void
head_replace(Iterator first, Iterator end, Compare comp) {
  auto len = std::distance(first, end);
  if(len <= 1)
    return;
  decltype(len) nnode = 0;
  while(true) {
    const decltype(len) c = nnode * 2 + 1, c1 = c < len ? c : nnode,
                        c2 = c + 1 < len ? c + 1 : nnode;
    auto best = comp(first[c2], first[c1]) ? c1 : c2;
    if(best == nnode || !comp(first[nnode], first[best]))
      break;
    std::swap(first[nnode], first[best]);
    nnode = best;
  } // while
} // heap_head_replace

template<typename T, typename key_type>
void
merge(const T & ma, std::vector<key_type> & v) {

  std::size_t total = 0, cur = 0, size = 0;
  for(auto & m : ma) {
    total += m.span().size();
    if(m.span().size())
      ++size;
  }

  v.resize(total);
  if(total == 0)
    return;
  std::vector<std::size_t> heap(size);
  std::iota(heap.begin(), heap.end(), 0);
  auto h_b = heap.begin();
  auto h_e = heap.end();

  std::vector<decltype(ma[0].span().begin())> it;
  std::vector<decltype(ma[0].span().begin())> end_it;
  for(auto & m : ma)
    if(m.span().size()) {
      it.push_back(m.span().begin());
      end_it.push_back(m.span().end());
    }

  auto comp = [&](std::size_t idx1, std::size_t idx2) {
    return *it[idx1] > *it[idx2];
  };

  std::make_heap(h_b, h_e, comp);
  v[cur++] = *it[heap[0]];

  while(cur != total) {
    if(++it[heap[0]] == end_it[heap[0]])
      std::pop_heap(h_b, h_e--, comp);
    else
      head_replace(h_b, h_e, comp);
    v[cur++] = *it[heap[0]];
  } // while
}
} // namespace heap

struct sort_base {

protected:
  using hist_int_t = std::uint64_t;

  sort_base(std::size_t c) {
    colors = c;
    std::vector<std::size_t> sizes(colors, 0);
    probes_s.allocate(sizes);
  }

  struct min {
    template<class T>
    FLECSI_INLINE_TARGET static T combine(T a, T b) {
      return std::min(a, b);
    }
    template<class T>
    static constexpr T identity = std::numeric_limits<T>::max();
  }; // struct min

  struct max {
    template<class T>
    FLECSI_INLINE_TARGET static T combine(T a, T b) {
      return std::max(a, b);
    }
    template<class T>
    static constexpr T identity = std::numeric_limits<T>::min();
  }; // struct max

  template<typename T>
  static void index_sort(T * ptr, util::span<const std::size_t> schanges) {
    index_sort(reinterpret_cast<std::byte *>(ptr), schanges, sizeof(T));
  }

  static void index_sort(std::byte * ptr,
    util::span<const std::size_t> schanges,
    const int size) {
    std::vector<std::size_t> changes(schanges.begin(), schanges.end());
    std::vector<std::byte> data(size);
    std::byte * tmp = data.data();
    for(std::size_t i = 0; i < changes.size(); ++i) {
      if(i == changes[i])
        continue;
      memcpy(tmp, ptr + i * size, size);
      memcpy(ptr + i * size, ptr + changes[i] * size, size);
      memcpy(ptr + changes[i] * size, tmp, size);
      auto it = std::find(changes.begin(), changes.end(), i);
      if(it != changes.end())
        std::swap(*it, changes[i]);
    }
  } // index_sort

  static void copy_sizes_task(topo::resize::Field::accessor<wo> a,
    topo::resize::Field::accessor<ro> b) {
    a = b;
  } // copy_sizes_task

  static void update_sizes_copy_task(topo::resize::Field::accessor<wo> a,
    field<std::size_t>::accessor<ro> cpy) {
    a = cpy[color()];
  } // update_sizes_copy_task

  // Using reduction as a gather copy
  // We could use multi-accessor like in other places but chose to use global
  // reduction for simplicity
  static void compute_copy_task(field<std::size_t>::accessor<wo> transfers,
    data::reduction_accessor<exec::fold::sum, int> copy) {
    auto c = colors;
    std::size_t output = color() * c;
    for(unsigned int j = 0; j < c; ++j) {
      int count = 0;
      for(std::size_t i = 0; i < transfers.span().size(); ++i)
        if(transfers[i] == j)
          ++count;
      copy[output + j](count);
    }
  } // compute_copy_task

  static void init_hist_task(field<hist_int_t>::accessor<wo> v) {
    std::fill(v.span().begin(), v.span().end(), 0);
  } // init_hist_task

  static void init_copy_task(field<int>::accessor<wo> v) {
    std::fill(v.span().begin(), v.span().end(), 0);
  }
  static void init_sizes_task(field<std::size_t>::accessor<wo> v) {
    std::fill(v.span().begin(), v.span().end(), 0);
  }

  static inline const field<hist_int_t>::definition<topo::global> hist_g_f;
  static inline const field<int>::definition<topo::global> copy_g_f;
  static inline const field<std::size_t>::definition<topo::global> sizes_g_f;

  topo::global::slot hist_g_s;
  topo::global::slot copy_g_s;
  topo::global::slot sizes_g_s;

  struct sort_array_type {};

  using sort_array_t = topo::array<sort_array_type>;

  // Transfer indices
  sort_array_t::slot transfer_s;
  const static inline field<std::size_t>::definition<sort_array_t> transfer_f;

  // Indices for the sort
  sort_array_t::slot idx_s;
  const static inline field<std::size_t>::definition<sort_array_t> indices_f;

  sort_array_t::slot meta_s;
  sort_array_t::slot probes_s;

  struct sort_color : topo::specialization<topo::color, sort_color> {};
  sort_color::slot intervals_s;

  static inline std::size_t colors = 1;

}; // sort_base

// Intermediary struct to capture the privileges.
template<typename KeyType, PrivilegeCount PC>
struct sort_privilege : sort_base {

protected:
  using key_type = KeyType;

  struct interval {
    key_type lower, upper;
  };

  // Meta data
  struct meta {
    std::size_t initial;
    key_type min, max;
  };

  using sort_base::sort_base;

  static std::pair<key_type, key_type> check_sort_task(
    typename field<key_type>::template accessor1<privilege_repeat<rw, PC>>
      val) {
    return {val[0], val.span().back()};
  } // check_sort_task

  // The type of the "values" accessor is unknown in this task (std::byte used).
  // The type size is known with the size parameter. The values.span() end is
  // meaningless but the total number of elements correspond to the size of
  // changes.span().
  static void sort_others_task(
    field<std::byte, data::raw>::accessor1<privilege_repeat<rw, PC>> values,
    field<std::size_t>::accessor<ro> changes,
    const std::size_t size) {
    index_sort(values.span().data(), changes.span(), size);
  } // sort_others_task

  // The type of the "values" accessor is unknown in this task (std::byte used).
  // The type size is known with the size parameter. The values.span() end is
  // meaningless but the total number of elements correspond to the size of
  // changes.span().
  static void reorder_other_task(
    field<std::byte, data::raw>::accessor1<privilege_repeat<rw, PC>> values,
    field<std::size_t>::accessor<ro> changes,
    const std::size_t size) {
    auto ptr = values.span().data();
    for(std::size_t i = 0; i < changes.span().size(); ++i)
      memcpy(ptr + i * size, ptr + changes[i] * size, size);
  } // reorder_other_task

  static void sort_values_task(
    typename field<key_type>::template accessor1<privilege_repeat<rw, PC>> val,
    field<std::size_t>::accessor<wo> idx) {
    std::iota(idx.span().begin(), idx.span().end(), 0);
    std::stable_sort(idx.span().begin(),
      idx.span().end(),
      [&val](std::size_t i1, std::size_t i2) { return val[i1] < val[i2]; });
    index_sort<key_type>(val.span().data(), idx.span());
  } // sort_values_task

  static void reorder_values_task(
    typename field<key_type>::template accessor1<privilege_repeat<rw, PC>>
      values,
    typename field<interval>::template accessor<ro> intervals,
    field<std::size_t>::accessor<wo> changes) {
    // updates changes to indices
    constexpr key_type min = std::numeric_limits<key_type>::min();
    constexpr key_type max = std::numeric_limits<key_type>::max();

    std::iota(changes.span().begin(), changes.span().end(), 0);
    // Keep all the values between my threshold
    auto c = colors;
    std::size_t current = 0;
    key_type lower(color() ? intervals[color() - 1].lower : min);
    key_type upper(color() != c - 1 ? intervals[color()].lower : max);
    for(std::size_t i = 0; i < values.span().size(); i++) {
      if(values[i] > lower && values[i] <= upper) {
        values[current] = values[i];
        changes[current++] = i;
      } // if
    } // for
  } // reorder_values_task

  static void set_pointers_task(
    typename field<data::points::Value>::template accessor1<
      privilege_repeat<wo, PC>> a,
    field<int>::accessor<ro> copy,
    typename field<meta, data::single>::template accessor<wo> m) {
    auto c = colors;
    std::size_t cur = m->initial;
    for(unsigned int i = 0; i < c; ++i) {
      if(i == color())
        continue;
      std::size_t icur = 0;
      std::size_t ptr = c * i + color();
      // Sum color before my color sent
      // Basically count for each color
      for(unsigned int j = 0; j < color(); ++j)
        icur += copy[c * i + j];
      for(int j = 0; j < copy[ptr]; ++j)
        a(cur++) = data::points::make(i, icur++);
    }
  } // set_pointers_task

  static void update_transfer_task(
    typename field<key_type>::template accessor1<privilege_repeat<ro, PC>>
      values,
    typename field<interval>::template accessor<rw> intervals,
    data::reduction_accessor<exec::fold::sum, std::size_t> sizes,
    field<std::size_t>::accessor<wo> transfers) {
    // Count the number of values
    std::size_t j = 0, localsize = 0;
    for(std::size_t i = 0; i < values.span().size();) {
      if(j >= intervals.span().size() || values[i] <= intervals[j].lower) {
        ++localsize;
        transfers[i++] = j;
      }
      else {
        sizes[j++](localsize);
        localsize = 0;
      } // if
    } // for
    // Last value
    sizes[j](localsize);
  } // update_transfer_task

  // Compute the first samples
  // This function is used to count and to feed the probes (count = true/false)
  // In the first case the variable maybe_probes is not used.
  template<bool count>
  static std::size_t probes_task(
    typename field<key_type>::template accessor1<privilege_repeat<ro, PC>>
      values,
    std::conditional_t<count,
      topo::resize::Field::accessor<wo>,
      typename field<key_type>::template accessor<wo>> maybe_probes,
    typename field<interval>::template accessor<ro> intervals,
    const std::size_t totalents,
    const int iteration,
    const int iterations,
    const double epsilon) {
    if(values.span().size() == 0)
      return 0;
    std::minstd_rand mrnd(iteration + colors + color());
    constexpr double m_rnd = static_cast<double>(std::minstd_rand::max()) + 1;
    std::size_t nprobes = 0;

    const double ratio = (iteration + 1.) / iterations;
    auto c = colors;
    const double sj = std::pow((2. * std::log(c) / epsilon), ratio);
    const double proba = c * sj / totalents;

    std::size_t j = 0;
    for(unsigned int i = 0; i < c - 1; ++i) {
      for(; j < values.span().size() && values[j] < intervals[i].upper; ++j) {
        if(values[j] > intervals[i].lower) {
          if(mrnd() / m_rnd < proba) {
            if constexpr(count) {
              nprobes++;
            }
            else {
              maybe_probes[nprobes++] = values[j];
            }
          }
        } // if
      } // for
    } // for
    if constexpr(count)
      maybe_probes = nprobes;
    return nprobes;
  } // probes_task

  static std::size_t size_task(
    typename field<key_type>::template accessor1<privilege_repeat<ro, PC>> v) {
    return v.span().size();
  } // size_task

  static void histo_task(
    typename field<key_type>::template accessor1<privilege_repeat<ro, PC>> vals,
    data::multi<typename field<key_type>::template accessor<ro>> probes,
    data::reduction_accessor<exec::fold::sum, hist_int_t> histo) {

    std::vector<key_type> sorted_probes;
    sort_probes(probes, sorted_probes);

    std::vector<hist_int_t> local_histo(sorted_probes.size() + 1);
    std::size_t np = sorted_probes.size(), localsize = 0, j = 0;
    for(std::size_t i = 0; i < vals.span().size();) {
      if(j >= np || vals[i] <= sorted_probes[j]) {
        ++localsize;
        ++i;
      }
      else {
        local_histo[j++] = localsize;
        localsize = 0;
      }
    } // for
    local_histo[j] = localsize;

    std::partial_sum(
      local_histo.begin(), local_histo.end(), local_histo.begin());
    for(std::size_t i = 0; i < local_histo.size(); ++i)
      histo[i](local_histo[i]);
  } // histo_task

  static void init_meta_task(
    typename field<key_type>::template accessor1<privilege_repeat<ro, PC>> v,
    typename field<meta, data::single>::template accessor<wo> m) {
    constexpr key_type max = std::numeric_limits<key_type>::max();
    constexpr key_type min = std::numeric_limits<key_type>::min();
    m = {v.span().size(),
      v.span().size() ? v.span().front() : min,
      v.span().size() ? v.span().back() : max};
  } // init_meta_task

  // Subroutine to sort probes and output the result in a std::vector
  static void sort_probes(
    const data::multi<typename field<key_type>::template accessor<ro>> & probes,
    std::vector<key_type> & sorted_probes) {
    heap::merge(probes.accessors(), sorted_probes);
  } // sort_probes

  static void set_destination_task(
    field<data::intervals::Value>::accessor<wo> a,
    field<int>::accessor<ro> copy,
    typename field<meta, data::single>::template accessor<ro> m) {
    auto c = sort_base::colors;
    std::size_t total = 0;
    for(unsigned int i = 0; i < c; ++i)
      if(i != color())
        total += copy[c * i + color()];
    std::size_t start = m->initial;
    std::size_t stop = start + total;
    a(0) = data::intervals::make({start, stop}, color());
  } // set_destination_task

  static void update_sizes_task(topo::resize::Field::accessor<wo> a,
    field<int>::accessor<ro> cpy,
    typename field<meta, data::single>::template accessor<ro> m) {
    auto c = sort_base::colors;
    // Compute data that will be sent to me
    std::size_t total = m->initial;
    for(unsigned int i = 0; i < c; ++i) {
      if(i == color())
        continue;
      total += cpy[c * i + color()];
    }
    a = total;
  } // udpate_sizes_task

  static void update_bound_task(
    typename field<interval>::template accessor<rw> intervals,
    field<hist_int_t>::accessor<ro> histo,
    data::multi<typename field<key_type>::template accessor<ro>> p,
    const std::size_t totalents) {
    constexpr key_type max = std::numeric_limits<key_type>::max();
    constexpr key_type min = std::numeric_limits<key_type>::min();

    std::vector<key_type> sorted_probes;
    sort_probes(p, sorted_probes);

    auto c = sort_base::colors;
    std::vector<std::size_t> ideal(c, totalents / c);
    for(unsigned int i = 0; i < c; ++i) {
      if(totalents % (ideal[i] * c) > i)
        ++ideal[i];
      if(i > 0)
        ideal[i] += ideal[i - 1];
    }
    for(std::size_t i = 0; i < intervals.span().size(); ++i) {
      auto & [L, U] = intervals[i];
      if(L == U)
        continue;
      for(std::size_t j = 0; j < histo.span().size() - 1; ++j) {
        key_type low = j == 0 ? min : sorted_probes[j - 1];
        key_type high = j == histo.span().size() ? max : sorted_probes[j];
        if(histo[j] <= ideal[i] && low > L)
          L = low;
        if(histo[j] >= ideal[i] && high < U)
          U = high;
      } // for
    } // for
  } // update_bound_task

  static key_type reduce_min_meta_task(
    typename field<meta, data::single>::template accessor<ro> m) {
    return m->min;
  }

  static key_type reduce_max_meta_task(
    typename field<meta, data::single>::template accessor<ro> m) {
    return m->max;
  }

  static void init_intervals_task(
    typename field<interval>::template accessor<wo> intervals,
    key_type min,
    key_type max) {
    std::fill(
      intervals.span().begin(), intervals.span().end(), interval{min, max});
  } // init_intervals_task

  static void fake_initialize(
    typename field<key_type>::template accessor1<privilege_repeat<rw, PC>>) {}
  static void fake_initialize_others(
    field<std::byte, data::raw>::accessor1<privilege_repeat<rw, PC>>) {}

  // Intervals
  const static inline
    typename field<interval>::template definition<sort_base::sort_color>
      intervals_f;

  // Meta
  const static inline typename field<meta,
    data::single>::template definition<sort_base::sort_array_t>
    meta_f;

  // Probes
  const static inline
    typename field<key_type>::template definition<sort_base::sort_array_t>
      probes_f;

}; // sort_privilege

/// \cond core

/// Sort object implementing a distributed sort and load balancing
/// \tparam FieldRef The field reference of the field containing the keys.
template<typename FieldRef>
struct sort : sort_privilege<typename FieldRef::value_type,
                FieldRef::Topology::template privilege_count<FieldRef::space>> {

private:
  using meta = typename sort::meta;
  using interval = typename sort::interval;
  using key_type = typename FieldRef::value_type;
  using topology = typename FieldRef::Topology;
  using hist_int_t = typename sort_base::hist_int_t;

  static constexpr auto space = FieldRef::space;

public:
  /// The sort uses a copy plan on the index space of the provided
  /// field \p fr. This might prevent other copy plans from being used on this
  /// index space. By default all the fields on the same index space as the
  /// field reference \p fr will be sorted and resized. In order to ignore
  /// fields on this index space, one can specify the \p ignored_fields
  /// parameter with the ids of fields to not include in the sort. The fields
  /// that are not initialized need to be ignored.
  /// The sort triggers ghost copies as usual. If this is inappropriate it can
  /// be avoided by using the correct permissions on the task preceding the
  /// call to the sort.
  /// \param eps The percent of error in load balancing > 0.0
  /// \param ignored_fields Fields to be ignored during the sort on this index
  /// space
  sort(FieldRef fr,
    std::vector<field_id_t> ignored_fields = {},
    const double eps = 0.005)
    : sort::sort_privilege(fr.topology().colors()), values(fr), epsilon(eps),
      lm_probes(data::launch::make(sort::probes_s,
        data::launch::gather(sort_base::colors, sort_base::colors))) {

    std::vector<std::size_t> sizes(sort_base::colors, 0);
    sort::idx_s.allocate(sizes);
    sort::transfer_s.allocate(sizes);
    std::fill(sizes.begin(), sizes.end(), 1);
    sort::meta_s.allocate(sizes);
    std::fill(sizes.begin(), sizes.end(), sort_base::colors - 1);
    sort::intervals_s.allocate({sort_base::colors, sort_base::colors - 1});

    // Fields ignored for copies
    ignored_fields.insert(ignored_fields.end(),
      {values.fid(),
        sort::intervals_f.fid,
        sort::meta_f.fid,
        sort::probes_f.fid,
        sort::transfer_f.fid,
        sort::indices_f.fid,
        data::copy_plan::get_field_id<topology, space>()});
    for(auto & f : run::context::instance().field_info_store<topology, space>())
      if(std::find(ignored_fields.begin(), ignored_fields.end(), f->fid) ==
         ignored_fields.end())
        apply_fields.push_back(f.get());
  }

  /// Sort values of an index space based on a field in this index space.
  /// The partitions might be resized to fit the new distribution.
  /// Based on: Histogram Sort with Sampling (HSS) Birpul Harsh & al.
  void operator()() {
    auto & tt = values.topology();

    // Global sizes
    sort::sizes_g_s.allocate(sort_base::colors);
    execute<sort_base::init_sizes_task>(sort::sizes_g_f(sort::sizes_g_s));
    // Copy area
    sort::copy_g_s.allocate(sort_base::colors * sort_base::colors);
    execute<sort_base::init_copy_task>(sort::copy_g_f(sort::copy_g_s));

    std::vector<std::size_t> sizes(sort_base::colors, 0);

    // Compute total number of entities to sort
    auto fm_tsizes = reduce<sort::size_task, exec::fold::sum>(values);
    // Resize the index array to fit the values
    execute<sort::copy_sizes_task>(
      sort::idx_s->sizes(), tt.template get_partition<space>().sizes());
    sort::idx_s->resize();

    // Local sort
    execute<sort::sort_values_task>(values, sort::indices_f(sort::idx_s));
    // apply this displacement to all other fields
    // Apply the copy plan on all fields
    for(auto & af : apply_fields) {
      auto fr = data::field_reference<std::byte, data::raw, topology, space>(
        af->fid, tt);
      execute<sort::sort_others_task>(
        fr, sort::indices_f(sort::idx_s), af->type_size);
    }

    // Check if the array is already sorted
    bool sorted = true;
    auto fm_check = execute<sort::check_sort_task>(values);
    for(unsigned int j = 0; j < sort_base::colors - 1; ++j) {
      if(fm_check.get(j).second > fm_check.get(j + 1).first) {
        sorted = false;
        break;
      }
    }
    if(sorted) {
      return;
    }

    // Init meta data
    auto meta_fh = sort::meta_f(sort::meta_s);
    execute<sort::init_meta_task>(values, meta_fh);

    // Init intervals using the values (local min/max)
    auto intervals_fh = sort::intervals_f(sort::intervals_s);
    auto fm_min = reduce<sort::reduce_min_meta_task, sort_base::min>(meta_fh);
    auto fm_max = reduce<sort::reduce_max_meta_task, sort_base::max>(meta_fh);
    execute<sort::init_intervals_task>(
      intervals_fh, fm_min.get(), fm_max.get());

    int iterations = std::log(std::log(sort_base::colors) / epsilon);
    std::size_t tsizes = fm_tsizes.get();

    // Compute splitters
    for(int i = 0; i < iterations; ++i) {
      // Count number of probes: three steps, count + resize + fill
      std::size_t totalprobes =
        reduce<sort::template probes_task<true>, exec::fold::sum>(values,
          sort::probes_s->sizes(),
          intervals_fh,
          tsizes,
          i,
          iterations,
          epsilon)
          .get();
      if(totalprobes == 0)
        continue;
      // Resize probes array
      sort::probes_s->resize();

      // Sample probes
      execute<sort::template probes_task<false>>(values,
        sort::probes_f(sort::probes_s),
        intervals_fh,
        tsizes,
        i,
        iterations,
        epsilon);
      // Allocate histogram
      // Should not allocate but resize.
      sort::hist_g_s.allocate(totalprobes + 1);
      auto hist_fh = sort::hist_g_f(sort::hist_g_s);
      execute<sort::init_hist_task>(hist_fh);

      execute<sort::histo_task>(values, sort::probes_f(lm_probes), hist_fh);
      // Update bounds on each color
      execute<sort::update_bound_task>(
        intervals_fh, hist_fh, sort::probes_f(lm_probes), tsizes);
    } // for

    auto sizes_fh = sort::sizes_g_f(sort::sizes_g_s);
    auto copy_fh = sort::copy_g_f(sort::copy_g_s);
    // Transfer array (destination of the entities)
    // Need to be of the same size as the array of values to sort
    auto transfer_fh = sort::transfer_f(sort::transfer_s);
    // Copy the same sizes as the topology
    execute<sort::copy_sizes_task>(
      sort::transfer_s->sizes(), tt.template get_partition<space>().sizes());
    // Apply resize
    sort::transfer_s->resize();

    // Init transfer: who goes where from initial values + reduce sizes
    execute<sort::update_transfer_task>(
      values, intervals_fh, sizes_fh, transfer_fh);
    execute<sort::compute_copy_task>(transfer_fh, copy_fh);

    // Resize values' partition to have room for the copies
    // This could be changed to use a buffer
    execute<sort::update_sizes_task>(
      tt.template get_partition<space>().sizes(), copy_fh, meta_fh);
    tt.template get_partition<space>().resize();
    execute<sort::update_sizes_task>(sort::idx_s->sizes(), copy_fh, meta_fh);
    sort::idx_s->resize();

    execute<sort::fake_initialize>(values);
    for(auto & af : apply_fields) {
      auto fr = data::field_reference<std::byte, data::raw, topology, space>(
        af->fid, tt);
      execute<sort::fake_initialize_others>(fr);
    }

    // Create copy plan operation and issue
    auto dest = [&](auto f) {
      execute<sort::set_destination_task>(f, copy_fh, meta_fh);
    };
    auto src = [&](auto f) {
      execute<sort::set_pointers_task>(f, copy_fh, meta_fh);
    };
    data::copy_plan cp(tt,
      data::copy_plan::Sizes(sort_base::colors, 1),
      dest,
      src,
      util::constant<space>());

    cp.issue_copy(values.fid());

    // 1 Apply sort on values and keep track of changes
    execute<sort::reorder_values_task>(
      values, intervals_fh, sort::indices_f(sort::idx_s));

    // Apply the copy plan on all fields
    for(auto & af : apply_fields) {
      cp.issue_copy(af->fid);
      auto fr = data::field_reference<std::byte, data::raw, topology, space>(
        af->fid, tt);
      execute<sort::reorder_other_task>(
        fr, sort::indices_f(sort::idx_s), af->type_size);
    }

    // Resize
    execute<sort::update_sizes_copy_task>(
      tt.template get_partition<space>().sizes(), sizes_fh);
    tt.template get_partition<space>().resize();
    execute<sort::update_sizes_copy_task>(sort::idx_s->sizes(), sizes_fh);
    sort::idx_s->resize();

    execute<sort::fake_initialize>(values);
    for(auto & af : apply_fields) {
      auto fr = data::field_reference<std::byte, data::raw, topology, space>(
        af->fid, tt);
      execute<sort::fake_initialize_others>(fr);
    }

    execute<sort::sort_values_task>(values, sort::indices_f(sort::idx_s));

    for(auto & af : apply_fields) {
      auto fr = data::field_reference<std::byte, data::raw, topology, space>(
        af->fid, tt);
      execute<sort::sort_others_task>(
        fr, sort::indices_f(sort::idx_s), af->type_size);
    }
  } // sort

private:
  std::vector<const data::field_info_t *> apply_fields;
  FieldRef values;
  double epsilon;
  data::launch::mapping<sort_base::sort_array_t> lm_probes;

}; // sort
/// \endcond

/// \}
} // namespace util
} // namespace flecsi

#endif // FLECSI_UTIL_SORT_HH
