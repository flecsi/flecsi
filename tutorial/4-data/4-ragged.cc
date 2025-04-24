#include <flecsi/data.hh>
#include <flecsi/execution.hh>

#include "../3-execution/control.hh"
#include "canonical.hh"

using namespace flecsi;

using ints = field<int, data::ragged>;
const ints::definition<canon, canon::cells> rag;

// This special field type is predefined for this purpose.
void
allocate(topo::resize::Field::accessor<wo> a) noexcept {
  a = 6;
}

// Use wo to initialize any field.
void
init(ints::mutator<wo> m) noexcept {
  int i = 0;
  for(auto r : m) {
    r.resize(i, i);
    ++i;
  }
}

// Accessors can modify but not create or destroy values.
int
total(ints::accessor<ro> a) noexcept {
  int ret = 0;
  for(auto r : a)
    for(auto i : r)
      ret += i;
  return ret;
}

void
advance(control_policy & p) {
  auto & s = p.scheduler();

  canon::slot mesh;
  mesh.allocate(s, canon::mpi_coloring(s, "4"));

  const auto f = rag(mesh);

  auto & elem = f.get_elements();
  s.execute<allocate>(elem.sizes());
  elem.resize();
  s.execute<init>(f);
  if(s.reduce<total, exec::fold::sum>(f).get() != 14)
    throw control_policy::exception{1};
} // advance()
control::action<advance, cp::advance> advance_action;
