#include <flecsi/data.hh>
#include <flecsi/execution.hh>

#include "../3-execution/control.hh"
#include "canonical.hh"

using namespace flecsi;

using ints = field<int, data::ragged>;
const ints::definition<canon, canon::cells> rag;

// This special field type is predefined for this purpose.
void
allocate(topo::resize::Field::accessor<wo> a) {
  a = 6;
}

// Use wo to initialize any field.
void
init(ints::mutator<wo> m) {
  int i = 0;
  for(auto r : m) {
    r.resize(i, i);
    ++i;
  }
}

// Accessors can modify but not create or destroy values.
int
total(ints::accessor<ro> a) {
  int ret = 0;
  for(auto r : a)
    for(auto i : r)
      ret += i;
  return ret;
}

void
advance(control_policy &) {
  canon::slot mesh;
  mesh.allocate(canon::mpi_coloring("4"));

  const auto f = rag(mesh);

  auto & elem = f.get_elements();
  execute<allocate>(elem.sizes());
  elem.resize();
  execute<init>(f);
  if(reduce<total, exec::fold::sum>(f).get() != 14)
    throw control_policy::exception{1};
} // advance()
control::action<advance, cp::advance> advance_action;
