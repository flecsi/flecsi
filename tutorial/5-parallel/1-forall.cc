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

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "../4-data/canonical.hh"
#include "control.hh"

// this tutorial is based on a 04-data/3-dence.cc tutorial example
// here we will add several forall / parallel_for interfaces

using namespace flecsi;

canon::slot canonical;
canon::cslot coloring;

const field<double>::definition<canon, canon::cells> pressure;

void
init(canon::accessor<wo> t, field<double>::accessor<wo> p) {
  std::size_t off{0};
  for(const auto c : t.cells()) {
    p[c] = (off++) * 2.0;
  } // for
} // init

#if defined(FLECSI_ENABLE_KOKKOS)
void
modify1(canon::accessor<ro> t, field<double>::accessor<rw> p) {
  forall(c, t.cells(), "modify1") {
    p[c] += 1;
  }; // forall
} // modify

void
modify2(canon::accessor<ro> t, field<double>::accessor<rw> p) {
  flecsi::exec::parallel_for(
    t.cells(), KOKKOS_LAMBDA(auto c) { p[c] += 1; }, std::string("modify2"));
} // modifyendif

#endif

void
print(canon::accessor<ro> t, field<double>::accessor<ro> p) {
  std::size_t off{0};
  for(auto c : t.cells()) {
    flog(info) << "cell " << off++ << " has pressure " << p[c] << std::endl;
  } // for
} // print

int
advance() {
  coloring.allocate("test.txt");
  canonical.allocate(coloring.get());

  auto pf = pressure(canonical);

  // cpu task, default
  execute<init>(canonical, pf);
#if defined(FLECSI_ENABLE_KOKKOS)
  // accelerated task, will be executed on the Kokkos default execution space
  // In case of Kookos bult with GPU, default execution space will be GPU
  // We rely on Legion moving data between devices for the legion back-end and
  // UVM for the MPI back-end
  execute<modify1, default_accelerator>(canonical, pf);
  execute<modify2, default_accelerator>(canonical, pf);
#endif
  // cpu_task
  execute<print>(canonical, pf);

  return 0;
}
control::action<advance, cp::advance> advance_action;
