macro(flecsi_enable_mpi target)
  find_package(Threads REQUIRED)
  target_link_libraries(${target} PUBLIC Threads::Threads)

  find_package(MPI COMPONENTS CXX REQUIRED)
  target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
endmacro()
