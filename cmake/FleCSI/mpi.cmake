macro(flecsi_enable_mpi target ENABLE_CXX_BINDINGS)
  find_package(Threads REQUIRED)
  target_link_libraries(${target} PUBLIC Threads::Threads)

  if(${ENABLE_CXX_BINDINGS})
    find_package(MPI COMPONENTS CXX MPICXX REQUIRED)
    set(MPI_LANGUAGE CXX)
  else()
    find_package(MPI COMPONENTS CXX REQUIRED)
    set(MPI_LANGUAGE C)
  endif()
  target_link_libraries(${target} PUBLIC MPI::MPI_CXX)

  if(MSVC)
    target_compile_definitions(${target} PUBLIC
      _CRT_SECURE_NO_WARNINGS
      NOMINMAX)
  endif()

endmacro()
