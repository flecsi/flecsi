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

/*! @file */

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include "flecsi/run/context.hh"

#include <boost/program_options.hpp>

#include <map>

namespace flecsi::run {

struct context_t : context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  /*!
    Documentation for this interface is in the top-level context type.
   */

  int initialize(int argc, char ** argv, bool dependent);

  /*!
    Documentation for this interface is in the top-level context type.
   */

  void finalize();

  /*!
    Documentation for this interface is in the top-level context type.
   */

  int start(const std::function<int()> &);

  /*!
    Documentation for this interface is in the top-level context type.
   */

  Color process() const {
    return process_;
  }

  Color processes() const {
    return processes_;
  }

  Color threads_per_process() const {
    return threads_per_process_;
  }

  Color threads() const {
    return threads_;
  }

  /*
    Documentation for this interface is in the top-level context type.
   */

  static int task_depth() {
    return 0;
  } // task_depth

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color color() const {
    return process_;
  }

  /*
    Documentation for this interface is in the top-level context type.
   */

  Color colors() const {
    return processes_;
  }
};

} // namespace flecsi::run
