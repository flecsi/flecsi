/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include <flecsi/execution.hh>
#include <flecsi/util/annotation.hh>

using namespace flecsi::util;

struct user_execution : annotation::context<user_execution> {
  static constexpr char name[] = "User-Execution";
};
