#------------------------------------------------------------------------------#
# Copyright (c) 2016, Triad National Security, LLC
# All rights reserved
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.12)
project(Flaxpy LANGUAGES CXX C)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

find_package(FleCSI REQUIRED)

add_executable(flaxpy flaxpy.cc)
target_link_libraries(flaxpy FleCSI::FleCSI)
