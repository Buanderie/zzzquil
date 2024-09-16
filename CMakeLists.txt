cmake_minimum_required(VERSION 3.15)

project(ZZZQUIL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

add_executable(zzzquil src/main.cc)
target_link_libraries( zzzquil liquid dlib lapack blas )