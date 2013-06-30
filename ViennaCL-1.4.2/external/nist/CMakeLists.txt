cmake_minimum_required(VERSION 2.6)

project(nist CXX)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../cmake")
include(compilers)
set(SOURCES 
    mmio.cpp
)
include_directories( . ${timers_SOURCE_DIR})
add_library(${PROJECT_NAME} STATIC ${SOURCES})
