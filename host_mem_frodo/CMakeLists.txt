cmake_minimum_required(VERSION 2.6)
set(CMAKE_C_COMPILER "icc")                                                                                       
set(CMAKE_CXX_COMPILER "icpc")                                                                                       
#set(CMAKE_CXX_COMPILER /opt/intel/composer_xe_2013.5.192/bin/intel64/icpc)

project(mem_tests_host)

set(D ../src_template/)
set(SOURCES 
    ${D}/bandwidth_tests_host.cpp
    ${D}/projectsettings.cpp 
    ${D}/timer_eb.cpp 
    ${D}/util.cpp 
)

include_directories( . 
    ../include_template 
    #../ViennaCL-1.4.2
) 

add_definitions(-std=c++0x -O2 -openmp)
#add_library(${PROJECT_NAME} STATIC ${SOURCES})

set(USRLIBDIR /usr/lib64)
FIND_LIBRARY(LIBTBB tbb_preview PATH /opt/intel/composer_xe_2013.1.117/tbb/lib/intel64/)
FIND_LIBRARY(LIBOMP iomp5 PATH /opt/intel/composer_xe_2013.5.192/compiler/lib/intel64/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../linux/release)  # relative to cwd


set(EXEC mem_test_host)
ADD_EXECUTABLE(${EXEC} ${SOURCES})
TARGET_LINK_LIBRARIES ( ${EXEC}
    ${LIBOMP}
    ${LIBTBB}
)
