cmake_minimum_required(VERSION 2.6)
set(CMAKE_C_COMPILER "icc")                                                                                       
set(CMAKE_CXX_COMPILER "icpc")                                                                                       
#set(CMAKE_CXX_COMPILER /opt/intel/composer_xe_2013.5.192/bin/intel64/icpc)

project(spmv_openmp_mic)

set(D ../src_template/)
set(SOURCES 
    ${D}/bandwidth_tests.cpp
    ${D}/timer_eb.cpp
    ${D}/util.cpp
    ${D}/projectsettings.cpp
)

include_directories( 
    ../include_template 
    ../ViennaCL-1.4.2
    #/usr/boost-1.45/include
) 

set(CMAKE_CXX_FLAGS "-std=c++0x -O3 -openmp -mmic -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS "-mmic")

add_definitions(-std=c++0x -O3 -openmp -mmic) 
#add_library(${PROJECT_NAME} STATIC ${SOURCES})

set(USRLIBDIR /usr/lib64)
FIND_LIBRARY(LIBOMP iomp5 PATH
    /opt/intel/composer_xe_2013/lib/mic/libiomp5.so  # frodo
    /soft/intel/x86_64/2013/composer_xe_2013/composer_xe_2013.5.192/compiler/lib/mic #cascade
    /panfs/storage.local/opt/intel/new/lib/intel64/)  # hpc15-35

#FIND_LIBRARY(LIBTBB  tbb PATH
FIND_LIBRARY(LIBTBB  tbb_preview PATH
    /opt/intel/opencl-1.2-3.0.67279/libmic/
    /panfs/storage.local/opt/intel/new/tbb/lib/mic)
    #/panfs/storage.local/opt/intel/new/tbb/lib/intel64)  # hpc15-35

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../linux/release)  # relative to cwd


set(EXEC memory_tests)
ADD_EXECUTABLE(${EXEC} ${SOURCES})
TARGET_LINK_LIBRARIES ( ${EXEC}
    ${LIBOMP}
    ${LIBTBB}
)
