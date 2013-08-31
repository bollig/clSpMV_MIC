cmake_minimum_required(VERSION 2.6)
set(CMAKE_C_COMPILER "icc")                                                                                       
set(CMAKE_CXX_COMPILER "icpc")                                                                                       
#set(CMAKE_CXX_COMPILER /opt/intel/composer_xe_2013.5.192/bin/intel64/icpc)

project(spmv_openmp_mic)

set(D ../src_template/)
set(SOURCES 
    ${D}/spmv_ell_openmp.cpp 
    ${D}/util.cpp 
    ${D}/projectsettings.cpp 
    ${D}/timer_eb.cpp 
    ${D}/runs.cpp 
    ${D}/mmio.cpp 
    #${D}/rcm.cpp 
    #${D}/burkardt_rcm.cpp
)

include_directories( 
    ../include_template 
    ../ViennaCL-1.4.2
    #/usr/boost-1.45/include
) 

set(CMAKE_CXX_FLAGS "-std=c++0x -O2 -openmp -mmic")
set(CMAKE_EXE_LINKER_FLAGS "-mmic")

#add_definitions(-std=c++0x -O2 -openmp -mmic) 
#add_library(${PROJECT_NAME} STATIC ${SOURCES})

set(USRLIBDIR /usr/lib64)
FIND_LIBRARY(LIBOMP iomp5 PATH /opt/intel/composer_xe_2013/lib/mic/libiomp5.so)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../linux/release)  # relative to cwd


set(EXEC spmv_openmp_mic)
ADD_EXECUTABLE(${EXEC} ${SOURCES})
TARGET_LINK_LIBRARIES ( ${EXEC}
    ${LIBOMP}
)
