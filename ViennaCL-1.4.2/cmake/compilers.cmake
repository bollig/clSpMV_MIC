IF (USE_GCC)
    set (CMAKE_C_COMPILER "gcc")
    set (CMAKE_CXX_COMPILER "g++")      
ENDIF (USE_GCC)

IF (USE_GCC46) 
    set (CMAKE_C_COMPILER "gcc-4.6")
    set (CMAKE_CXX_COMPILER "g++-4.6")      
ENDIF (USE_GCC46)

IF (USE_APPLE_GCC) 
    set (CMAKE_C_COMPILER "llvm-gcc")
    set (CMAKE_CXX_COMPILER "llvm-g++")      
ENDIF (USE_APPLE_GCC)



IF (USE_GCC47) 
    set (CMAKE_C_COMPILER "gcc-4.7")
    set (CMAKE_CXX_COMPILER "g++-4.7")      
ENDIF (USE_GCC47)


IF (USE_GCC44) 
    set (CMAKE_C_COMPILER "gcc-4.4")
    set (CMAKE_CXX_COMPILER "g++-4.4")      
ENDIF (USE_GCC44)


IF (USE_GCC44_KEENELAND) 
    set (CMAKE_C_COMPILER "gcc44")
    set (CMAKE_CXX_COMPILER "g++44")      
ENDIF (USE_GCC44_KEENELAND)


IF (USE_ICC)
    set (CMAKE_C_COMPILER "icc")
    set (CMAKE_CXX_COMPILER "icpc")
ENDIF (USE_ICC) 

message("C   compiler: ${CMAKE_C_COMPILER}")
message("C++ compiler: ${CMAKE_CXX_COMPILER}")
