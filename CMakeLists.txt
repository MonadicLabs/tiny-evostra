project (TINYEVOSTRA)

## CHECK FOR COMPILER CAPABILITIES
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
##

include_directories(${TINYEVOSTRA_SOURCE_DIR}/include)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread" )

file (GLOB_RECURSE TINYEVOSTRA_SRC ${TINYEVOSTRA_SOURCE_DIR}/src/*.c*)
file (GLOB_RECURSE TINYEVOSTRA_HEADERS ${TINYEVOSTRA_SOURCE_DIR}/include/*.h)

# set( DEPS "${DEPS};${OpenCV_LIBS}" )

# Find Tiny DNN
include_directories( ${TINYEVOSTRA_SOURCE_DIR}/tiny-dnn )

add_executable(tiny-evostra ${TINYEVOSTRA_SRC} ${TINYEVOSTRA_HEADERS} )
target_link_libraries (tiny-evostra ${DEPS} )
