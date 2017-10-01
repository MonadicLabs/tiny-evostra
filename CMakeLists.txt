project (TINYEVOSTRA)

include_directories(${TINYEVOSTRA_SOURCE_DIR}/include)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread" )

file (GLOB_RECURSE TINYEVOSTRA_SRC ${TINYEVOSTRA_SOURCE_DIR}/src/*.c*)
file (GLOB_RECURSE TINYEVOSTRA_HEADERS ${TINYEVOSTRA_SOURCE_DIR}/include/*.h)

# set( DEPS "${DEPS};${OpenCV_LIBS}" )

# Find Tiny DNN
include_directories( ${TINYEVOSTRA_SOURCE_DIR}/tiny-dnn )

add_executable(tiny-evostra ${TINYEVOSTRA_SRC} ${TINYEVOSTRA_HEADERS} )
target_link_libraries (tiny-evostra ${DEPS} )
