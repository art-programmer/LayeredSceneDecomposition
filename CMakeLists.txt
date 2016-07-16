cmake_minimum_required(VERSION 2.6)
project (LayeredSceneDecomposition)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++0x -w")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(cv_utils)

#link_directories(cv_utils)
#} /opt/gurobi650/linux64/include)
file(GLOB SOURCES "*.cpp")
add_executable(LayeredSceneDecomposition ${SOURCES})
target_link_libraries(LayeredSceneDecomposition ${OpenCV_LIBS} ${GFlags_LIBS} gflags cv_utils)
