cmake_minimum_required(VERSION 3.17)
project(vol_pred)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/)


#SET(CMAKE_CXX_FLAGS "-std=c++17 -O3 -s -Wall -Werror")
SET(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror")

message ("cxx Flags: " ${CMAKE_CXX_FLAGS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

# Source code
include_directories(SYSTEM /opt/3rd/common/include)
include_directories(SYSTEM /opt/3rd/interface/ctp/include)
include_directories(SYSTEM third/printemps)
include_directories(/opt/version/latest/zergbase/include)
include_directories(/opt/version/latest/zergtool/include)
include_directories(SYSTEM /opt/version/latest/frenzy/include)
include_directories(/opt/version/latest/drone/include)
include_directories(/opt/version/latest/ornate/include)
include_directories(include)

link_directories(/opt/3rd/common/lib)
link_directories(/opt/version/latest/frenzy/lib)
link_directories(/opt/version/latest/lib)

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(src/common)
add_subdirectory(src/play)