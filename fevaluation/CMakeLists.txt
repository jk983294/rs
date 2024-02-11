CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(fevaluation)

include_directories(include)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/)
# set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

#set(FEVAL_BUILD_WITH_CUDA TRUE)
set(FEVAL_BUILD_WITH_CUDA FALSE)

if (FEVAL_BUILD_WITH_CUDA)
    message("FEVAL_BUILD_WITH_CUDA")
    ADD_DEFINITIONS(-DFEVAL_BUILD_WITH_CUDA)
    link_directories(/usr/local/cuda/lib64/)
    add_subdirectory(src/gpu_xx)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -s -Wall -Werror -Wno-format-truncation")
set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "-std=c++17 -g -fsanitize=address -Wall -Werror")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

ADD_DEFINITIONS(-DUNICODE -D_UNICODE)
ADD_DEFINITIONS(-DTYRANDE_LINUX)
ADD_DEFINITIONS(-DFMT_HEADER_ONLY)
ADD_DEFINITIONS(-DBOOST_STACKTRACE_LINK -DBOOST_STACKTRACE_USE_BACKTRACE)

FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(Boost 1.57 REQUIRED)
FIND_PACKAGE(PkgConfig REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("No OpenMP detected, fstlib builds without OpenMP but needs it for optimal performance!")
endif()

find_package(Arrow REQUIRED)

include_directories(~/github/cppfst/include/)
include_directories(~/github/cppfst/ext/lz4/)
include_directories(~/github/cppfst/ext/zstd/)
include_directories(SYSTEM /opt/3rd/Eigen)
include_directories(SYSTEM ~/github/concurrent/zergtool/include)
include_directories(SYSTEM ~/github/math/include)

link_directories(~/github/cppfst/cmake-build-debug/lib)

ADD_SUBDIRECTORY(src)

