cmake_minimum_required(VERSION 3.0.2)
project(jointbilateralfilter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -g3")# -g3
find_package(OpenCV 3.4.11 REQUIRED)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)
find_package(CUDA 10.1 REQUIRED)

### directories where header files exists
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  cuda_toolkit/
)

cuda_add_executable(main
  main.cpp
  joint_bilateral_filter.cu
  OPTIONS -G
)
# OPTIONS -G -g


target_link_libraries(main
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)

# for debug
message(WARNING "prefix ${CMAKE_PREFIX_PATH}")    

message("-- CUDA --")
message(WARNING "version: " ${CUDA_VERSION})
message(WARNING "library: " ${CUDA_CUDA_LIBRARY})
message(WARNING "runtime library: " ${CUDA_CUDART_LIBRARY})
message(WARNING "include dirs: " ${CUDA_INCLUDE_DIRS})

message("-- OpenCV --")
message(WARNING "version ${OpenCV_VERSION}")
message(WARNING "install path ${OpenCV_INSTALL_PATH}") 
message(WARNING "config path ${OpenCV_CONFIG_PATH}") # look at the output of this message
message(WARNING "libs ${OpenCV_LIBS}")
message(WARNING "include dirs ${OpenCV_INCLUDE_DIRS}")


