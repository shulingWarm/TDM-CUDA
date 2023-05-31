#cmake的最小版本
cmake_minimum_required(VERSION 3.2)
project(CUDATDM)
#寻找进程间通信用的库
find_package(ZeroMQ REQUIRED)
include_directories(${ZeroMQ_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
find_package(CUDA REQUIRED)
#寻找opencv库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
cuda_add_executable(directDOM directDOM.cu)
target_link_libraries(directDOM ${OpenCV_LIBS} ${ZeroMQ_LIBRARY})