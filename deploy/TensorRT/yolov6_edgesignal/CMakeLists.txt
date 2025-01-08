cmake_minimum_required(VERSION 2.6)

project(yolov6)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
# add_definitions("-Wall -g")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/local/cuda/include
    /usr/include/x86_64-linux-gnu/
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu/
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-O2 -pthread)

# Source files
set(SOURCES
    main.cpp
    yolov6_class_edgesignal.cpp
)

# Create executable
add_executable(yolov6 ${SOURCES})

# Link libraries
target_link_libraries(yolov6
    nvinfer
    cudart
    ${OpenCV_LIBS}
)
