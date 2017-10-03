cmake_minimum_required(VERSION 3.8)
project(project1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp imProc.cpp)
add_executable(project1 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(project1 ${OpenCV_LIBS})