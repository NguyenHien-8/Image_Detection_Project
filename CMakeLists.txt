cmake_minimum_required(VERSION 3.15)
project(ImageDetectionProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

file(GLOB SOURCES src/*.cpp)

add_executable(face_app ${SOURCES})

target_link_libraries(face_app PRIVATE opencv::opencv)

add_custom_command(TARGET face_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_BINARY_DIR}/models)
