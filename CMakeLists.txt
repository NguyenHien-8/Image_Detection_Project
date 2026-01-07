cmake_minimum_required(VERSION 3.24.0)
project(ImageDetectionProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

set(SOURCES
    src/layer1_capture.cpp
    src/layer2_detection.cpp
    src/main.cpp 
)

add_executable(face_app ${SOURCES})
target_link_libraries(face_app PRIVATE ${OpenCV_LIBS})

add_custom_command(TARGET face_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/models
    ${CMAKE_CURRENT_BINARY_DIR}/models
    COMMENT "Copying models directory to binary directory..."
)