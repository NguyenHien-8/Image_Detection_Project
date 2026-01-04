cmake_minimum_required(VERSION 3.15)
project(FaceDetectionPipeline VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================
# FIND PACKAGES FROM CONAN
# ============================================

message(STATUS "Finding packages from Conan...")

find_package(opencv REQUIRED)
if(opencv_FOUND)
    message(STATUS "✓ OpenCV found")
endif()

find_package(asio REQUIRED)
if(asio_FOUND)
    message(STATUS "✓ ASIO found")
endif()

find_package(websocketpp REQUIRED)
if(websocketpp_FOUND)
    message(STATUS "✓ WebSocket++ found")
endif()

find_package(Threads REQUIRED)

# ============================================
# INCLUDE DIRECTORIES
# ============================================

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# ============================================
# SOURCE FILES
# ============================================

set(SOURCES
    src/main.cpp
    src/layer1_capture.cpp
    src/layer2_detection.cpp
    src/layer3_liveness.cpp
    src/layer4_alignment.cpp
    src/layer5_packaging.cpp
)

set(HEADERS
    include/layer1_capture.h
    include/layer2_detection.h
    include/layer3_liveness.h
    include/layer4_alignment.h
    include/layer5_packaging.h
)

# ============================================
# CREATE EXECUTABLE
# ============================================

add_executable(face_recognition ${SOURCES} ${HEADERS})

# ============================================
# LINK LIBRARIES
# ============================================

target_link_libraries(face_recognition
    PRIVATE
        opencv::opencv_core
        opencv::opencv_imgproc
        opencv::opencv_highgui
        opencv::opencv_dnn
        opencv::opencv_videoio
        asio::asio
        websocketpp::websocketpp
        Threads::Threads
)

# ============================================
# COMPILER FLAGS
# ============================================

if(MSVC)
    # Visual Studio
    target_compile_options(face_recognition PRIVATE /W4 /EHsc)
    target_compile_definitions(face_recognition PRIVATE 
        _CRT_SECURE_NO_WARNINGS
        _WINSOCK_DEPRECATED_NO_WARNINGS
    )
else()
    # GCC / Clang
    target_compile_options(face_recognition PRIVATE 
        -Wall -Wextra -Wpedantic
    )
endif()

# ============================================
# BUILD INFORMATION
# ============================================

message(STATUS "")
message(STATUS "================================================")
message(STATUS "Face Detection Pipeline - Build Configuration")
message(STATUS "================================================")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenCV:         ✓ Found")
message(STATUS "  WebSocket++:    ✓ Found")
message(STATUS "  ASIO:           ✓ Found")
message(STATUS "  Threads:        ✓ Found")
message(STATUS "")
message(STATUS "Output Directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "================================================")
message(STATUS "")