#!/bin/bash

# ============================================
# FILE: build.sh
# Face Detection Pipeline Build Script
# ============================================

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# ============================================
# MAIN BUILD LOGIC
# ============================================

main() {
    print_header "Face Detection Pipeline - Build System"
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    if ! command_exists cmake; then
        print_error "CMake not found. Install with:"
        echo "  Ubuntu: sudo apt-get install cmake"
        echo "  macOS:  brew install cmake"
        exit 1
    fi
    print_success "CMake found"
    
    if ! command_exists pkg-config; then
        print_warning "pkg-config not found (optional)"
    else
        print_success "pkg-config found"
    fi
    
    # Check OpenCV
    if pkg-config --exists opencv4; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_success "OpenCV found (v${OPENCV_VERSION})"
    elif pkg-config --exists opencv; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        print_success "OpenCV found (v${OPENCV_VERSION})"
    else
        print_warning "OpenCV not found in pkg-config"
        print_info "CMake will attempt to find it automatically"
    fi
    
    # Create build directory
    print_info "Creating build directory..."
    if [ ! -d "build" ]; then
        mkdir -p build
        print_success "Directory created: build/"
    else
        print_info "Directory exists: build/"
    fi
    
    # Create models directory
    print_info "Creating models directory..."
    if [ ! -d "models" ]; then
        mkdir -p models
        print_success "Directory created: models/"
    fi
    
    # Download models if missing
    if [ ! -f "models/opencv_face_detector.pbtxt" ]; then
        print_info "Downloading face detector model..."
        cd models
        wget -q https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
        print_success "Downloaded: opencv_face_detector.pbtxt"
        cd ..
    else
        print_info "Model already exists: opencv_face_detector.pbtxt"
    fi
    
    if [ ! -f "models/opencv_face_detector_uint8.pb" ]; then
        print_info "Downloading face detector weights..."
        cd models
        wget -q https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb
        print_success "Downloaded: opencv_face_detector_uint8.pb"
        cd ..
    else
        print_info "Model already exists: opencv_face_detector_uint8.pb"
    fi
    
    # Configure CMake
    print_info "Configuring CMake..."
    cd build
    
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        print_info "Building for Windows..."
        cmake -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 16 2019" .. || {
            print_error "CMake configuration failed"
            cd ..
            exit 1
        }
    else
        # Linux / macOS
        print_info "Building for Unix-like system..."
        cmake -DCMAKE_BUILD_TYPE=Release .. || {
            print_error "CMake configuration failed"
            cd ..
            exit 1
        }
    fi
    
    print_success "CMake configured"
    
    # Build project
    print_info "Building project..."
    
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        cmake --build . --config Release -j 4 || {
            print_error "Build failed"
            cd ..
            exit 1
        }
    else
        make -j 4 || {
            print_error "Build failed"
            cd ..
            exit 1
        }
    fi
    
    print_success "Build completed"
    
    # Check if executable exists
    cd ..
    
    if [ -f "build/bin/face_recognition" ] || [ -f "build/bin/face_recognition.exe" ]; then
        print_success "Executable created successfully!"
        
        print_header "Build Complete"
        echo ""
        echo "To run the application:"
        echo ""
        echo "  1. Start WebSocket server (in terminal 1):"
        echo "     python3 websocket_server.py"
        echo ""
        echo "  2. Run face detection (in terminal 2):"
        echo "     ./build/bin/face_recognition ./models \\\"
        echo "       ws://localhost:8080/face device_001"
        echo ""
        echo "Controls:"
        echo "  q / ESC    - Quit"
        echo "  s          - Save face image"
        echo ""
    else
        print_error "Executable not found after build"
        exit 1
    fi
}

# Run main function
main "$@"