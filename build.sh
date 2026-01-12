#!/bin/bash
# ========================== Nguyen Hien ==========================
# FILE: build.sh (Auto Build Script)
# Developer: TRAN NGUYEN HIEN
# Email: trannguyenhien29085@gmail.com
# =================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required files
check_required_files() {
    print_info "Checking required files..."
    
    required_files=(
        "src/anti_spoof_decision.cpp"
        "include/anti_spoof_decision.h"
        "src/main.cpp"
        "src/layer1_capture.cpp"
        "src/layer2_detection.cpp"
        "src/layer3_liveness.cpp"
        "src/layer4_hybrid.cpp"
        "CMakeLists.txt"
    )
    
    missing_files=()
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        print_error "Missing required files:"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi
    
    print_info "All required files present ✓"
}

# Build function
build_project() {
    local build_type=$1
    local build_dir="build_${build_type,,}"
    
    print_info "Building ${build_type} version..."
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure
    print_info "Running CMake..."
    cmake -DCMAKE_BUILD_TYPE="$build_type" .. || {
        print_error "CMake configuration failed!"
        cd ..
        exit 1
    }
    
    # Build
    print_info "Compiling..."
    make -j$(nproc) || {
        print_error "Compilation failed!"
        cd ..
        exit 1
    }
    
    cd ..
    
    print_info "${build_type} build complete ✓"
    print_info "Executable: ${build_dir}/face_app"
}

# Clean function
clean_builds() {
    print_warn "Cleaning all build directories..."
    rm -rf build_debug build_release build
    print_info "Clean complete ✓"
}

# Test function
test_build() {
    local build_dir=$1
    
    if [ ! -f "${build_dir}/face_app" ]; then
        print_error "Executable not found in ${build_dir}!"
        return 1
    fi
    
    print_info "Testing executable..."
    
    # Check if models exist
    if [ ! -d "${build_dir}/models" ]; then
        print_error "Models directory not found!"
        return 1
    fi
    
    # Check dependencies
    print_info "Checking dependencies..."
    ldd "${build_dir}/face_app" | grep -q "opencv" || {
        print_error "OpenCV not linked properly!"
        return 1
    }
    
    print_info "Build test passed ✓"
}

# Main script
main() {
    echo "=========================================="
    echo "  Anti-Spoofing System Build Script v2.1"
    echo "=========================================="
    echo
    
    # Parse arguments
    case "${1:-release}" in
        debug|Debug|DEBUG)
            check_required_files
            build_project "Debug"
            test_build "build_debug"
            ;;
        release|Release|RELEASE)
            check_required_files
            build_project "Release"
            test_build "build_release"
            ;;
        both|all)
            check_required_files
            build_project "Debug"
            build_project "Release"
            test_build "build_debug"
            test_build "build_release"
            ;;
        clean)
            clean_builds
            ;;
        rebuild)
            clean_builds
            check_required_files
            build_project "Release"
            test_build "build_release"
            ;;
        *)
            echo "Usage: $0 [debug|release|both|clean|rebuild]"
            echo
            echo "Options:"
            echo "  debug    - Build debug version (with symbols)"
            echo "  release  - Build release version (optimized) [default]"
            echo "  both     - Build both versions"
            echo "  clean    - Remove all build directories"
            echo "  rebuild  - Clean and build release"
            echo
            exit 1
            ;;
    esac
    
    echo
    print_info "Build script completed successfully!"
}

# Run main
main "$@"