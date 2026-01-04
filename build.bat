@echo off
REM ============================================
REM FILE: build.bat
REM Face Detection Pipeline - Conan Build Script
REM ============================================

setlocal enabledelayedexpansion

REM Color codes
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set BLUE=[94m
set NC=[0m

echo.
echo ================================================
echo  Face Detection Pipeline - Conan Build System
echo ================================================
echo.

REM Check if Conan is installed
echo [*] Checking Conan installation...
conan --version >nul 2>&1
if errorlevel 1 (
    echo [X] Conan not found!
    echo.
    echo Install Conan with:
    echo   pip install conan
    echo.
    exit /b 1
)
echo [OK] Conan found

REM Check if CMake is installed
echo [*] Checking CMake installation...
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [X] CMake not found!
    echo.
    echo Download from: https://cmake.org/download/
    echo.
    exit /b 1
)
echo [OK] CMake found

REM Check if Visual Studio is installed
echo [*] Checking Visual Studio...
where msbuild >nul 2>&1
if errorlevel 1 (
    echo [!] Visual Studio Build Tools not found
    echo Install from: https://visualstudio.microsoft.com/
)

REM Create models folder
echo [*] Creating models folder...
if not exist "models" mkdir models
echo [OK] Models folder created

REM Download models (optional, skip if exists)
if exist "models\opencv_face_detector.pbtxt" (
    echo [OK] Model files already exist
) else (
    echo [*] Downloading models...
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt' -OutFile 'models\opencv_face_detector.pbtxt'"
    powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb' -OutFile 'models\opencv_face_detector_uint8.pb'"
    echo [OK] Models downloaded
)

REM Create build directory
echo [*] Creating build directory...
if not exist "build" mkdir build
cd build

REM Run Conan
echo [*] Running Conan install...
conan install .. --build=missing --settings build_type=Release
if errorlevel 1 (
    echo [X] Conan install failed!
    cd ..
    exit /b 1
)
echo [OK] Conan dependencies installed

REM Configure CMake
echo [*] Configuring CMake...
cmake .. -DCMAKE_TOOLCHAIN_FILE="conan_toolchain.cmake" -DCMAKE_BUILD_TYPE=Release -G "Visual Studio 16 2019"
if errorlevel 1 (
    echo [X] CMake configuration failed!
    cd ..
    exit /b 1
)
echo [OK] CMake configured

REM Build project
echo [*] Building project...
cmake --build . --config Release -j 4
if errorlevel 1 (
    echo [X] Build failed!
    cd ..
    exit /b 1
)
echo [OK] Build successful

cd ..

REM Check if executable exists
if exist "build\bin\Release\face_recognition.exe" (
    echo.
    echo ================================================
    echo [OK] Build Complete!
    echo ================================================
    echo.
    echo To run the application:
    echo.
    echo   .\build\bin\Release\face_recognition.exe .\models
    echo.
    echo Controls:
    echo   q / ESC    - Quit
    echo   s          - Save face image
    echo.
) else (
    echo [X] Executable not found!
    exit /b 1
)

endlocal