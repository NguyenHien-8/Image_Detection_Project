# Image_Detection_Project

```
Video Stream (Camera)
        ↓
[Layer 1] Video Capture
        ↓
[Layer 2] Face Detection
        ↓
[Layer 3] Liveness Detection
        ↓
[Layer 4] Face Alignment
        ↓
[Layer 5] WebSocket Transmission
        ↓
    Server/Database
```

### Step 1: Clone/Download Project

```bash
cd ~
mkdir -p projects
cd projects
git clone <repository-url> Image_Detection_Project
cd Image_Detection_Project
```

### Step 2: Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    libssl-dev

# Install WebSocket++ (optional, can use system package)
git clone https://github.com/zaphoyd/websocketpp.git
sudo cp -r websocketpp/websocketpp /usr/local/include/

# Install ASIO
git clone https://github.com/chriskohlhoff/asio.git
sudo cp -r asio/asio/include/asio* /usr/local/include/
```

**macOS:**
```bash
brew install cmake opencv openssl
brew install websocketpp asio

# May need to manually install WebSocket++
git clone https://github.com/zaphoyd/websocketpp.git
cp -r websocketpp/websocketpp /usr/local/include/
```

### Step 3: Build Project

```bash
# Make build script executable
chmod +x build.sh

# Automated build (downloads models + builds)
./build.sh
```

### Step 4: Run Application

**Terminal 1 - Start Server:**
```bash
# Python WebSocket server (optional, for data reception)
pip3 install websockets
python3 websocket_server.py localhost 8080
```

**Terminal 2 - Run Pipeline:**
```bash
./build/bin/face_recognition ./models ws://localhost:8080/face device_001
```

## 📁 Project Structure

```
Image_Detection_Project/
├── include/                      # Header files
│   ├── layer1_capture.h         # Video capture
│   ├── layer2_detection.h       # Face detection
│   ├── layer3_liveness.h        # Liveness detection
│   ├── layer4_alignment.h       # Face alignment
│   └── layer5_packaging.h       # WebSocket client
├── src/                         # Source files
│   ├── main.cpp                 # Main application
│   ├── layer*.cpp               # Layer implementations
│   └── (5 files total)
├── models/                      # ML models (auto-downloaded)
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── build/                       # Build output (auto-created)
│   └── bin/
│       └── face_recognition    # Executable
├── CMakeLists.txt              # Build configuration
└── build.sh                    # Build script
```

## 🔧 Configuration

Edit parameters in `src/main.cpp`:

```cpp
// Line in main():
std::string proto_path = "./models/opencv_face_detector.pbtxt";
std::string weights_path = "./models/opencv_face_detector_uint8.pb";
std::string server_uri = "ws://localhost:8080/face";
std::string device_id = "device_001";
```

## 💻 Usage

### Basic Usage
```bash
./build/bin/face_recognition ./models ws://localhost:8080/face device_001
```

### With Custom Server
```bash
./build/bin/face_recognition \
    ./models \
    ws://192.168.1.100:8080/face \
    my_device_002
```

### Keyboard Controls
- **q** or **ESC** - Quit application
- **s** - Save detected face image
- Any other key - Continue processing

## 🏗️ Layer Architecture Details

### Layer 1: Video Capture
- Opens camera device
- Reads frames at 30 FPS
- Resizes to 640x480
- **Input**: Camera stream
- **Output**: cv::Mat frame

### Layer 2: Face Detection
- Uses OpenCV DNN module
- Loads pre-trained model
- Detects faces with confidence score
- Extracts 6 facial landmarks
- **Input**: cv::Mat frame
- **Output**: Face struct (bbox, landmarks, confidence)

### Layer 3: Liveness Detection
- Calculates Eye Aspect Ratio (EAR)
- Detects blink patterns
- Tracks head movement
- Distinguishes live vs fake faces
- **Input**: Face landmarks
- **Output**: LivenessInfo struct

### Layer 4: Face Alignment
- Calculates rotation angle from eyes
- Applies affine transformation
- Crops face to 112x112 pixels
- Normalizes pixel values
- **Input**: Face frame + landmarks
- **Output**: AlignedFace struct (112x112 image)

### Layer 5: Data Packaging
- Encodes aligned face to JPEG
- Converts to Base64
- Creates JSON payload
- Sends via WebSocket
- **Input**: AlignedFace + metadata
- **Output**: JSON data to server

## 🔗 WebSocket Message Format

### Client → Server
```json
{
    "type": "face_data",
    "device_id": "device_001",
    "timestamp": 1704294000000,
    "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "metadata": {
        "blink_count": 2,
        "liveness_confidence": 0.95,
        "rotation_angle": 5.2
    }
}
```

### Server → Client
```json
{
    "status": "success",
    "face_id": 42,
    "message": "Face #42 received"
}
```

## 🐛 Troubleshooting

### Camera Not Detected
```bash
# Linux: Check available devices
ls /dev/video*

# Add user to video group
sudo usermod -a -G video $USER
newgrp video
```

### OpenCV Library Error
```bash
# Reinstall OpenCV
sudo apt-get remove libopencv-dev
sudo apt-get install libopencv-dev

# Or build from source
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make -j 4
sudo make install
```

### WebSocket Connection Failed
```bash
# Check if server is running
netstat -an | grep 8080

# Verify firewall rules
sudo ufw allow 8080/tcp

# Start server with debugging
python3 websocket_server.py localhost 8080 --debug
```

### High CPU Usage
- Increase `frame_skip` in main.cpp
- Reduce camera resolution
- Disable WebSocket transmission temporarily

## 📈 Performance Optimization

### For Faster Processing:
1. **Skip frames**: Process every 2nd or 3rd frame
2. **Lower resolution**: Use 320x240 instead of 640x480
3. **Reduce model confidence**: Set to 0.7 instead of 0.5

### For Better Accuracy:
1. **Use MediaPipe**: Better facial landmarks
2. **Improve lighting**: Ensure good camera lighting
3. **Increase confidence threshold**: Only detect confident faces

## 🔐 Security Considerations

- **HTTPS/WSS**: Use WebSocket Secure (WSS) in production
- **Authentication**: Add token-based auth to API
- **Data Encryption**: Encrypt face images before transmission
- **Privacy**: Implement data retention policies
- **Compliance**: Follow GDPR/privacy regulations

## 📚 API Reference

### VideoCapture Class
```cpp
class VideoCapture {
    bool open();
    bool getFrame(cv::Mat& frame);
    void close();
};
```

## 📞 Support & Resources

- **Documentation**: See included `.md` files
- **GitHub Issues**: Report bugs
- **OpenCV Docs**: https://docs.opencv.org/
- **WebSocket++**: https://www.zaphoyd.com/websocketpp/
