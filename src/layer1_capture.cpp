// ========================== Nguyen Hien ==========================
// FILE: src/layer1_capture.cpp (HIGH-RES MODE)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer1_capture.h"
#include <iostream>

Layer1Capture::Layer1Capture() : isInitialized(false), displaySize(640, 480) {}

Layer1Capture::~Layer1Capture() {
    release();
}

bool Layer1Capture::init(int camID, int captureWidth, int captureHeight, 
                         int displayWidth, int displayHeight) {
    if (isInitialized) release();

    displaySize = cv::Size(displayWidth, displayHeight);
    
    // Lưu lại cấu hình để dùng cho getter
    this->captureWidth = captureWidth;
    this->captureHeight = captureHeight;

    int maxRetries = 3;
    for(int i = 0; i < maxRetries; ++i) {
        cap.open(camID, cv::CAP_V4L2);
        if (!cap.isOpened()) cap.open(camID, cv::CAP_ANY);
        if (cap.isOpened()) break;
        std::cout << "[Layer1] WARN: Camera busy, retrying... (" << i+1 << ")" << std::endl;
        cv::waitKey(500);
    }

    if (!cap.isOpened()) return false;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, captureWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, captureHeight);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Đọc bỏ vài frame đầu để camera ổn định cân bằng trắng
    cv::Mat dummy;
    for(int i = 0; i < 10; i++) cap.read(dummy);

    isInitialized = true;
    std::cout << "[Layer1] INFO: Camera OK (" << captureWidth << "x" << captureHeight << ")" << std::endl;
    return true;
}

// FIX: Thêm các hàm bị thiếu mà main.cpp gọi
int Layer1Capture::getMinFaceWidth() const {
    // Logic: Khuôn mặt nhỏ quá 1/8 chiều rộng ảnh thì coi là xa/nhiễu
    return captureWidth / 8; 
}

cv::Size Layer1Capture::getCaptureSize() const {
    return cv::Size(captureWidth, captureHeight);
}

bool Layer1Capture::grabFrame(cv::Mat& frame) {
    if (!isInitialized || !cap.isOpened()) return false;
    
    // TỐI ƯU: Không convert sang RGB tại đây. Giữ BGR gốc của OpenCV để nhanh hơn.
    if (!cap.read(frame) || frame.empty()) {
        return false;
    }
    return true; 
}

void Layer1Capture::show(const cv::String& windowName, const cv::Mat& frame) {
    if (frame.empty()) return;
    
    // MEMORY OPTIMIZATION: Tái sử dụng displayBuffer
    // cv::resize thông minh: nếu displayBuffer đã đủ kích thước, nó không cấp phát lại bộ nhớ
    if (frame.size() == displaySize) {
        cv::imshow(windowName, frame);
    } else {
        cv::resize(frame, displayBuffer, displaySize);
        cv::imshow(windowName, displayBuffer);
    }
}

void Layer1Capture::release() {
    if (cap.isOpened()) cap.release();
    isInitialized = false;
}