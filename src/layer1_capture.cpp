// ============================================
// FILE: src/layer1_capture.cpp
// ============================================
#include "layer1_capture.h"
#include <iostream>

VideoCapture::VideoCapture(int camera_id, int w, int h, int f)
    : width(w), height(h), fps(f), is_open(false) {
    camera.open(camera_id);
}

VideoCapture::~VideoCapture() {
    close();
}

bool VideoCapture::open() {
    if (!camera.isOpened()) {
        std::cerr << "[ERROR] Cannot open camera" << std::endl;
        return false;
    }
    
    // Set camera properties
    camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    camera.set(cv::CAP_PROP_FPS, fps);
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    is_open = true;
    std::cout << "[INFO] Camera opened: " << width << "x" << height 
              << " @ " << fps << "fps" << std::endl;
    return true;
}

bool VideoCapture::getFrame(cv::Mat& frame) {
    if (!is_open) {
        std::cerr << "[ERROR] Camera is not open" << std::endl;
        return false;
    }
    
    cv::Mat raw_frame;
    if (!camera.read(raw_frame)) {
        std::cerr << "[ERROR] Failed to read frame" << std::endl;
        return false;
    }
    
    // Resize frame if needed
    if (raw_frame.cols != width || raw_frame.rows != height) {
        cv::resize(raw_frame, frame, cv::Size(width, height));
    } else {
        frame = raw_frame.clone();
    }
    
    return true;
}

void VideoCapture::close() {
    if (camera.isOpened()) {
        camera.release();
        is_open = false;
        std::cout << "[INFO] Camera closed" << std::endl;
    }
}