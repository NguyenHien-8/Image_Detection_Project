// ========================== Nguyen Hien ==========================
// FILE: src/layer1_capture.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer1_capture.h"
#include <iostream>

Layer1Capture::Layer1Capture() : isInitialized(false) {}

Layer1Capture::~Layer1Capture() {
    release();
}

bool Layer1Capture::init(int camID, int width, int height) {
    if (isInitialized) {
        release(); 
    }

    cap.open(camID, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cout << "[WARN] V4L2 backend not found, trying CAP_ANY..." << std::endl;
        cap.open(camID, cv::CAP_ANY);
    }

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open camera " << camID << std::endl;
        return false;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_FPS, 30); // Try setting the FPS if your hardware supports it.

    cv::Mat dummy;
    for(int i = 0; i < 5; i++) {
        cap.read(dummy);
    }

    isInitialized = true;
    std::cout << "[INFO] Camera initialized successfully." << std::endl;
    return true;
}

bool Layer1Capture::grabFrame(cv::Mat& frame) {
    if (!isInitialized || !cap.isOpened()) return false;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "[ERROR] Lost frame capture." << std::endl;
        return false;
    }
    return true;
}

void Layer1Capture::release() {
    if (cap.isOpened()) {
        cap.release();
    }
    isInitialized = false;
}

void Layer1Capture::convertToRGB(const cv::Mat& srcBgr, cv::Mat& dstRgb) {
    if (srcBgr.empty()) return;
    cv::cvtColor(srcBgr, dstRgb, cv::COLOR_BGR2RGB);
}

void Layer1Capture::show(const cv::String& windowName, const cv::Mat& frame) {
    if (!frame.empty()) {
        cv::imshow(windowName, frame);
    }
}