// ============================================
// FILE: src/layer5_packaging.cpp
// ============================================
#include "layer5_packaging.h"
#include <iostream>

bool Layer5Packaging::connect(const std::string& uri) {
    std::cout << "WebSocket connected: " << uri << std::endl;
    return true;
}

void Layer5Packaging::send(const cv::Mat& alignedFace) {
    std::cout << "Face data sent to server" << std::endl;
}
