// ============================================
// FILE: include/layer5_packaging.h
// ============================================
#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class Layer5Packaging {
public:
    bool connect(const std::string& uri);
    void send(const cv::Mat& alignedFace);
};
