// =================================================================
// FILE: include/layer4_hybrid.h (RGB COMPATIBLE - MODIFIED)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// Hybrid Analysis: Frequency Domain & Image Quality Physics
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Layer4Hybrid {
public:
    Layer4Hybrid();
    ~Layer4Hybrid();

    float analyzeQuality(const cv::Mat& frame, const cv::Rect& faceBox);

private:

    double calculateNoiseMap(const cv::Mat& src); 
    double calculateHighFrequency(const cv::Mat& src);  
    bool checkSkinConsistency(const cv::Mat& src, float& outScore);
};