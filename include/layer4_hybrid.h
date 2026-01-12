// ========================== Nguyen Hien ==========================
// FILE: include/layer4_hybrid.h (IMPROVED)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
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
    // --- Core Logic Methods ---
    float analyzeTextureGradient(const cv::Mat& src);
    float detectMoirePattern(const cv::Mat& src);
    double calculateHighFrequency(const cv::Mat& src);
    bool checkSkinConsistency(const cv::Mat& src, float& outScore);
    float analyzeColorTemperature(const cv::Mat& src);
    float detectScreenEdges(const cv::Mat& src); // MỚI: Phát hiện viền màn hình

    // --- MEMORY OPTIMIZATION: Persistent Buffers ---
    
    // 1. Buffers cho Texture Gradient & High Freq
    cv::Mat grayBuffer, gradX, gradY, magnitude;
    cv::Mat resizedBuffer, padded, complexI, magI;
    cv::Mat mask;
    cv::Mat dftPlanes[2];

    // 2. Buffers cho Moire Pattern
    cv::Mat moireResized, moireGray, moireLaplacian;

    // 3. Buffers cho Skin Consistency
    cv::Mat skinSmall, skinYCrCb, skinHSV;
    cv::Mat plane0, plane1;

    // 4. Buffers cho Color Temp
    cv::Mat tempSmall;
    
    // 5. MỚI: Buffers cho Screen Edge Detection
    cv::Mat edgeBuffer, edgeMap;
};