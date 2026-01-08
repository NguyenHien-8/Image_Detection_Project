// ========================== Nguyen Hien ==========================
// FILE: include/layer4_hybrid.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// Hybrid Analysis: Frequency Domain & Image Quality Physics
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include "layer3_liveness.h"

class Layer4Hybrid {
public:
    Layer4Hybrid();
    ~Layer4Hybrid();

    // Hàm chính: Trả về True nếu vượt qua bài test kỹ sâu, False nếu fail
    bool verifyDeepAnalysis(const cv::Mat& frame, const cv::Rect& faceBox, float layer3Score);

private:
    // 1. Phân tích miền tần số (Fourier Transform)
    // Phát hiện mẫu lặp (Moire pattern) hoặc sự thiếu hụt tần số cao (Blurry screen)
    double analyzeFrequencyEnergy(const cv::Mat& src);

    // 2. Phân tích độ sắc nét cục bộ (Laplacian Variance)
    // Ảnh chụp lại thường bị mờ hơn ảnh gốc
    double analyzeBlur(const cv::Mat& src);
};