// ========================== Nguyen Hien ==========================
// FILE: include/layer3_liveness.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class Layer3Liveness {
public:
    Layer3Liveness();
    
    // Trả về true nếu là NGƯỜI THẬT + GÓC CHUẨN
    bool checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, const std::vector<cv::Point2f>& landmarks);

private:
    std::vector<cv::Point3f> modelPoints;
    
    // Các hàm kiểm tra con
    bool analyzePose(const std::vector<cv::Point2f>& lm, const cv::Size& frameSize);
    bool analyzeTexture(const cv::Mat& faceROI);
    bool analyzeColor(const cv::Mat& faceROI);
};
