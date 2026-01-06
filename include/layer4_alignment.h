// ========================== Nguyen Hien ==========================
// FILE: include/layer4_alignment.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>

class Layer4Alignment {
public:
    cv::Mat align(const cv::Mat& frame,
                  const cv::Point2f& leftEye,
                  const cv::Point2f& rightEye);
};
