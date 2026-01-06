// ========================== Nguyen Hien ==========================
// FILE: src/layer3_liveness.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer3_liveness.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Layer3Liveness::Layer3Liveness() {
    // Model 3D chuẩn
    modelPoints.push_back(cv::Point3f(-22.5f, -20.0f, 0.0f)); // R Eye
    modelPoints.push_back(cv::Point3f( 22.5f, -20.0f, 0.0f)); // L Eye
    modelPoints.push_back(cv::Point3f(  0.0f,   0.0f, 0.0f)); // Nose
    modelPoints.push_back(cv::Point3f(-15.0f,  20.0f, 0.0f)); // R Mouth
    modelPoints.push_back(cv::Point3f( 15.0f,  20.0f, 0.0f)); // L Mouth
}

bool Layer3Liveness::checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, const std::vector<cv::Point2f>& landmarks) {
    cv::Rect safeBox = faceBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.area() < 100) return false;
    
    cv::Mat faceROI = frame(safeBox);

    // 1. Check Texture (Quan trọng: Đã chỉnh lại threshold cho 480p)
    if (!analyzeTexture(faceROI)) return false;

    // 2. Check Color
    if (!analyzeColor(faceROI)) return false;

    // 3. Check Pose
    if (!analyzePose(landmarks, frame.size())) return false;

    return true;
}

bool Layer3Liveness::analyzePose(const std::vector<cv::Point2f>& lm, const cv::Size& frameSize) {
    if (lm.size() != 5) return false;

    double focalLength = frameSize.width;
    cv::Point2d center(frameSize.width / 2, frameSize.height / 2);
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength, 0, center.x, 0, focalLength, center.y, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);
    cv::Mat rvec, tvec;

    if (!cv::solvePnP(modelPoints, lm, cameraMatrix, distCoeffs, rvec, tvec)) return false;

    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);
    double yaw   = asin(rotMat.at<double>(0, 2)) * (180.0 / M_PI);
    double pitch = asin(rotMat.at<double>(1, 2)) * (180.0 / M_PI);

    // Góc chấp nhận được
    bool strictPitch = (pitch > -15.0 && pitch < 15.0); // Nới lỏng xíu cho dễ dùng
    bool strictYaw   = (yaw > -20.0 && yaw < 20.0);

    return strictPitch && strictYaw;
}

bool Layer3Liveness::analyzeTexture(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;

    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mu, sigma;
    cv::meanStdDev(laplacian, mu, sigma);
    
    double focusMeasure = sigma.val[0] * sigma.val[0]; 

    // === THRESHOLD CHO 640x480 ===
    // Ở độ phân giải thấp, ảnh sẽ ít chi tiết hơn.
    // 1. Blur Check: Nếu < 50.0 là quá mờ (ảnh in chất lượng thấp hoặc mất nét).
    // 2. Screen Check: Nếu > 1500.0 là quá sắc (dấu hiệu nhiễu pixel màn hình).
    
    if (focusMeasure < 50.0) { 
        // std::cout << "Blurry: " << focusMeasure << std::endl; 
        return false; 
    }
    
    if (focusMeasure > 1500.0) {
        // std::cout << "Screen Noise: " << focusMeasure << std::endl; 
        return false; 
    }

    return true;
}

bool Layer3Liveness::analyzeColor(const cv::Mat& src) {
    cv::Mat ycrcb;
    cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    cv::Scalar meanCr, stdCr;
    cv::meanStdDev(channels[1], meanCr, stdCr); 

    double crValue = meanCr.val[0];
    
    // Dải màu da phổ biến
    if (crValue < 133 || crValue > 173) return false; 

    return true;
}