// ========================== Nguyen Hien ==========================
// FILE: include/layer2_detection.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct FaceResult {
    cv::Rect bbox;
    std::vector<cv::Point2f> landmarks;
    float confidence;
};

class Layer2Detection {
public:
    Layer2Detection();
    ~Layer2Detection();

    bool init(const std::string& modelPath = "models/face_detection_yunet_2023mar.onnx", 
              float scoreThreshold = 0.6f, 
              float nmsThreshold = 0.3f);

    bool detect(const cv::Mat& bgrFrame, FaceResult& result);

private:
    bool isInitialized;
    cv::Ptr<cv::FaceDetectorYN> model; 
    cv::Size currentInputSize; 
};