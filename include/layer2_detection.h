// =================================================================
// FILE: include/layer2_detection.h 
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct FaceResult {
    cv::Rect bbox;                      
    float confidence;                
    std::vector<cv::Point2f> landmarks; 
};

class Layer2Detection {
public:
    Layer2Detection();
    ~Layer2Detection();

    bool init(const std::string& modelPath, 
              float scoreThreshold = 0.6f, 
              float nmsThreshold = 0.3f);

    bool detect(const cv::Mat& frame, FaceResult& result);

private:
    bool isInitialized;
    cv::Ptr<cv::FaceDetectorYN> model; 
    cv::Size currentInputSize; 
    
    cv::Mat facesResultBuffer;
};