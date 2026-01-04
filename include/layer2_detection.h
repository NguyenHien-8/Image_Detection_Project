// ============================================
// FILE: include/layer2_detection.h
// ============================================
#ifndef LAYER2_DETECTION_H
#define LAYER2_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct Face {
    cv::Rect bbox;                      // Bounding box
    std::vector<cv::Point2f> landmarks; // 6 landmarks
    float confidence;
    bool detected;
};

class FaceDetector {
private:
    cv::dnn::Net net;
    bool is_loaded;
    float confidence_threshold;
    
    std::vector<cv::Point2f> extractLandmarks(const cv::Mat& face_roi, 
                                             const cv::Rect& bbox);

public:
    FaceDetector(float conf_threshold = 0.5f);
    ~FaceDetector();
    
    bool loadModel(const std::string& proto_path, 
                   const std::string& weights_path);
    Face detect(const cv::Mat& frame);
};

#endif // LAYER2_DETECTION_H