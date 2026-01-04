// ============================================
// FILE: include/layer4_alignment.h
// ============================================
#ifndef LAYER4_ALIGNMENT_H
#define LAYER4_ALIGNMENT_H

#include <opencv2/opencv.hpp>
#include <vector>

struct AlignedFace {
    cv::Mat face_image;      // Aligned 112x112 face
    float rotation_angle;
    cv::Point2f face_center;
    bool success;
};

class FaceAligner {
private:
    int output_size;  // 112 or 160
    
    float calculateRotationAngle(const cv::Point2f& left_eye, 
                                const cv::Point2f& right_eye);
    cv::Point2f calculateFaceCenter(const std::vector<cv::Point2f>& landmarks);

public:
    FaceAligner(int size = 112);
    ~FaceAligner();
    
    AlignedFace align(const cv::Mat& frame, 
                     const std::vector<cv::Point2f>& landmarks);
};

#endif // LAYER4_ALIGNMENT_H