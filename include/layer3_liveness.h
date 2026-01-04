// ============================================
// FILE: include/layer3_liveness.h
// ============================================
#ifndef LAYER3_LIVENESS_H
#define LAYER3_LIVENESS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

struct LivenessInfo {
    bool is_live;
    int blink_count;
    float head_movement;
    float confidence;
    std::string status_message;
};

class LivenessDetector {
private:
    std::deque<float> eye_aspect_ratio_history;
    std::deque<cv::Point2f> left_eye_history;
    std::deque<cv::Point2f> right_eye_history;
    
    int blink_counter;
    int history_size;
    float ear_threshold;
    float blink_sensitivity;
    
    float calculateEyeAspectRatio(const cv::Point2f& left_eye, 
                                  const cv::Point2f& right_eye);
    float calculateHeadMovement();
    void detectBlink(float current_ear);

public:
    LivenessDetector(int hist_size = 15);
    ~LivenessDetector();
    
    void init();
    LivenessInfo detect(const std::vector<cv::Point2f>& landmarks);
    void reset();
};

#endif // LAYER3_LIVENESS_H