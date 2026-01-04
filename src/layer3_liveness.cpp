// ============================================
// FILE: src/layer3_liveness.cpp
// ============================================
#include "layer3_liveness.h"
#include <iostream>
#include <cmath>
#include <algorithm>

LivenessDetector::LivenessDetector(int hist_size)
    : blink_counter(0), history_size(hist_size), 
      ear_threshold(0.2f), blink_sensitivity(0.15f) {}

LivenessDetector::~LivenessDetector() {}

void LivenessDetector::init() {
    blink_counter = 0;
    eye_aspect_ratio_history.clear();
    left_eye_history.clear();
    right_eye_history.clear();
    std::cout << "[INFO] Liveness detector initialized" << std::endl;
}

float LivenessDetector::calculateEyeAspectRatio(
    const cv::Point2f& left_eye, const cv::Point2f& right_eye) {
    
    // Calculate simple EAR (Eye Aspect Ratio)
    float horizontal_dist = cv::norm(right_eye - left_eye);
    
    // Approximate vertical distance (simplified)
    float vertical_dist = horizontal_dist * 0.4f;
    
    float ear = vertical_dist / (horizontal_dist + 1e-6f);
    return ear;
}

float LivenessDetector::calculateHeadMovement() {
    if (left_eye_history.size() < 5 || right_eye_history.size() < 5) {
        return 0.0f;
    }
    
    // Calculate movement between first and last frames
    float left_movement = cv::norm(
        left_eye_history.back() - left_eye_history.front());
    float right_movement = cv::norm(
        right_eye_history.back() - right_eye_history.front());
    
    float avg_movement = (left_movement + right_movement) / 2.0f;
    return avg_movement;
}

void LivenessDetector::detectBlink(float current_ear) {
    if (eye_aspect_ratio_history.empty()) {
        eye_aspect_ratio_history.push_back(current_ear);
        return;
    }
    
    float prev_ear = eye_aspect_ratio_history.back();
    eye_aspect_ratio_history.push_back(current_ear);
    
    // Keep history size
    if (eye_aspect_ratio_history.size() > history_size) {
        eye_aspect_ratio_history.pop_front();
    }
    
    // Detect blink: eye closes (EAR drops below threshold) then opens again
    bool eye_closing = (prev_ear > ear_threshold) && 
                       (current_ear < ear_threshold);
    
    if (eye_closing) {
        blink_counter++;
    }
}

LivenessInfo LivenessDetector::detect(const std::vector<cv::Point2f>& landmarks) {
    LivenessInfo result;
    result.blink_count = blink_counter;
    result.confidence = 0.0f;
    
    // Validate landmarks
    if (landmarks.size() < 6) {
        result.is_live = false;
        result.status_message = "Invalid landmarks";
        return result;
    }
    
    // Extract key landmarks
    cv::Point2f left_eye = landmarks[0];   // Left eye
    cv::Point2f right_eye = landmarks[1];  // Right eye
    
    // Add to history
    left_eye_history.push_back(left_eye);
    right_eye_history.push_back(right_eye);
    
    // Keep history size
    if (left_eye_history.size() > history_size) {
        left_eye_history.pop_front();
        right_eye_history.pop_front();
    }
    
    // Calculate Eye Aspect Ratio
    float ear = calculateEyeAspectRatio(left_eye, right_eye);
    detectBlink(ear);
    
    // Calculate head movement
    float head_move = calculateHeadMovement();
    result.head_movement = head_move;
    
    // Liveness criteria:
    // 1. Has blink (most reliable)
    // 2. Has head movement
    
    bool has_blink = (blink_counter >= 1);
    bool has_movement = (head_move > 8.0f); // Movement threshold in pixels
    
    result.is_live = has_blink || has_movement;
    
    // Calculate confidence
    if (has_blink) {
        result.confidence = 0.95f;
        result.status_message = "Live (blink detected)";
    } else if (has_movement) {
        result.confidence = 0.75f;
        result.status_message = "Live (movement detected)";
    } else {
        result.confidence = 0.30f;
        result.status_message = "Potentially fake (no liveness indicators)";
    }
    
    return result;
}

void LivenessDetector::reset() {
    blink_counter = 0;
    eye_aspect_ratio_history.clear();
    left_eye_history.clear();
    right_eye_history.clear();
}