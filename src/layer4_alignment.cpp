// ============================================
// FILE: src/layer4_alignment.cpp
// ============================================
#include "layer4_alignment.h"
#include <iostream>
#include <cmath>
#include <algorithm>

FaceAligner::FaceAligner(int size) : output_size(size) {
    std::cout << "[INFO] Face aligner initialized: " 
              << output_size << "x" << output_size << std::endl;
}

FaceAligner::~FaceAligner() {}

float FaceAligner::calculateRotationAngle(
    const cv::Point2f& left_eye, const cv::Point2f& right_eye) {
    
    // Calculate angle between two eyes
    float dy = right_eye.y - left_eye.y;
    float dx = right_eye.x - left_eye.x;
    
    float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
    
    // Normalize to [-90, 90]
    if (angle < -90.0f) angle += 180.0f;
    if (angle > 90.0f) angle -= 180.0f;
    
    return angle;
}

cv::Point2f FaceAligner::calculateFaceCenter(
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.empty()) {
        return cv::Point2f(0, 0);
    }
    
    // Calculate centroid of all landmarks
    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto& point : landmarks) {
        sum_x += point.x;
        sum_y += point.y;
    }
    
    return cv::Point2f(sum_x / landmarks.size(), 
                      sum_y / landmarks.size());
}

AlignedFace FaceAligner::align(const cv::Mat& frame, 
                               const std::vector<cv::Point2f>& landmarks) {
    AlignedFace result;
    result.success = false;
    
    // Validate input
    if (landmarks.size() < 6 || frame.empty()) {
        std::cerr << "[ERROR] Invalid input for alignment" << std::endl;
        return result;
    }
    
    // Extract eye landmarks (index 0 and 1)
    cv::Point2f left_eye = landmarks[0];
    cv::Point2f right_eye = landmarks[1];
    
    // Calculate rotation angle
    float angle = calculateRotationAngle(left_eye, right_eye);
    result.rotation_angle = angle;
    
    // Calculate face center
    cv::Point2f face_center = calculateFaceCenter(landmarks);
    result.face_center = face_center;
    
    // Create rotation matrix
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(
        face_center, angle, 1.0);
    
    // Apply rotation to frame
    cv::Mat rotated_frame;
    cv::warpAffine(frame, rotated_frame, rotation_matrix, 
                   frame.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    
    // Define ROI for face extraction
    int roi_x = std::max(0, (int)(face_center.x - output_size / 2));
    int roi_y = std::max(0, (int)(face_center.y - output_size / 2));
    int roi_w = output_size;
    int roi_h = output_size;
    
    // Clamp to frame boundaries
    if (roi_x + roi_w > rotated_frame.cols) {
        roi_x = rotated_frame.cols - roi_w;
    }
    if (roi_y + roi_h > rotated_frame.rows) {
        roi_y = rotated_frame.rows - roi_h;
    }
    
    roi_x = std::max(0, roi_x);
    roi_y = std::max(0, roi_y);
    
    if (roi_w <= 0 || roi_h <= 0 || 
        roi_x + roi_w > rotated_frame.cols || 
        roi_y + roi_h > rotated_frame.rows) {
        std::cerr << "[ERROR] Invalid ROI for face extraction" << std::endl;
        return result;
    }
    
    // Extract face ROI
    cv::Mat face_roi = rotated_frame(cv::Rect(roi_x, roi_y, roi_w, roi_h));
    
    // Resize to standard output size
    cv::resize(face_roi, result.face_image, 
               cv::Size(output_size, output_size), 0, 0, cv::INTER_LINEAR);
    
    // Normalize pixel values to [0, 1]
    if (result.face_image.type() == CV_8UC3) {
        result.face_image.convertTo(result.face_image, CV_32FC3, 1.0/255.0);
    } else if (result.face_image.type() == CV_8UC1) {
        result.face_image.convertTo(result.face_image, CV_32FC1, 1.0/255.0);
    }
    
    result.success = true;
    return result;
}