// ============================================
// FILE: src/layer2_detection.cpp
// ============================================
#include "layer2_detection.h"
#include <iostream>
#include <algorithm>
#include <cmath>

FaceDetector::FaceDetector(float conf_threshold)
    : is_loaded(false), confidence_threshold(conf_threshold) {}

FaceDetector::~FaceDetector() {}

bool FaceDetector::loadModel(const std::string& proto_path, 
                             const std::string& weights_path) {
    try {
        net = cv::dnn::readNetFromTensorflow(weights_path, proto_path);
        
        if (net.empty()) {
            std::cerr << "[ERROR] Failed to load model" << std::endl;
            return false;
        }
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        is_loaded = true;
        std::cout << "[INFO] Face detection model loaded" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Model loading failed: " << e.what() << std::endl;
        return false;
    }
}

Face FaceDetector::detect(const cv::Mat& frame) {
    Face result;
    result.detected = false;
    result.confidence = 0.0f;
    
    if (!is_loaded || frame.empty()) {
        return result;
    }
    
    int h = frame.rows;
    int w = frame.cols;
    
    // Create blob for neural network input
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, 
        cv::Size(300, 300), 
        cv::Scalar(104.0, 177.0, 123.0), 
        false, false);
    
    net.setInput(blob);
    cv::Mat detections = net.forward();
    
    // Resize detection output
    detections = detections.reshape(1, detections.total() / 7);
    
    // Parse detections
    for (int i = 0; i < detections.rows; i++) {
        float confidence = detections.at<float>(i, 2);
        
        if (confidence > confidence_threshold) {
            // Get bounding box coordinates
            int left = static_cast<int>(detections.at<float>(i, 3) * w);
            int top = static_cast<int>(detections.at<float>(i, 4) * h);
            int right = static_cast<int>(detections.at<float>(i, 5) * w);
            int bottom = static_cast<int>(detections.at<float>(i, 6) * h);
            
            // Clamp to frame boundaries
            left = std::max(0, left);
            top = std::max(0, top);
            right = std::min(w, right);
            bottom = std::min(h, bottom);
            
            result.bbox = cv::Rect(left, top, right - left, bottom - top);
            result.confidence = confidence;
            result.detected = true;
            
            // Extract landmarks from face region
            cv::Mat face_roi = frame(result.bbox);
            result.landmarks = extractLandmarks(face_roi, result.bbox);
            
            return result; // Return first best detection
        }
    }
    
    return result;
}

std::vector<cv::Point2f> FaceDetector::extractLandmarks(const cv::Mat& face_roi, 
                                                        const cv::Rect& bbox) {
    std::vector<cv::Point2f> landmarks;
    
    int roi_h = face_roi.rows;
    int roi_w = face_roi.cols;
    
    // Approximate 6 key landmarks positions in face ROI
    // These are normalized positions, you can improve with better models
    
    // Left eye
    cv::Point2f left_eye(roi_w * 0.35f, roi_h * 0.35f);
    
    // Right eye
    cv::Point2f right_eye(roi_w * 0.65f, roi_h * 0.35f);
    
    // Left ear
    cv::Point2f left_ear(roi_w * 0.1f, roi_h * 0.4f);
    
    // Right ear
    cv::Point2f right_ear(roi_w * 0.9f, roi_h * 0.4f);
    
    // Nose
    cv::Point2f nose(roi_w * 0.5f, roi_h * 0.5f);
    
    // Mouth
    cv::Point2f mouth(roi_w * 0.5f, roi_h * 0.75f);
    
    // Convert to absolute frame coordinates
    landmarks.push_back(left_eye + cv::Point2f(bbox.x, bbox.y));
    landmarks.push_back(right_eye + cv::Point2f(bbox.x, bbox.y));
    landmarks.push_back(left_ear + cv::Point2f(bbox.x, bbox.y));
    landmarks.push_back(right_ear + cv::Point2f(bbox.x, bbox.y));
    landmarks.push_back(nose + cv::Point2f(bbox.x, bbox.y));
    landmarks.push_back(mouth + cv::Point2f(bbox.x, bbox.y));
    
    return landmarks;
}