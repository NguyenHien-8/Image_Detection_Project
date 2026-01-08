// ========================== Nguyen Hien ==========================
// FILE: src/layer2_detection.cpp (Use models YuNet)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer2_detection.h"
#include <iostream>

Layer2Detection::Layer2Detection() : isInitialized(false), currentInputSize(0, 0) {}

Layer2Detection::~Layer2Detection() {}

bool Layer2Detection::init(const std::string& modelPath, float scoreThreshold, float nmsThreshold) {
    try {
        model = cv::FaceDetectorYN::create(
            modelPath, "", cv::Size(320, 320), 
            scoreThreshold, nmsThreshold, 5000
        );
        
        if (model.empty()) {
            std::cerr << "[Layer2] ERROR: Failed to load YuNet model at " << modelPath << std::endl;
            return false;
        }

        isInitialized = true;
        std::cout << "[Layer2] INFO: YuNet initialized successfully." << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[Layer2] EXCEPTION: " << e.what() << std::endl;
        return false;
    }
}

bool Layer2Detection::detect(const cv::Mat& frame, FaceResult& result) {
    if (!isInitialized || model.empty() || frame.empty()) return false;
    if (frame.size() != currentInputSize) {
        model->setInputSize(frame.size());
        currentInputSize = frame.size();
    }

    cv::Mat faces;
    model->detect(frame, faces);
    if (faces.rows < 1) return false;

    float* data = faces.ptr<float>(0);  
    result.confidence = data[14];
    result.bbox = cv::Rect((int)data[0], (int)data[1], (int)data[2], (int)data[3]);
    result.bbox = result.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    result.landmarks.clear();
    
    // Các landmarks: Mắt phải, Mắt trái, Mũi, Miệng phải, Miệng trái
    // YuNet trả về float, giữ nguyên độ chính xác sub-pixel
    result.landmarks.push_back(cv::Point2f(data[4], data[5]));   
    result.landmarks.push_back(cv::Point2f(data[6], data[7]));   
    result.landmarks.push_back(cv::Point2f(data[8], data[9]));   
    result.landmarks.push_back(cv::Point2f(data[10], data[11])); 
    result.landmarks.push_back(cv::Point2f(data[12], data[13]));

    return true;
}