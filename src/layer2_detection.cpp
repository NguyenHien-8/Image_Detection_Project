// ========================== Nguyen Hien ==========================
// FILE: src/layer2_detection.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
// ========================== Nguyen Hien ==========================
// FILE: src/layer2_detection.cpp
// =================================================================
#include "layer2_detection.h"
#include <iostream>

static cv::Ptr<cv::FaceDetectorYN> model;
static cv::Size inputSize(0, 0);

Layer2Detection::Layer2Detection() : isInitialized(false) {}
Layer2Detection::~Layer2Detection() {}

bool Layer2Detection::init(const std::string& modelPath, float scoreThreshold, float nmsThreshold) {
    try {
        // Init YuNet với size chuẩn 320x320 (Model sẽ tự scale input)
        // Tuy nhiên để tối ưu, tí nữa ta setInputSize theo frame thực tế
        model = cv::FaceDetectorYN::create(
            modelPath, "", cv::Size(320, 320), 
            scoreThreshold, nmsThreshold, 5000
        );
        
        if (model.empty()) {
            std::cerr << "[ERROR] Failed to load YuNet." << std::endl;
            return false;
        }
        isInitialized = true;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return false;
    }
}

bool Layer2Detection::detect(const cv::Mat& frame, FaceResult& result) {
    if (!isInitialized || model.empty() || frame.empty()) return false;

    // Cập nhật size model khớp với frame input (640x480)
    if (frame.size() != inputSize) {
        model->setInputSize(frame.size());
        inputSize = frame.size();
    }

    cv::Mat faces;
    model->detect(frame, faces);

    if (faces.rows < 1) return false;

    // Lấy khuôn mặt đầu tiên
    float* data = faces.ptr<float>(0);
    
    // Không cần nhân Scale nữa vì frame input là frame gốc
    result.confidence = data[14];
    result.bbox = cv::Rect((int)data[0], (int)data[1], (int)data[2], (int)data[3]);

    result.landmarks.clear();
    
    // GIỮ NGUYÊN FIX MIRROR EFFECT (Đảo thứ tự mắt)
    // Index 0: Mắt Phải (Right Eye)
    result.landmarks.push_back(cv::Point2f(data[4], data[5]));   
    // Index 1: Mắt Trái (Left Eye)
    result.landmarks.push_back(cv::Point2f(data[6], data[7]));   
    // Index 2: Mũi
    result.landmarks.push_back(cv::Point2f(data[8], data[9]));   
    // Index 3: Khóe miệng Phải
    result.landmarks.push_back(cv::Point2f(data[10], data[11])); 
    // Index 4: Khóe miệng Trái
    result.landmarks.push_back(cv::Point2f(data[12], data[13])); 

    return true;
}