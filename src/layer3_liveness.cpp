// ========================== Nguyen Hien ==========================
// FILE: src/layer3_liveness.cpp 
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer3_liveness.h"
#include <iostream>

Layer3Liveness::Layer3Liveness() : isInitialized(false), inputSize(80, 80) {}
Layer3Liveness::~Layer3Liveness() {}

bool Layer3Liveness::init(const std::string& modelPath) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) {
            std::cerr << "[Layer3]ERROR: Failed to load Liveness model at " << modelPath << std::endl;
            return false;
        }

        // Dùng CPU để đảm bảo tính tương thích, nếu có GPU thì đổi backend
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        isInitialized = true;
        std::cout << "[Layer3]INFO: Liveness Model initialized." << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[Layer3]EXCEPTION: " << e.what() << std::endl;
        return false;
    }
}

void Layer3Liveness::resetHistory() { scoreHistory.clear(); }

float Layer3Liveness::getSmoothedScore(float currentScore) {
    scoreHistory.push_back(currentScore);
    if (scoreHistory.size() > maxHistorySize) scoreHistory.pop_front();
    float sum = std::accumulate(scoreHistory.begin(), scoreHistory.end(), 0.0f);
    return sum / scoreHistory.size();
}

bool Layer3Liveness::checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, LivenessResult& output) {
    if (!isInitialized || frame.empty()) return false;

// 1. Prepare ROI (Scale 1.5x context)
    float scale = 1.5f; 
    int boxSize = std::max(faceBox.width, faceBox.height);
    int newSize = (int)(boxSize * scale);
    int cx = faceBox.x + faceBox.width / 2;
    int cy = faceBox.y + faceBox.height / 2;
    int x = cx - newSize / 2;
    int y = cy - newSize / 2;

    cv::Mat faceRoi = cv::Mat::zeros(newSize, newSize, frame.type());
    int srcX = std::max(0, x), srcY = std::max(0, y);
    int srcW = std::min(x + newSize, frame.cols) - srcX;
    int srcH = std::min(y + newSize, frame.rows) - srcY;

    if (srcW > 0 && srcH > 0) {
        frame(cv::Rect(srcX, srcY, srcW, srcH))
             .copyTo(faceRoi(cv::Rect(srcX - x, srcY - y, srcW, srcH)));
    }

    // 2. Inference MiniFASNet
    cv::Mat blob;
    cv::dnn::blobFromImage(faceRoi, blob, 1.0, inputSize, cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    cv::Mat prob = net.forward();
    
    cv::Mat softmaxProb;
    cv::exp(prob, softmaxProb);
    float sum = (float)cv::sum(softmaxProb)[0];
    float rawRealScore = softmaxProb.at<float>(0, 1) / sum; // Index 1 = Real

    // 3. Smoothing & Decision Making
    float finalScore = getSmoothedScore(rawRealScore);
    output.score = finalScore;

    // --- LOGIC PHÂN TẦNG (CASCADE) ---
    // Ngưỡng Cao: Chắc chắn thật
    if (finalScore > 0.85f) {
        output.status = LivenessStatus::REAL;
        output.message = "Real (Layer 3)";
    } 
    // Ngưỡng Thấp: Chắc chắn giả
    else if (finalScore < 0.40f) {
        output.status = LivenessStatus::SPOOF;
        output.message = "Spoof (Layer 3)";
    } 
    // Vùng Xám: Nghi ngờ -> Đẩy sang Layer 4
    else {
        output.status = LivenessStatus::UNCERTAIN;
        output.message = "Analyzing (Layer 4)...";
    }

    return true;
}