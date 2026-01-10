// ========================== Nguyen Hien ==========================
// FILE: src/layer3_liveness.cpp 
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
// FILE: src/layer3_liveness.cpp 
#include "layer3_liveness.h"
#include <iostream>

Layer3Liveness::Layer3Liveness() : isInitialized(false), inputSize(80, 80) {}
Layer3Liveness::~Layer3Liveness() {}

bool Layer3Liveness::init(const std::string& modelPath) {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        if (net.empty()) return false;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        isInitialized = true;
        std::cout << "[Layer3] INFO: MiniFASNet Optimized Init." << std::endl;
        return true;
    } catch (...) { return false; }
}

void Layer3Liveness::resetHistory() { scoreHistory.clear(); }

float Layer3Liveness::getSmoothedScore(float currentScore) {
    scoreHistory.push_back(currentScore);
    if (scoreHistory.size() > maxHistorySize) scoreHistory.pop_front();
    
    float sum = 0, weightSum = 0;
    for (size_t i = 0; i < scoreHistory.size(); ++i) {
        // Trọng số mũ: Frame mới nhất quan trọng gấp đôi frame cũ nhất
        float w = std::exp((float)i / scoreHistory.size()); 
        sum += scoreHistory[i] * w;
        weightSum += w;
    }
    return sum / weightSum;
}

bool Layer3Liveness::checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, LivenessResult& output) {
    if (!isInitialized || frame.empty()) return false;

    // --- TỐI ƯU CROP: Xử lý biên tốt hơn (Tránh viền đen) ---
    // Scale box rộng ra để lấy ngữ cảnh (scale 2.0 thay vì 1.5 để bắt được viền thiết bị nếu có)
    int size = std::max(faceBox.width, faceBox.height);
    int cx = faceBox.x + faceBox.width / 2;
    int cy = faceBox.y + faceBox.height / 2;
    int side = (int)(size * 1.8f); // Tăng scale lên 1.8 - 2.0 chuẩn model hơn

    // Tính toán vùng ROI an toàn
    int x1 = cx - side / 2;
    int y1 = cy - side / 2;
    int x2 = x1 + side;
    int y2 = y1 + side;

    // Xử lý padding nếu box vượt ra ngoài khung hình (Border Replicate)
    int top = 0, bottom = 0, left = 0, right = 0;
    if (x1 < 0) { left = -x1; x1 = 0; }
    if (y1 < 0) { top = -y1; y1 = 0; }
    if (x2 > frame.cols) { right = x2 - frame.cols; x2 = frame.cols; }
    if (y2 > frame.rows) { bottom = y2 - frame.rows; y2 = frame.rows; }

    cv::Mat croppedPart = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat finalInput;
    
    // Quan trọng: Dùng BORDER_REPLICATE thay vì để đen
    cv::copyMakeBorder(croppedPart, finalInput, top, bottom, left, right, cv::BORDER_REPLICATE);

    // Resize chuẩn model
    if (finalInput.size() != inputSize)
        cv::resize(finalInput, finalInput, inputSize);

    // Inference
    cv::Mat blob;
    // Chuẩn hóa: Mean=[0,0,0], Std=[1,1,1] (MiniFASNet thường không cần mean subtraction phức tạp nếu train từ scratch)
    cv::dnn::blobFromImage(finalInput, blob, 1.0, inputSize, cv::Scalar(0, 0, 0), true, false);
    
    net.setInput(blob);
    cv::Mat prob = net.forward();
    
    cv::Mat softmax;
    cv::exp(prob, softmax);
    float sumProb = (float)cv::sum(softmax)[0];
    float realScore = softmax.at<float>(0, 1) / sumProb;

    output.score = getSmoothedScore(realScore);
    
    // Logic threshold mềm dẻo hơn
    if (output.score > 0.80) output.status = LivenessStatus::REAL; // Hạ nhẹ threshold vì đã xử lý input sạch
    else if (output.score < 0.30) output.status = LivenessStatus::SPOOF;
    else output.status = LivenessStatus::UNCERTAIN;

    return true;
}