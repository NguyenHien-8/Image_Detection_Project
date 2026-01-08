// ========================== Nguyen Hien ==========================
// FILE: include/layer3_liveness.h 
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <deque>
#include <numeric>

enum class LivenessStatus {
    REAL,       // Chắc chắn là thật (> 0.85)
    SPOOF,      // Chắc chắn là giả (< 0.40)
    UNCERTAIN   // Nghi ngờ (0.40 - 0.85) -> Cần Layer 4 check
};

struct LivenessResult {
    LivenessStatus status; // Trạng thái xử lý
    float score;           // Điểm số raw từ MiniFASNet
    std::string message;   // Message hiển thị
};

class Layer3Liveness {
public:
    Layer3Liveness();
    ~Layer3Liveness();

    bool init(const std::string& modelPath = "models/MiniFASNetV1SE.onnx");
    bool checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, LivenessResult& output);
    void resetHistory();

private:
    bool isInitialized;
    cv::dnn::Net net;
    cv::Size inputSize; 
    
    // Smoothing buffer
    std::deque<float> scoreHistory;
    const size_t maxHistorySize = 8; // Giảm buffer xuống 8 để phản ứng nhanh hơn
    float getSmoothedScore(float currentScore);
};