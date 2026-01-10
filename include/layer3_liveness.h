// =================================================================
// FILE: include/layer3_liveness.h (RGB COMPATIBLE - MODIFIED)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <deque>
#include <numeric>

enum class LivenessStatus {
    REAL,       // Real person detected (score > 0.80)
    SPOOF,      // Spoof/fake detected (score < 0.30)
    UNCERTAIN   // Uncertain - needs Layer 4 verification (0.30 - 0.80)
};

struct LivenessResult {
    LivenessStatus status; // Classification status
    float score;           // Raw score from MiniFASNet [0-1]
    std::string message;   // Status message for display
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
    
    // Temporal smoothing buffer (exponential weighted moving average)
    std::deque<float> scoreHistory;
    const size_t maxHistorySize = 8; // 8 frames for fast response
    float getSmoothedScore(float currentScore);
};