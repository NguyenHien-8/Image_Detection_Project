// ========================== Nguyen Hien ==========================
// FILE: include/layer3_liveness.h (IMPROVED)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <deque>

enum class LivenessStatus {
    REAL,
    SPOOF,
    UNCERTAIN
};

struct LivenessResult {
    float score;
    LivenessStatus status;
};

class Layer3Liveness {
public:
    Layer3Liveness();
    ~Layer3Liveness();

    bool init(const std::string& modelPath);
    bool checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, LivenessResult& output);
    void resetHistory();
    float getLastRawScore() const; // MỚI: Lấy raw score để debug

private:
    bool isInitialized;
    cv::dnn::Net net;
    cv::Size inputSize;
    cv::Mat borderBuffer;
    
    std::deque<float> scoreHistory;
    std::string outputName;
    const size_t maxHistorySize = 8;
    float previousScore = -1.0f;
    float lastRawScore = -1.0f; // MỚI: Lưu raw score
    int consecutiveLowCount = 0; // MỚI: Đếm frame liên tục thấp
    
    float getSmoothedScore(float currentScore);

    // MEMORY OPTIMIZATION: Member variables
    cv::Mat validCrop;
    cv::Mat finalInput;
    cv::Mat blob;
    cv::Mat prob;
    cv::Mat softmax;
};