// ========================== Nguyen Hien ==========================
// FILE: src/layer3_liveness.cpp (IMPROVED ACCURACY)
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
        if (net.empty()) return false;
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
        if (!outNames.empty()) {
            outputName = outNames[0];
        }

        isInitialized = true;
        return true;
    } catch (...) { return false; }
}

void Layer3Liveness::resetHistory() { 
    scoreHistory.clear();
    previousScore = -1.0f;
    consecutiveLowCount = 0;
}

float Layer3Liveness::getSmoothedScore(float currentScore) {
    if (currentScore < 0.25f) {
        scoreHistory.clear();
        scoreHistory.push_back(currentScore);
        previousScore = currentScore;
        consecutiveLowCount++;
        return currentScore;
    }
    
    if (previousScore > 0.70f && currentScore < 0.45f) {
        scoreHistory.clear();
        previousScore = currentScore;
        consecutiveLowCount = 0;
        return currentScore; 
    }
    
    if (currentScore < 0.40f) {
        consecutiveLowCount++;
        if (consecutiveLowCount >= 3) {
            scoreHistory.clear();
        }
    } else {
        consecutiveLowCount = 0;
    }
    
    scoreHistory.push_back(currentScore);
    if (scoreHistory.size() > maxHistorySize) scoreHistory.pop_front();
    
    float sum = 0, weightSum = 0;
    for (size_t i = 0; i < scoreHistory.size(); ++i) {
        float w = std::pow(1.8f, (float)i / scoreHistory.size()); 
        sum += scoreHistory[i] * w;
        weightSum += w;
    }
    
    float smoothed = sum / weightSum;
    if (currentScore < smoothed - 0.20f) {
        smoothed = currentScore * 0.7f + smoothed * 0.3f;
    }
    
    previousScore = currentScore;
    return smoothed;
}

bool Layer3Liveness::checkLiveness(const cv::Mat& frame, const cv::Rect& faceBox, LivenessResult& output) {
    if (!isInitialized || frame.empty()) return false;
    int cx = faceBox.x + faceBox.width / 2;
    int cy = faceBox.y + faceBox.height / 2;
    int maxSide = std::max(faceBox.width, faceBox.height);
    int side = (int)(maxSide * 1.8f);
    int desiredX = cx - side / 2;
    int desiredY = cy - side / 2;

    cv::Rect desiredRect(desiredX, desiredY, side, side);
    cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    cv::Rect validRect = desiredRect & frameRect;
    if (validRect.area() == 0) return false;

    validCrop = frame(validRect); 

    int top = validRect.y - desiredRect.y;
    int bottom = desiredRect.br().y - validRect.br().y;
    int left = validRect.x - desiredRect.x;
    int right = desiredRect.br().x - validRect.br().x;

    top = std::max(0, top); bottom = std::max(0, bottom);
    left = std::max(0, left); right = std::max(0, right);

    cv::copyMakeBorder(validCrop, borderBuffer, top, bottom, left, right, cv::BORDER_REPLICATE);
    cv::resize(borderBuffer, finalInput, inputSize);
    cv::dnn::blobFromImage(finalInput, blob, 1.0, inputSize, cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    
    if (!outputName.empty()) {
        net.forward(prob, outputName);
    } else {
        prob = net.forward();
    }
    
    cv::exp(prob, softmax); 
    float sumProb = (float)cv::sum(softmax)[0];
    float realScore = softmax.at<float>(0, 1) / sumProb;
    lastRawScore = realScore;
    output.score = getSmoothedScore(realScore);
    
    if (output.score > 0.85f) {
        output.status = LivenessStatus::REAL;
    } else if (output.score < 0.30f) {
        output.status = LivenessStatus::SPOOF;
    } else if (output.score > 0.72f) {
        output.status = LivenessStatus::REAL;
    } else if (output.score < 0.45f) {
        output.status = LivenessStatus::SPOOF;
    } else {
        output.status = LivenessStatus::UNCERTAIN;
    }

    return true;
}

float Layer3Liveness::getLastRawScore() const {
    return lastRawScore;
}