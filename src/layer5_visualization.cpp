// ========================== Nguyen Hien ==========================
// FILE: src/layer5_visualization.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer5_visualization.h"
#include <iostream>

Layer5Visualization::Layer5Visualization() 
    : fontScale(1.0f), thickness(2), frameCount(0), startTime(std::chrono::steady_clock::now())
{
    colors.idle       = cv::Scalar(200, 200, 200);
    colors.verifying  = cv::Scalar(255, 255, 0);
    colors.real       = cv::Scalar(0, 255, 0);
    colors.spoof      = cv::Scalar(255, 0, 0);
    colors.warning    = cv::Scalar(255, 165, 0);
    colors.info       = cv::Scalar(200, 200, 200);
}

Layer5Visualization::~Layer5Visualization() {}

void Layer5Visualization::init(const cv::Size& dispSize) {
    displaySize = dispSize;
    fontScale = displaySize.width / 600.0f; 
    thickness = std::max(1, (int)(fontScale * 2));
    
    std::cout << "[Layer5] INFO: Visualizer ready. Target Display: " 
              << displaySize.width << "x" << displaySize.height << std::endl;
}

void Layer5Visualization::render(cv::Mat& displayFrame, const VisualizationData& data, float scaleX, float scaleY) {
    if (displayFrame.empty()) return;

    if (data.faceDetected) {
        cv::Rect scaledBox;
        scaledBox.x = (int)(data.faceBox.x * scaleX);
        scaledBox.y = (int)(data.faceBox.y * scaleY);
        scaledBox.width = (int)(data.faceBox.width * scaleX);
        scaledBox.height = (int)(data.faceBox.height * scaleY);

        drawFaceBox(displayFrame, scaledBox, data.status);
        drawStatusText(displayFrame, scaledBox, data.status);
        
        if (data.status != DisplayStatus::TOO_FAR) {
            drawScoreInfo(displayFrame, scaledBox, data.aiScore, data.physicalScore, data.finalScore);
            
            if (data.status == DisplayStatus::VERIFYING || data.status == DisplayStatus::REAL_PERSON) {
                drawProgressBar(displayFrame, scaledBox, data.consecutiveRealFrames, 10, data.status);
            }
        }
    } else {
        drawWarningMessage(displayFrame, "Please show your face");
    }
    
    updateFPS(const_cast<float&>(data.fps)); 
    drawFPSCounter(displayFrame, const_cast<float&>(data.fps));
}


void Layer5Visualization::show(const std::string& windowName, const cv::Mat& frame) {
    if (frame.empty()) return;
    
    cv::Mat bgrFrame;
    cv::cvtColor(frame, bgrFrame, cv::COLOR_RGB2BGR);
    cv::imshow(windowName, bgrFrame);
}