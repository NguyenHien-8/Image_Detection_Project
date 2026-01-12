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
    // Giữ nguyên phần khởi tạo colors...
    colors.idle       = cv::Scalar(200, 200, 200);
    colors.verifying  = cv::Scalar(255, 255, 0);
    colors.real       = cv::Scalar(0, 255, 0);
    colors.spoof      = cv::Scalar(255, 0, 0);
    colors.warning    = cv::Scalar(255, 165, 0);
    colors.info       = cv::Scalar(200, 200, 200);
}

Layer5Visualization::~Layer5Visualization() {}

// Chỉ cần truyền kích thước hiển thị mong muốn
void Layer5Visualization::init(const cv::Size& dispSize) {
    displaySize = dispSize;
    
    // Tính toán font scale dựa trên màn hình hiển thị (nhỏ)
    // Giúp text to rõ, không bị bé tí như khi tính trên ảnh Full HD
    fontScale = displaySize.width / 600.0f; 
    thickness = std::max(1, (int)(fontScale * 2));
    
    std::cout << "[Layer5] INFO: Visualizer ready. Target Display: " 
              << displaySize.width << "x" << displaySize.height << std::endl;
}

// Logic vẽ giữ nguyên, nhưng sẽ được gọi trên frame nhỏ
void Layer5Visualization::render(cv::Mat& displayFrame, const VisualizationData& data, float scaleX, float scaleY) {
    if (displayFrame.empty()) return;

    if (data.faceDetected) {
        // QUAN TRỌNG: Scale bounding box từ tọa độ AI sang tọa độ hiển thị
        cv::Rect scaledBox;
        scaledBox.x = (int)(data.faceBox.x * scaleX);
        scaledBox.y = (int)(data.faceBox.y * scaleY);
        scaledBox.width = (int)(data.faceBox.width * scaleX);
        scaledBox.height = (int)(data.faceBox.height * scaleY);

        // Vẽ trên box đã scale
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
    
    updateFPS(const_cast<float&>(data.fps)); // Hack nhẹ để update FPS
    drawFPSCounter(displayFrame, const_cast<float&>(data.fps));
}

// Hàm show giờ đây chỉ đơn thuần hiển thị, không resize nữa (vì đã làm ở Main)
void Layer5Visualization::show(const std::string& windowName, const cv::Mat& frame) {
    if (frame.empty()) return;
    
    // OpenCV imshow dùng BGR
    cv::Mat bgrFrame;
    cv::cvtColor(frame, bgrFrame, cv::COLOR_RGB2BGR);
    cv::imshow(windowName, bgrFrame);
}

// Giữ nguyên các hàm draw helper (drawFaceBox, drawStatusText, v.v...)
// ... (Copy lại phần body của các hàm draw từ code cũ)