// ========================== Nguyen Hien ==========================
// FILE: include/layer5_visualization.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// Description: UI/Display Management Layer - Tách biệt logic hiển thị
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

// ===== ENUMS CHO TRẠNG THÁI HIỂN THỊ =====
enum class DisplayStatus {
    IDLE,           // Không phát hiện khuôn mặt
    TOO_FAR,        // Khuôn mặt quá xa
    VERIFYING,      // Đang xác minh
    REAL_PERSON,    // Xác nhận người thật
    SPOOF_DETECTED  // Phát hiện giả mạo
};

// ===== CẤU TRÚC DỮ LIỆU HIỂN THỊ =====
struct VisualizationData {
    // Thông tin khuôn mặt
    cv::Rect faceBox;
    bool faceDetected;
    
    // Điểm số AI
    float aiScore;          // Score từ Deep Learning
    float physicalScore;    // Score từ Heuristic
    float finalScore;       // Score cuối cùng sau fusion
    
    // Trạng thái
    DisplayStatus status;
    int consecutiveRealFrames;
    int consecutiveSpoofFrames;
    
    // FPS
    float fps;
    
    // Constructor mặc định
    VisualizationData() : 
        faceBox(0, 0, 0, 0),
        faceDetected(false),
        aiScore(0.0f),
        physicalScore(0.0f),
        finalScore(0.0f),
        status(DisplayStatus::IDLE),
        consecutiveRealFrames(0),
        consecutiveSpoofFrames(0),
        fps(0.0f) {}
};

// ===== LAYER 5: VISUALIZATION =====
class Layer5Visualization {
private:
    // Cấu hình hiển thị
    cv::Size captureSize;    // Kích thước ảnh gốc từ camera
    cv::Size displaySize;    // Kích thước hiển thị
    float fontScale;         // Tự động scale font theo resolution
    int thickness;           // Độ dày nét vẽ
    
    // FPS tracking
    int frameCount;
    std::chrono::steady_clock::time_point startTime;
    
    // Màu sắc theme
    struct ColorScheme {
        cv::Scalar idle;        // Màu khi không có mặt
        cv::Scalar verifying;   // Màu khi đang xác minh
        cv::Scalar real;        // Màu khi xác nhận thật
        cv::Scalar spoof;       // Màu khi phát hiện giả
        cv::Scalar warning;     // Màu cảnh báo
        cv::Scalar info;        // Màu thông tin
    } colors;
    
    // Helper functions
    void drawFaceBox(cv::Mat& frame, const cv::Rect& box, DisplayStatus status);
    void drawStatusText(cv::Mat& frame, const cv::Rect& box, DisplayStatus status);
    void drawScoreInfo(cv::Mat& frame, const cv::Rect& box, 
                      float aiScore, float physicalScore, float finalScore);
    void drawFPSCounter(cv::Mat& frame, float fps);
    void drawProgressBar(cv::Mat& frame, const cv::Rect& box, 
                        int consecutive, int threshold, DisplayStatus status);
    void drawWarningMessage(cv::Mat& frame, const std::string& message);
    
public:
    Layer5Visualization();
    ~Layer5Visualization();
    
    // Khởi tạo với thông tin resolution
    void init(const cv::Size& captureSize, const cv::Size& displaySize);
    
    // Cập nhật FPS
    void updateFPS(float& outFPS);
    
    // Vẽ toàn bộ UI lên frame
    void render(cv::Mat& frame, const VisualizationData& data);
    
    // Hiển thị frame (resize nếu cần)
    void show(const std::string& windowName, const cv::Mat& frame);
    
    // Các hàm tiện ích
    void drawLandmarks(cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    void drawDebugGrid(cv::Mat& frame); // Vẽ grid debug nếu cần
    
    // Getters
    float getFontScale() const { return fontScale; }
    int getThickness() const { return thickness; }
};
