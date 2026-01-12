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

enum class DisplayStatus {
    IDLE,         
    TOO_FAR,        
    VERIFYING,     
    REAL_PERSON,  
    SPOOF_DETECTED 
};

struct VisualizationData {
    cv::Rect faceBox;
    bool faceDetected;
    float aiScore;        
    float physicalScore;   
    float finalScore;    
    
    DisplayStatus status;
    int consecutiveRealFrames;
    int consecutiveSpoofFrames;
    float fps;
    
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

class Layer5Visualization {
private:
    cv::Size captureSize;    
    cv::Size displaySize;  
    float fontScale;    
    int thickness;        
    
    int frameCount;
    std::chrono::steady_clock::time_point startTime;
    
    struct ColorScheme {
        cv::Scalar idle;       
        cv::Scalar verifying;  
        cv::Scalar real;       
        cv::Scalar spoof;      
        cv::Scalar warning;   
        cv::Scalar info;      
    } colors;

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
    
    void init(const cv::Size& captureSize, const cv::Size& displaySize);
    void updateFPS(float& outFPS); 
    void render(cv::Mat& frame, const VisualizationData& data);
    void show(const std::string& windowName, const cv::Mat& frame);
    void drawLandmarks(cv::Mat& frame, const std::vector<cv::Point2f>& landmarks);
    void drawDebugGrid(cv::Mat& frame); 
    float getFontScale() const { return fontScale; }
    int getThickness() const { return thickness; }
};
