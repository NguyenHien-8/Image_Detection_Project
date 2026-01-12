// ========================== Nguyen Hien ==========================
// FILE: include/layer1_capture.h (AUTO-CONFIG MODE)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>
// =========================================================
// Full HD: 1920 | HD: 1280 | nHD: 960 | HD: 800 | nHD: 640
// Full HD: 1080 | HD: 720  | nHD: 540 | HD: 600 | nHD: 480
// =========================================================
class Layer1Capture {
public:
    Layer1Capture();
    ~Layer1Capture();

    bool init(int camID = 2, int captureWidth = 1280, int captureHeight = 720, 
              int displayWidth = 640, int displayHeight = 480);

    void release();
    bool grabFrame(cv::Mat& frame);
    void show(const cv::String& windowName, const cv::Mat& frame);
    int getMinFaceWidth() const;
    cv::Size getCaptureSize() const;

private:
    bool isInitialized;
    int captureWidth;
    int captureHeight;
    cv::VideoCapture cap;
    cv::Size displaySize;
    cv::Mat displayBuffer; 
};