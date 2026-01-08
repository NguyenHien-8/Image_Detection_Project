// ========================== Nguyen Hien ==========================
// FILE: include/layer1_capture.h
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#pragma once
#include <opencv2/opencv.hpp>

class Layer1Capture {
public:
    Layer1Capture();  
    ~Layer1Capture(); 

    bool init(int camID = 2, int width = 640, int height = 480);
    bool grabFrame(cv::Mat& frame);
    void release();
    static void convertToRGB(const cv::Mat& srcBgr, cv::Mat& dstRgb);
    void show(const cv::String& windowName, const cv::Mat& frame);

private:
    cv::VideoCapture cap;
    bool isInitialized;
};