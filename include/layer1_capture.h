// ============================================
// FILE: include/layer1_capture.h
// ============================================
#ifndef LAYER1_CAPTURE_H
#define LAYER1_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <string>

class VideoCapture {
private:
    cv::VideoCapture camera;
    int width;
    int height;
    int fps;
    bool is_open;

public:
    VideoCapture(int camera_id = 0, int w = 640, int h = 480, int f = 30);
    ~VideoCapture();
    
    bool open();
    bool getFrame(cv::Mat& frame);
    void close();
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    bool isOpened() const { return is_open; }
};

#endif // LAYER1_CAPTURE_H