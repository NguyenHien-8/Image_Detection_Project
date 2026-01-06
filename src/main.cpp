// ========================== Nguyen Hien ==========================
// FILE: src/main.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include <opencv2/opencv.hpp>
#include <chrono>
#include "layer1_capture.h"
#include "layer2_detection.h"
#include "layer3_liveness.h"
#include "layer4_alignment.h"
#include "layer5_packaging.h"

void drawStatus(cv::Mat& img, cv::Rect box, std::string text, cv::Scalar color) {
    cv::rectangle(img, box, color, 2);
    cv::putText(img, text, cv::Point(box.x, box.y - 10), 
        cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

int main() {
    Layer1Capture capture;
    Layer2Detection detector;
    Layer3Liveness liveness;
    Layer4Alignment aligner;
    Layer5Packaging network;

    // [THAY ĐỔI] Chuyển về 640x480 để mượt (Real-time)
    // Nếu dùng Laptop cam thì để 0, Webcam rời thì để 1
    if (!capture.init(1, 640, 480)) { 
        std::cerr << "Init Camera Failed\n";
        return -1;
    }
    
    // Ngưỡng detect: 0.85 để lọc bớt nhiễu
    detector.init("models/face_detection_yunet_2023mar.onnx", 0.85f, 0.3f);

    cv::Mat bgr, rgb;
    FaceResult face;
    
    while (true) {
        if (!capture.grabFrame(bgr)) break;
        // Layer1Capture::convertToRGB(bgr, rgb); 
        
        // Detect trực tiếp trên frame gốc (nhanh gọn)
        bool hasFace = detector.detect(bgr, face);
        
        if (hasFace) {
            // Check Liveness
            bool isReal = liveness.checkLiveness(bgr, face.bbox, face.landmarks);

            if (isReal) {
                drawStatus(bgr, face.bbox, "REAL", cv::Scalar(0, 255, 0)); // Xanh
                
                // Căn chỉnh và gửi
                cv::Mat aligned = aligner.align(bgr, face.landmarks[1], face.landmarks[0]);
                if (!aligned.empty()) network.send(aligned);
            } 
            else {
                drawStatus(bgr, face.bbox, "FAKE", cv::Scalar(0, 0, 255)); // Đỏ
            }

            // Vẽ 5 điểm landmark
            for(auto& p : face.landmarks) cv::circle(bgr, p, 2, cv::Scalar(0,255,255), -1);
        } else {
            cv::putText(bgr, "Scanning...", cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);
        }

        capture.show("Face System (640x480)", bgr);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}