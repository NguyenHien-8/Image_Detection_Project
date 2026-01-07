// ========================== Nguyen Hien ==========================
// FILE: src/layer1_capture.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include <iostream>
#include <string>
#include <chrono> 
#include <exception> 

#include "layer1_capture.h"
#include "layer2_detection.h"

int main(int argc, char** argv) {
    std::cout << "=== FACE DETECTION SYSTEM STARTED ===" << std::endl;
    Layer1Capture camera;
    Layer2Detection detector;
    
    try {
        // ===== Init =====
        if (!camera.init(0, 640, 480)) { 
            throw std::runtime_error("Failed to init camera! Please check connection.");
        }

        std::string modelPath = "models/face_detection_yunet_2023mar.onnx"; 
        if (!detector.init(modelPath, 0.6f, 0.3f)) {
            throw std::runtime_error("Failed to init face detector! Check path: " + modelPath);
        }

        cv::Mat frameBgr;
        FaceResult faceResult;
        
        // ===== FPS Calculation =====
        int frameCount = 0;
        auto startTime = std::chrono::steady_clock::now();
        float fps = 0.0f;

        std::cout << "[INFO] System running. Press 'e' or 'ESC' to exit." << std::endl;

        // ===== Main loop =====
        while (true) {
            // B1: Lấy ảnh từ Layer 1
            if (!camera.grabFrame(frameBgr)) {
                // Ta có thể break để thoát, hoặc thêm logic reconnect tại đây.
                std::cerr << "[WARN] Frame capture failed (Camera disconnected?). Exiting loop..." << std::endl;
                break;
            }
            
            // B3: Phát hiện khuôn mặt bằng Layer 2
            bool found = detector.detect(frameBgr, faceResult);

            // B4: Vẽ kết quả lên frame
            if (found) {
                // ---- Draw bounding box(Green) ---- 
                cv::rectangle(frameBgr, faceResult.bbox, cv::Scalar(0, 255, 0), 2);

                // ---- Draw landmarks(Red) ----
                for (const auto& point : faceResult.landmarks) {
                    cv::circle(frameBgr, point, 3, cv::Scalar(0, 0, 255), -1);
                }

                // Hiển thị độ tin cậy
                std::string label = cv::format("Conf: %.2f", faceResult.confidence);
                cv::putText(frameBgr, label, cv::Point(faceResult.bbox.x, faceResult.bbox.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // Tính toán và hiển thị FPS
            frameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
            
            if (elapsed >= 1000) { // Mỗi 1 giây cập nhật FPS
                fps = frameCount * 1000.0f / elapsed;
                frameCount = 0;
                startTime = currentTime;
            }

            // ===== Draw FPS =====
            cv::putText(frameBgr, cv::format("FPS: %.1f", fps), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

            // B5: Hiển thị
            camera.show("Face Detection System", frameBgr);

            // Nhấn 'e' hoặc ESC để thoát
            char key = (char)cv::waitKey(1);
            if (key == 'e' || key == 27) break;
        }
    }
    // Bắt các lỗi liên quan đến OpenCV (ví dụ: lỗi tính toán ma trận, format ảnh sai)
    catch (const cv::Exception& e) {
        std::cerr << "\n[CRITICAL OPENCV ERROR]: " << e.what() << std::endl;
        std::cerr << "Code: " << e.code << ", Func: " << e.func << ", Line: " << e.line << std::endl;
    }
    // Bắt các lỗi C++ chuẩn (ví dụ: lỗi cấp phát bộ nhớ, runtime_error do ta ném ra ở trên)
    catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL SYSTEM ERROR]: " << e.what() << std::endl;
    }
    // Bắt tất cả các lỗi không xác định còn lại
    catch (...) {
        std::cerr << "\n[UNKNOWN ERROR]: An unexpected error occurred!" << std::endl;
    }

    // Cleanup - Luôn chạy dù có lỗi hay không
    std::cout << "[INFO] Cleaning up resources..." << std::endl;
    camera.release();
    cv::destroyAllWindows();
    std::cout << "=== SYSTEM STOPPED ===" << std::endl;

    return 0;
}