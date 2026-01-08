// ========================== Nguyen Hien ==========================
// FILE: src/main.cpp (HYBRID CASCADE ARCHITECTURE)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include <iostream>
#include <string>
#include <chrono> 
#include <exception> 

#include "layer1_capture.h"
#include "layer2_detection.h"
#include "layer3_liveness.h" // Đã cập nhật enum LivenessStatus
#include "layer4_hybrid.h"   // Module mới

int main(int argc, char** argv) {
    std::cout << "=== FACE ANTI-SPOOFING SYSTEM (HYBRID CASCADE) ===" << std::endl;
    
    // Khởi tạo các lớp xử lý
    Layer1Capture camera;
    Layer2Detection detector;
    Layer3Liveness livenessLayer3;
    Layer4Hybrid   hybridLayer4;   
    
    try {
        // ===== 1. Init Camera =====
        if (!camera.init(0, 640, 480)) 
            throw std::runtime_error("[main] Failed to init camera! Check connection.");

        // ===== 2. Init Face Detector (YuNet) =====
        if (!detector.init("models/face_detection_yunet_2023mar.onnx")) 
            throw std::runtime_error("[main] Detector Init Failed");

        // ===== 3. Init Liveness Layer 3 (MiniFASNet) =====
        if (!livenessLayer3.init("models/MiniFASNetV1SE.onnx")) 
            throw std::runtime_error("[main] Liveness Init Failed");

        // Các biến dữ liệu ảnh
        cv::Mat frameBgr;
        FaceResult faceResult;
        LivenessResult liveResult; // Struct mới từ Layer 3
        
        // ===== Biến kiểm soát logic (Logic Control Variables) =====
        int realFrameCounter = 0;           // Đếm số frame Real liên tiếp
        const int FRAMES_TO_UNLOCK = 10;    // Giảm xuống 10 để cảm giác nhanh hơn nhờ Layer 4 hỗ trợ
        const int MIN_FACE_SIZE = 90;       // Mặt nhỏ hơn 90px sẽ không check

        // FPS vars
        int frameCount = 0;
        auto startTime = std::chrono::steady_clock::now();
        float fps = 0.0f;

        std::cout << "[main] INFO: System running. Press 'e' or 'ESC' to exit." << std::endl;

        // ===== Main Loop =====
        while (true) {
            // B1: Grab Frame
            if (!camera.grabFrame(frameBgr)) {
                std::cerr << "[main] WARN: Frame capture failed." << std::endl;
                break;
            }
            
            // B2: Detect Face
            bool found = detector.detect(frameBgr, faceResult);

            if (found) {
                // [OPTIMIZATION 1]: Kiểm tra kích thước khuôn mặt
                bool isTooSmall = (faceResult.bbox.width < MIN_FACE_SIZE || faceResult.bbox.height < MIN_FACE_SIZE);

                if (isTooSmall) {
                    cv::rectangle(frameBgr, faceResult.bbox, cv::Scalar(255, 255, 0), 2); // Cyan
                    cv::putText(frameBgr, "Move Closer", 
                        cv::Point(faceResult.bbox.x, faceResult.bbox.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
                    
                    realFrameCounter = 0;
                    livenessLayer3.resetHistory();
                } 
                else {
                    // B3: Check Liveness (Layer 3 - Deep Learning)
                    // Hàm này trả về Status: REAL, SPOOF hoặc UNCERTAIN
                    livenessLayer3.checkLiveness(frameBgr, faceResult.bbox, liveResult);

                    bool isFinalReal = false;
                    std::string debugTag = "";
                    cv::Scalar color;
                    std::string statusText;

                    // --- [CORE LOGIC: CASCADE SYSTEM] ---
                    
                    // CASE 1: Layer 3 rất tự tin là thật (> 0.85)
                    if (liveResult.status == LivenessStatus::REAL) {
                        isFinalReal = true;
                        debugTag = "L3-PASS"; // Layer 3 thông qua trực tiếp
                    }
                    // CASE 2: Layer 3 rất tự tin là giả (< 0.40)
                    else if (liveResult.status == LivenessStatus::SPOOF) {
                        isFinalReal = false;
                        debugTag = "L3-FAIL"; // Layer 3 chặn trực tiếp
                    }
                    // CASE 3: Layer 3 nghi ngờ (0.40 - 0.85) -> Kích hoạt Layer 4
                    else {
                        // Gọi Layer 4: Phân tích sâu (Tần số & Độ nét)
                        // Đây là bước lọc cuối cùng (Final Filter)
                        bool passedLayer4 = hybridLayer4.verifyDeepAnalysis(frameBgr, faceResult.bbox, liveResult.score);
                        
                        if (passedLayer4) {
                            isFinalReal = true;
                            debugTag = "L4-VERIFIED"; // Layer 4 xác nhận cứu vớt
                        } else {
                            isFinalReal = false;
                            debugTag = "L4-REJECT"; // Layer 4 bác bỏ (do mờ hoặc mất tần số cao)
                        }
                    }

                    // --- XỬ LÝ KẾT QUẢ CUỐI CÙNG ---
                    if (isFinalReal) {
                        realFrameCounter++;
                        
                        if (realFrameCounter >= FRAMES_TO_UNLOCK) {
                            color = cv::Scalar(0, 255, 0); // Green
                            statusText = "REAL PERSON";
                            if (realFrameCounter > 100) realFrameCounter = FRAMES_TO_UNLOCK + 1;
                        } else {
                            color = cv::Scalar(0, 255, 255); // Yellow
                            statusText = "Verifying... " + std::to_string(realFrameCounter * 100 / FRAMES_TO_UNLOCK) + "%";
                        }
                    } else {
                        // Reset ngay lập tức nếu phát hiện giả
                        realFrameCounter = 0;
                        livenessLayer3.resetHistory(); // Xóa buffer điểm số để tránh quán tính sai
                        
                        color = cv::Scalar(0, 0, 255); // Red
                        statusText = "FAKE PERSON";
                    }

                    // ---- DRAW UI ----
                    cv::rectangle(frameBgr, faceResult.bbox, color, 2);
                    
                    // 1. Vẽ trạng thái chính (Header)
                    std::string mainLabel = cv::format("%s", statusText.c_str());
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(mainLabel, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
                    cv::rectangle(frameBgr, 
                        cv::Point(faceResult.bbox.x, faceResult.bbox.y - labelSize.height - 10),
                        cv::Point(faceResult.bbox.x + labelSize.width, faceResult.bbox.y),
                        color, -1); 
                    cv::putText(frameBgr, mainLabel, 
                        cv::Point(faceResult.bbox.x, faceResult.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

                    // 2. Vẽ thông tin kỹ thuật (Footer - Debug Info)
                    // Hiển thị Score và Tag nguồn (L3 hay L4) để dễ debug
                    std::string techInfo = cv::format("Score: %.2f [%s]", liveResult.score, debugTag.c_str());
                    cv::putText(frameBgr, techInfo, 
                        cv::Point(faceResult.bbox.x, faceResult.bbox.y + faceResult.bbox.height + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                }
                
                // Vẽ Landmarks (để biết detect chuẩn không)
                for (const auto& point : faceResult.landmarks) {
                    cv::circle(frameBgr, point, 2, cv::Scalar(0, 255, 255), -1);
                }

            } else {
                // Không thấy mặt -> Reset toàn bộ
                realFrameCounter = 0;
                livenessLayer3.resetHistory();
            }

            // ===== FPS & Show =====
            frameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
            if (elapsed >= 1000) {
                fps = frameCount * 1000.0f / elapsed;
                frameCount = 0;
                startTime = currentTime;
            }

            cv::putText(frameBgr, cv::format("FPS: %.1f", fps), cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

            camera.show("Hybrid Anti-Spoofing System", frameBgr);

            char key = (char)cv::waitKey(1);
            if (key == 'e' || key == 27) break;
        }
    }
    // Bắt các lỗi liên quan đến OpenCV (ví dụ: lỗi tính toán ma trận, format ảnh sai)
    catch (const cv::Exception& e) {
        std::cerr << "\n[main]CRITICAL OPENCV ERROR: " << e.what() << std::endl;
        std::cerr << "Code: " << e.code << ", Func: " << e.func << ", Line: " << e.line << std::endl;
    }
    // Bắt các lỗi C++ chuẩn (ví dụ: lỗi cấp phát bộ nhớ, runtime_error do ta ném ra ở trên)
    catch (const std::exception& e) {
        std::cerr << "\n[main]CRITICAL SYSTEM ERROR: " << e.what() << std::endl;
    }
    // Bắt tất cả các lỗi không xác định còn lại
    catch (...) {
        std::cerr << "\n[main]UNKNOWN ERROR!" << std::endl;
    }

    // Cleanup - Luôn chạy dù có lỗi hay không
    std::cout << "[main]INFO: Resources released..." << std::endl;
    camera.release();
    cv::destroyAllWindows();
    std::cout << "======= SYSTEM STOPPED =======" << std::endl;

    return 0;
}

