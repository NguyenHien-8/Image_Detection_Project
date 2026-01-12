// ========================== Nguyen Hien ==========================
// FILE: src/main.cpp (BALANCED ANTI-SPOOFING)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include "layer1_capture.h"
#include "layer2_detection.h"
#include "layer3_liveness.h"
#include "layer4_hybrid.h"

int main(int argc, char** argv) {
    std::cout << "=== ADVANCED ANTI-SPOOFING SYSTEM v2.2 ===" << std::endl;
    
    Layer1Capture camera;
    Layer2Detection detector;
    Layer3Liveness livenessLayer3;
    Layer4Hybrid   hybridLayer4;   
    
    try {
        // ===== 1. Init Camera =====
        if (!camera.init()) {
            throw std::runtime_error("[main] Failed to init camera! Check connection.");
        }

        // ===== 2. Init Face Detector =====
        if (!detector.init("models/face_detection_yunet_2023mar.onnx")) 
            throw std::runtime_error("[main] Detector Init Failed");

        // ===== 3. Init Liveness Layer 3 =====
        if (!livenessLayer3.init("models/MiniFASNetV1SE.onnx")) 
            throw std::runtime_error("[main] Liveness Init Failed");

        cv::Mat frameBgr; 
        FaceResult faceResult;
        LivenessResult liveResult;
        
        int realConsecutive = 0;
        int spoofConsecutive = 0;
        int missingFaceCounter = 0;
        
        float lastRealScore = -1.0f;
        int suddenDropCount = 0;
        float confidenceAccumulator = 0.0f;
        
        int minFaceWidth = camera.getMinFaceWidth(); 
        cv::Size captureSize = camera.getCaptureSize();
        float fontScale = std::max(0.5f, captureSize.width / 1200.0f);
        int thickness = std::max(1, (int)(fontScale * 2));

        std::cout << "[main] System Running. Resolution: " << captureSize << std::endl;

        // ===== Main Loop =====
        while (true) {
            if (!camera.grabFrame(frameBgr)) break;
            
            bool found = detector.detect(frameBgr, faceResult);

            if (found) {
                missingFaceCounter = 0;

                if (faceResult.bbox.width < minFaceWidth) {
                     cv::rectangle(frameBgr, faceResult.bbox, cv::Scalar(0, 255, 255), 2);
                     livenessLayer3.resetHistory();
                     realConsecutive = 0; 
                     spoofConsecutive = 0;
                     lastRealScore = -1.0f;
                     confidenceAccumulator = 0.0f;
                } else {
                    // Liveness check
                    livenessLayer3.checkLiveness(frameBgr, faceResult.bbox, liveResult);
                    float rawScore = livenessLayer3.getLastRawScore();
                    
                    // Quality analysis
                    float adjustment = hybridLayer4.analyzeQuality(frameBgr, faceResult.bbox);
                    
                    // CRITICAL FIX 1: Tăng weight Layer3, giảm ảnh hưởng Layer4
                    float finalScore = liveResult.score * 0.75f + adjustment * 0.25f; // Từ 0.70/0.30
                    
                    // CRITICAL FIX 2: Penalty nhẹ hơn cho adjustment âm
                    if (adjustment < -0.45f) {
                        finalScore *= 0.80f; // Từ 0.75
                    } else if (adjustment < -0.35f) {
                        finalScore *= 0.90f; // Từ 0.85
                    }
                    
                    finalScore = std::max(0.0f, std::min(1.0f, finalScore));

                    // Swap attack detection
                    if (lastRealScore > 0.70f && rawScore < 0.35f) {
                        suddenDropCount++;
                        if (suddenDropCount >= 3) { // Tăng từ 2
                            finalScore = std::min(finalScore, 0.35f);
                            spoofConsecutive = std::max(spoofConsecutive, 2);
                        }
                    } else if (rawScore > 0.60f) {
                        suddenDropCount = 0;
                    }

                    // CRITICAL FIX 3: Nới lỏng threshold
                    bool passLayer3Basic = (liveResult.score > 0.60f);  // Từ 0.65
                    bool passLayer3Strong = (liveResult.score > 0.70f); // Từ 0.75
                    bool passLayer4 = (adjustment > -0.40f);            // Từ -0.35
                    bool passLayer4Strong = (adjustment > -0.20f);      // Từ -0.15
                    
                    // CRITICAL FIX 4: Điều kiện REAL dễ dàng hơn
                    bool isStrongReal = (finalScore > 0.75f && passLayer3Strong && passLayer4Strong);
                    bool isWeakReal = (finalScore > 0.65f && passLayer3Basic && passLayer4);
                    
                    // CRITICAL FIX 5: Điều kiện FAKE chặt chẽ hơn (tránh false positive)
                    bool isStrongFake = (finalScore < 0.30f || adjustment < -0.50f || liveResult.score < 0.25f);
                    bool isWeakFake = (finalScore < 0.45f && liveResult.score < 0.55f && adjustment < -0.35f);

                    if (isStrongReal || isWeakReal) {
                        realConsecutive++;
                        spoofConsecutive = 0;
                        lastRealScore = finalScore;
                        
                        if (isStrongReal) {
                            confidenceAccumulator += 2.0f;
                        } else {
                            confidenceAccumulator += 1.5f; // Tăng từ 1.0
                        }
                    } else {
                        realConsecutive = 0;
                        confidenceAccumulator = 0.0f;
                        
                        if (isStrongFake || isWeakFake) {
                            spoofConsecutive++;
                            lastRealScore = -1.0f;
                        }
                    }

                    // CRITICAL FIX 6: Logic hiển thị cân bằng
                    cv::Scalar color;
                    
                    // REAL: Giảm yêu cầu frames và confidence
                    if (realConsecutive >= 4 && confidenceAccumulator >= 6.0f) {
                        color = cv::Scalar(0, 255, 0); // XANH
                    } 
                    // FAKE: Tăng yêu cầu để chắc chắn
                    else if (spoofConsecutive >= 4 || isStrongFake) {
                        color = cv::Scalar(0, 0, 255); // ĐỎ
                    }
                    // Analyzing
                    else {
                        color = cv::Scalar(0, 255, 255); // VÀNG
                    }

                    // Vẽ khung theo màu
                    cv::rectangle(frameBgr, faceResult.bbox, color, 2);
                }
            } else {
                // Không tìm thấy mặt
                missingFaceCounter++;
                if (missingFaceCounter > 10) {
                    realConsecutive = 0;
                    spoofConsecutive = 0;
                    lastRealScore = -1.0f;
                    suddenDropCount = 0;
                    confidenceAccumulator = 0.0f;
                    livenessLayer3.resetHistory();
                }
            }

            camera.show("Anti-Spoofing Pro v2.2", frameBgr);
            
            char key = cv::waitKey(1);
            if (key == 27) break; // ESC
            if (key == 'r' || key == 'R') {
                realConsecutive = 0;
                spoofConsecutive = 0;
                confidenceAccumulator = 0.0f;
                livenessLayer3.resetHistory();
                std::cout << "[main] Manual reset triggered" << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "An unknown error occurred!" << std::endl;
    }

    camera.release();
    cv::destroyAllWindows();
    std::cout << "======= SYSTEM STOPPED =======" << std::endl;

    return 0;
}