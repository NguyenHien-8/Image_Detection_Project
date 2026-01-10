// ========================== Nguyen Hien ==========================
// FILE: src/main.cpp (RGB PIPELINE - MODIFIED)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include <iostream>
#include <string>
#include <chrono>
#include "layer1_capture.h"
#include "layer2_detection.h"
#include "layer3_liveness.h"
#include "layer4_hybrid.h"

int main(int argc, char** argv) {
    std::cout << "=== OPTIMIZED ANTI-SPOOFING SYSTEM (RGB Mode) ===" << std::endl;
    
    Layer1Capture camera;
    Layer2Detection detector;
    Layer3Liveness livenessLayer3;
    Layer4Hybrid   hybridLayer4;   
    
    try {
        // ===== 1. Init Camera (outputs RGB) =====
        if (!camera.init(2, 640, 480)) 
            throw std::runtime_error("[main] Failed to init camera! Check connection.");

        // ===== 2. Init Face Detector (YuNet - handles RGB internally) =====
        if (!detector.init("models/face_detection_yunet_2023mar.onnx")) 
            throw std::runtime_error("[main] Detector Init Failed");

        // ===== 3. Init Liveness Layer 3 (MiniFASNet - RGB mode) =====
        if (!livenessLayer3.init("models/MiniFASNetV1SE.onnx")) 
            throw std::runtime_error("[main] Liveness Init Failed");

        // IMPORTANT: frameRgb now holds RGB data throughout the pipeline
        cv::Mat frameRgb;  // Changed variable name for clarity
        FaceResult faceResult;
        LivenessResult liveResult;
        
        int realConsecutive = 0;
        const int UNLOCK_THRESHOLD = 10;
        int spoofConsecutive = 0;
        float fps = 0.0f;
        int frameCount = 0;
        auto startTime = std::chrono::steady_clock::now();

        // ===== Main Loop =====
        while (true) {
            // B1: Grab Frame (automatically converted to RGB in Layer1)
            if (!camera.grabFrame(frameRgb)) {
                std::cerr << "[main] WARN: Frame capture failed." << std::endl;
                break;
            }
            
            // B2: Detect Face (Layer2 handles RGB->BGR conversion internally)
            bool found = detector.detect(frameRgb, faceResult);

            if (found) {
                if (faceResult.bbox.width < 100) { 
                     cv::putText(frameRgb, "Come Closer", 
                               cv::Point(faceResult.bbox.x, faceResult.bbox.y - 10), 
                               0, 0.6, cv::Scalar(255, 255, 0), 2);  // RGB: Yellow
                     realConsecutive = 0;
                } else {
                    // 1. Deep Learning Check (processes RGB)
                    livenessLayer3.checkLiveness(frameRgb, faceResult.bbox, liveResult);
                    
                    // 2. Heuristic Check (processes RGB)
                    float adjustment = hybridLayer4.analyzeQuality(frameRgb, faceResult.bbox);
                    
                    // --- FUSION LOGIC ---
                    float baseScore = liveResult.score;
                    float finalScore = baseScore;

                    if (adjustment < -0.25f) {
                        finalScore = std::min(baseScore, 0.2f); 
                    }
                    else if (baseScore > 0.3f && baseScore < 0.85f) {
                        finalScore += adjustment;
                    }
                    else if (baseScore >= 0.85f) {
                        if (adjustment < 0) finalScore += adjustment;
                    }

                    finalScore = std::max(0.0f, std::min(1.0f, finalScore));

                    // --- DECISION ---
                    bool isReal = (finalScore > 0.80f); 

                    cv::Scalar color;
                    std::string statusText;

                    if (isReal) {
                        realConsecutive++;
                        spoofConsecutive = 0;
                    } else {
                        realConsecutive = 0;
                        spoofConsecutive++;
                    }

                    if (realConsecutive >= UNLOCK_THRESHOLD) {
                        statusText = "REAL PERSON";
                        color = cv::Scalar(0, 255, 0);  // RGB: Green
                        cv::rectangle(frameRgb, faceResult.bbox, color, 3);
                    } else if (spoofConsecutive > 2) {
                        statusText = "SPOOF / REPLAY";
                        color = cv::Scalar(255, 0, 0);  // RGB: Red
                        cv::rectangle(frameRgb, faceResult.bbox, color, 2);
                    } else {
                        statusText = "Verifying...";
                        color = cv::Scalar(255, 255, 0);  // RGB: Yellow
                        cv::rectangle(frameRgb, faceResult.bbox, color, 1);
                    }

                    // Debug Info
                    std::string debug = cv::format("AI:%.2f Phy:%+.2f = %.2f", 
                                                  baseScore, adjustment, finalScore);
                    cv::putText(frameRgb, statusText, 
                              cv::Point(faceResult.bbox.x, faceResult.bbox.y - 10), 
                              0, 0.8, color, 2);
                    cv::putText(frameRgb, debug, 
                              cv::Point(faceResult.bbox.x, faceResult.bbox.y + faceResult.bbox.height + 25), 
                              0, 0.6, cv::Scalar(200, 200, 200), 1);
                }
            } else {
                realConsecutive = 0;
                livenessLayer3.resetHistory();
            }

            // FPS
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() >= 1000) {
                fps = (float)frameCount;
                frameCount = 0;
                startTime = now;
            }
            cv::putText(frameRgb, "FPS: " + std::to_string((int)fps), 
                       cv::Point(10, 30), 0, 0.6, cv::Scalar(0, 255, 0), 2);

            // Display (Layer1::show handles RGB->BGR conversion for cv::imshow)
            camera.show("Hybrid Anti-Spoofing V3 (RGB)", frameRgb);
            if (cv::waitKey(1) == 27) break;
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