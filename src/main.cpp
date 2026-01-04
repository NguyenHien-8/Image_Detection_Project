// ============================================
// FILE: src/main.cpp
// ============================================
#include "layer1_capture.h"
#include "layer2_detection.h"
#include "layer3_liveness.h"
#include "layer4_alignment.h"
#include "layer5_packaging.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

class FaceDetectionPipeline {
private:
    // Layer objects
    VideoCapture* video_capture;
    FaceDetector* face_detector;
    LivenessDetector* liveness_detector;
    FaceAligner* face_aligner;
    WebSocketSender* websocket_sender;
    
    // Statistics
    int total_frames;
    int faces_detected;
    int live_faces;
    int fake_faces;
    
    // Configuration
    std::string model_proto_path;
    std::string model_weights_path;
    std::string server_uri;
    std::string device_id;

public:
    FaceDetectionPipeline() 
        : video_capture(nullptr), face_detector(nullptr), 
          liveness_detector(nullptr), face_aligner(nullptr),
          websocket_sender(nullptr), total_frames(0), 
          faces_detected(0), live_faces(0), fake_faces(0) {}
    
    ~FaceDetectionPipeline() {
        cleanup();
    }
    
    bool initialize(const std::string& proto, const std::string& weights,
                   const std::string& server, const std::string& dev_id) {
        
        model_proto_path = proto;
        model_weights_path = weights;
        server_uri = server;
        device_id = dev_id;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Face Detection Pipeline Initialization" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // Layer 1: Video Capture
        std::cout << "[LAYER 1] Initializing video capture..." << std::endl;
        video_capture = new VideoCapture(0, 640, 480, 30);
        if (!video_capture->open()) {
            std::cerr << "[FATAL] Video capture failed" << std::endl;
            return false;
        }
        
        // Layer 2: Face Detection
        std::cout << "[LAYER 2] Initializing face detector..." << std::endl;
        face_detector = new FaceDetector(0.5f);
        if (!face_detector->loadModel(model_proto_path, model_weights_path)) {
            std::cerr << "[FATAL] Model loading failed" << std::endl;
            return false;
        }
        
        // Layer 3: Liveness Detection
        std::cout << "[LAYER 3] Initializing liveness detector..." << std::endl;
        liveness_detector = new LivenessDetector(15);
        liveness_detector->init();
        
        // Layer 4: Face Alignment
        std::cout << "[LAYER 4] Initializing face aligner..." << std::endl;
        face_aligner = new FaceAligner(112);
        
        // Layer 5: WebSocket Data Sender
        std::cout << "[LAYER 5] Initializing WebSocket sender..." << std::endl;
        websocket_sender = new WebSocketSender(device_id);
        if (!websocket_sender->connect(server_uri)) {
            std::cout << "[WARNING] WebSocket server unavailable - offline mode" 
                     << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "[SUCCESS] All layers initialized!" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return true;
    }
    
    void run() {
        std::cout << "Starting pipeline... (Press 'q' to quit, 's' to save)\n" << std::endl;
        
        cv::Mat frame;
        auto start_time = std::chrono::high_resolution_clock::now();
        int frame_count = 0;
        
        while (true) {
            // ==================== LAYER 1: CAPTURE ====================
            if (!video_capture->getFrame(frame)) {
                std::cerr << "[ERROR] Failed to capture frame" << std::endl;
                break;
            }
            
            total_frames++;
            frame_count++;
            
            // Skip every 2 frames for performance
            if (total_frames % 2 != 0) {
                displayFrame(frame, false);
                handleKeyPress();
                continue;
            }
            
            // ==================== LAYER 2: DETECTION ====================
            Face face_result = face_detector->detect(frame);
            
            if (!face_result.detected) {
                displayFrame(frame, false);
                handleKeyPress();
                continue;
            }
            
            faces_detected++;
            
            // ==================== LAYER 3: LIVENESS ====================
            LivenessInfo liveness_result = liveness_detector->detect(face_result.landmarks);
            
            if (liveness_result.is_live) {
                live_faces++;
            } else {
                fake_faces++;
            }
            
            // ==================== LAYER 4: ALIGNMENT ====================
            AlignedFace aligned_result = face_aligner->align(
                frame, face_result.landmarks);
            
            if (!aligned_result.success) {
                std::cerr << "[ERROR] Face alignment failed" << std::endl;
                handleKeyPress();
                continue;
            }
            
            // ==================== LAYER 5: PACKAGING ====================
            if (liveness_result.is_live && websocket_sender->isConnected()) {
                DataPackage package;
                package.face_image = aligned_result.face_image;
                package.device_id = device_id;
                package.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                package.blink_count = liveness_result.blink_count;
                package.liveness_confidence = liveness_result.confidence;
                package.rotation_angle = aligned_result.rotation_angle;
                
                websocket_sender->sendData(package);
            }
            
            // ==================== VISUALIZATION ====================
            drawResults(frame, face_result, liveness_result, aligned_result);
            displayFrame(frame, true);
            
            // ==================== LOGGING ====================
            if (frame_count % 30 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();
                
                float fps = total_frames / static_cast<float>(elapsed);
                
                std::cout << "\n[STATS] Frame: " << total_frames 
                         << " | FPS: " << std::fixed << std::setprecision(1) << fps
                         << " | Faces: " << faces_detected 
                         << " | Live: " << live_faces 
                         << " | Fake: " << fake_faces << std::endl;
            }
            
            handleKeyPress();
        }
        
        cleanup();
    }

private:
    void drawResults(cv::Mat& frame, const Face& face, 
                    const LivenessInfo& liveness, const AlignedFace& aligned) {
        
        if (!face.detected) return;
        
        // Draw bounding box
        cv::Scalar color = liveness.is_live ? 
                          cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(frame, face.bbox, color, 2);
        
        // Draw landmarks
        for (const auto& landmark : face.landmarks) {
            cv::circle(frame, cv::Point(landmark.x, landmark.y), 4, 
                      cv::Scalar(255, 0, 0), -1);
        }
        
        // Draw status text
        std::string status = liveness.is_live ? "LIVE" : "FAKE";
        std::string confidence_str = std::to_string(liveness.confidence);
        confidence_str = confidence_str.substr(0, 4);
        
        // Background box
        int x = face.bbox.x;
        int y = face.bbox.y - 35;
        cv::rectangle(frame, cv::Point(x, y - 25), 
                     cv::Point(x + 200, y + 10), color, -1);
        
        // Status text
        cv::putText(frame, status, cv::Point(x + 5, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                   cv::Scalar(255, 255, 255), 2);
        
        // Metadata
        cv::putText(frame, "Conf: " + confidence_str, 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Blinks: " + std::to_string(liveness.blink_count), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
        
        std::string rotation_str = std::to_string(aligned.rotation_angle);
        rotation_str = rotation_str.substr(0, 5);
        
        cv::putText(frame, "Angle: " + rotation_str + "°", 
                   cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 
                   0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    void displayFrame(const cv::Mat& frame, bool with_info) {
        cv::imshow("Face Detection Pipeline", frame);
    }
    
    void handleKeyPress() {
        int key = cv::waitKey(30) & 0xFF;
        
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::cout << "\n[INFO] Quitting..." << std::endl;
            cleanup();
            exit(0);
        } else if (key == 's') {  // 's' to save
            std::string filename = "face_" + std::to_string(faces_detected) + ".jpg";
            cv::Mat display_frame;
            cv::Mat temp;
            cv::imshow("Face Detection Pipeline", temp);
            std::cout << "[INFO] Face saved: " << filename << std::endl;
        }
    }
    
    void cleanup() {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Pipeline Shutdown" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // Statistics
        std::cout << "Total frames processed:  " << total_frames << std::endl;
        std::cout << "Faces detected:          " << faces_detected << std::endl;
        std::cout << "Live faces:              " << live_faces << std::endl;
        std::cout << "Fake faces:              " << fake_faces << std::endl;
        
        if (faces_detected > 0) {
            float live_rate = (100.0f * live_faces) / faces_detected;
            std::cout << "Live face rate:          " << std::fixed 
                     << std::setprecision(1) << live_rate << "%" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        
        // Cleanup resources
        if (websocket_sender) {
            websocket_sender->disconnect();
            delete websocket_sender;
        }
        if (face_aligner) delete face_aligner;
        if (liveness_detector) delete liveness_detector;
        if (face_detector) delete face_detector;
        if (video_capture) delete video_capture;
        
        cv::destroyAllWindows();
        
        std::cout << "[INFO] Pipeline closed successfully" << std::endl;
    }
};

// ============================================
// MAIN ENTRY POINT
// ============================================
int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "================================================" << std::endl;
    std::cout << "   Face Detection Pipeline v1.0 (Pure C++)" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "\n";
    
    // Default parameters
    std::string proto_path = "./models/opencv_face_detector.pbtxt";
    std::string weights_path = "./models/opencv_face_detector_uint8.pb";
    std::string server_uri = "ws://localhost:8080/face";
    std::string device_id = "device_001";
    
    // Parse command line arguments
    if (argc > 1) proto_path = argv[1];
    if (argc > 2) weights_path = argv[2];
    if (argc > 3) server_uri = argv[3];
    if (argc > 4) device_id = argv[4];
    
    std::cout << "Model Proto:   " << proto_path << std::endl;
    std::cout << "Model Weights: " << weights_path << std::endl;
    std::cout << "Server URI:    " << server_uri << std::endl;
    std::cout << "Device ID:     " << device_id << std::endl;
    std::cout << "\n";
    
    // Create and run pipeline
    FaceDetectionPipeline pipeline;
    
    if (!pipeline.initialize(proto_path, weights_path, server_uri, device_id)) {
        std::cerr << "[FATAL] Pipeline initialization failed" << std::endl;
        return 1;
    }
    
    pipeline.run();
    
    return 0;
}