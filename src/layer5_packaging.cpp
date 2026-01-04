// ============================================
// FILE: src/layer5_packaging.cpp
// ============================================
#include "layer5_packaging.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <ctime>

// Base64 encoding table
static const std::string base64_table = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64Encode(const unsigned char* data, size_t len) {
    std::string result;
    int val = 0;
    int valb = 0;
    
    for (size_t i = 0; i < len; i++) {
        val = (val << 8) + data[i];
        valb += 8;
        
        while (valb >= 6) {
            valb -= 6;
            result.push_back(base64_table[(val >> valb) & 0x3F]);
        }
    }
    
    if (valb > 0) {
        result.push_back(base64_table[(val << (6 - valb)) & 0x3F]);
    }
    
    while (result.size() % 4) {
        result.push_back('=');
    }
    
    return result;
}

WebSocketSender::WebSocketSender(const std::string& dev_id)
    : server_uri(""), device_id(dev_id), is_connected(false) {
    
    client.set_access_channels(websocketpp::log::alevel::all);
    client.clear_access_channels(websocketpp::log::alevel::frame_payload);
    client.init_asio();
}

WebSocketSender::~WebSocketSender() {
    disconnect();
}

std::string WebSocketSender::encodeImageToBase64(const cv::Mat& image) {
    // Encode image to JPEG format
    std::vector<uchar> buffer;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(90);  // Quality 90%
    
    cv::imencode(".jpg", image, buffer, params);
    
    // Convert to base64
    std::string encoded = base64Encode(buffer.data(), buffer.size());
    return "data:image/jpeg;base64," + encoded;
}

std::string WebSocketSender::createJsonPayload(const DataPackage& package) {
    // Create JSON using simple string concatenation
    // (avoiding external JSON library dependency)
    
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    
    std::stringstream ss;
    ss << "{"
       << "\"type\":\"face_data\","
       << "\"device_id\":\"" << device_id << "\","
       << "\"timestamp\":" << millis.count() << ","
       << "\"face_image\":\"" << encodeImageToBase64(package.face_image) << "\","
       << "\"metadata\":{"
       << "\"blink_count\":" << package.blink_count << ","
       << "\"liveness_confidence\":" << package.liveness_confidence << ","
       << "\"rotation_angle\":" << package.rotation_angle
       << "}"
       << "}";
    
    return ss.str();
}

bool WebSocketSender::connect(const std::string& uri) {
    server_uri = uri;
    
    try {
        websocketpp::lib::error_code ec;
        
        WebSocketClient::connection_ptr con = client.get_connection(uri, ec);
        if (ec) {
            std::cerr << "[ERROR] WebSocket connection failed: " 
                      << ec.message() << std::endl;
            return false;
        }
        
        connection = con->get_handle();
        client.connect(con);
        
        // Run client in separate thread
        client_thread = std::thread([this]() {
            client.run();
        });
        
        is_connected = true;
        std::cout << "[INFO] WebSocket connected to " << uri << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] WebSocket error: " << e.what() << std::endl;
        return false;
    }
}

bool WebSocketSender::sendData(const DataPackage& package) {
    if (!is_connected) {
        std::cerr << "[WARNING] WebSocket not connected" << std::endl;
        return false;
    }
    
    try {
        std::string payload = createJsonPayload(package);
        client.send(connection, payload, websocketpp::frame::opcode::text);
        
        std::cout << "[INFO] Face data sent (" << payload.size() << " bytes)" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Send failed: " << e.what() << std::endl;
        return false;
    }
}

void WebSocketSender::disconnect() {
    if (is_connected) {
        try {
            websocketpp::lib::error_code ec;
            client.close(connection, websocketpp::close::status::normal, "", ec);
            is_connected = false;
            std::cout << "[INFO] WebSocket disconnected" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Disconnect failed: " << e.what() << std::endl;
        }
    }
}