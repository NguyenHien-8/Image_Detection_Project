// ============================================
// FILE: include/layer5_packaging.h
// ============================================
#ifndef LAYER5_PACKAGING_H
#define LAYER5_PACKAGING_H

#include <opencv2/opencv.hpp>
#include <string>
#include <json/json.h>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <memory>
#include <thread>

typedef websocketpp::client<websocketpp::config::asio_client> WebSocketClient;
typedef websocketpp::connection_hdl ConnectionHandle;

struct DataPackage {
    cv::Mat face_image;
    std::string device_id;
    long long timestamp;
    int blink_count;
    float liveness_confidence;
    float rotation_angle;
};

class WebSocketSender {
private:
    WebSocketClient client;
    ConnectionHandle connection;
    std::string server_uri;
    std::string device_id;
    bool is_connected;
    std::thread client_thread;
    
    std::string encodeImageToBase64(const cv::Mat& image);
    std::string createJsonPayload(const DataPackage& package);
    
    void onOpen(ConnectionHandle hdl);
    void onClose(ConnectionHandle hdl);
    void onFail(ConnectionHandle hdl);

public:
    WebSocketSender(const std::string& dev_id);
    ~WebSocketSender();
    
    bool connect(const std::string& uri);
    bool sendData(const DataPackage& package);
    bool isConnected() const { return is_connected; }
    void disconnect();
};

#endif // LAYER5_PACKAGING_H