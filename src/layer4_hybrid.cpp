// ========================== Nguyen Hien ==========================
// FILE: src/layer4_hybrid.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer4_hybrid.h"
#include <numeric>
#include <cmath>

Layer4Hybrid::Layer4Hybrid() {}
Layer4Hybrid::~Layer4Hybrid() {}

// Tính toán nhiễu tần số cao (High Frequency Noise)
double Layer4Hybrid::calculateHighFrequency(const cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    // Sử dụng DFT để tìm năng lượng tần số cao
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    
    cv::Mat magI = planes[0];
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Replay Attack thường có các điểm sáng bất thường ở tần số cực cao (do lưới pixel)
    // Chúng ta cắt bỏ phần trung tâm (low freq) và tính trung bình phần rìa
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    
    // Mask che phần trung tâm (thông tin cấu trúc khuôn mặt)
    // Chỉ để lại tần số cao (chi tiết da/nhiễu/lưới pixel)
    cv::Mat mask = cv::Mat::ones(magI.size(), CV_8U);
    int maskSize = std::min(magI.cols, magI.rows) / 4; // Che 1/4 vùng trung tâm
    cv::rectangle(mask, cv::Rect(cx - maskSize, cy - maskSize, maskSize*2, maskSize*2), cv::Scalar(0), -1);

    cv::Scalar meanVal = cv::mean(magI, mask);
    return meanVal.val[0];
}

// Kiểm tra tính chất quang học của da trong không gian YCrCb
bool Layer4Hybrid::checkSkinConsistency(const cv::Mat& src, float& outScore) {
    cv::Mat ycrcb;
    cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
    
    std::vector<cv::Mat> planes;
    cv::split(ycrcb, planes);
    
    // Kênh Cr (Red-difference) là quan trọng nhất đối với da người
    // Da người thật: Cr thường nằm trong khoảng [133, 173] và Cb [77, 127]
    // Replay/Màn hình: Thường có Cr thấp hơn (ám xanh) hoặc Cb cao hơn
    
    cv::Scalar meanCr, stdCr;
    cv::meanStdDev(planes[1], meanCr, stdCr);
    
    // Logic heuristic:
    // Da thật có sự biến thiên (StdDev) nhất định nhưng Mean phải chuẩn.
    // Màn hình thường có màu sắc "phẳng" (Std thấp) hoặc sai màu (Mean lệch).
    
    float score = 0.0f;
    
    // 1. Kiểm tra sắc đỏ (Vitality Check)
    if (meanCr.val[0] >= 140 && meanCr.val[0] <= 165) {
        score += 0.2f; // Màu da chuẩn
    } else {
        score -= 0.3f; // Màu da bất thường (quá nhợt nhạt hoặc quá đỏ do màn hình)
    }

    // 2. Kiểm tra độ phản xạ (Specular highlight) qua kênh Y (Luminance)
    // Màn hình phát sáng đều, da thật có bóng đổ.
    double minVal, maxVal;
    cv::minMaxLoc(planes[0], &minVal, &maxVal);
    if ((maxVal - minVal) < 50) { 
        score -= 0.2f; // Tương phản quá thấp -> Màn hình phẳng
    } else {
        score += 0.1f;
    }

    outScore = score;
    return true;
}

float Layer4Hybrid::analyzeQuality(const cv::Mat& frame, const cv::Rect& faceBox) {
    // Crop rộng hơn một chút để lấy cả viền nếu có, nhưng tính toán chủ yếu trong safeBox
    cv::Rect safeBox = faceBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.area() == 0) return 0.0f;
    
    cv::Mat faceRoi = frame(safeBox).clone();
    cv::resize(faceRoi, faceRoi, cv::Size(128, 128)); 

    double freqHigh = calculateHighFrequency(faceRoi);
    float skinScore = 0.0f;
    checkSkinConsistency(faceRoi, skinScore);

    float totalAdjustment = 0.0f;

    // --- LOGIC PHÁT HIỆN REPLAY ---

    // 1. Moiré Pattern / Pixel Grid Check
    // Ảnh thật thường có freq ~ 9-11. Màn hình có lưới pixel sẽ đẩy freq lên > 12 hoặc làm mịn quá mức < 7
    // Replay video chất lượng cao thường có High Freq bất thường do nhiễu số.
    if (freqHigh > 13.5) { 
        totalAdjustment -= 0.4f; // PHẠT RẤT NẶNG: Phát hiện nhiễu lưới pixel (Moiré)
    } else if (freqHigh < 6.0) {
        totalAdjustment -= 0.2f; // Quá mờ/mịn (Replay màn hình kém)
    } else {
        totalAdjustment += 0.1f; // Độ chi tiết tự nhiên
    }

    // 2. Skin Color Physics
    totalAdjustment += skinScore;

    // Kẹp giá trị trả về
    if (totalAdjustment > 0.3f) totalAdjustment = 0.3f;
    if (totalAdjustment < -0.5f) totalAdjustment = -0.5f; // Ưu tiên phạt nặng để chặn giả

    return totalAdjustment;
}