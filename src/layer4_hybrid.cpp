// ========================== Nguyen Hien ==========================
// FILE: src/layer4_hybrid.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer4_hybrid.h"
#include <iostream>

Layer4Hybrid::Layer4Hybrid() {}
Layer4Hybrid::~Layer4Hybrid() {}

double Layer4Hybrid::analyzeBlur(const cv::Mat& src) {
    cv::Mat gray, laplacian;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;

    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // Variance = stddev^2. Giá trị càng cao -> Ảnh càng nét (Real).
    // Ảnh chụp màn hình/giấy in thường có variance thấp hơn.
    return stddev.val[0] * stddev.val[0]; 
}

double Layer4Hybrid::analyzeFrequencyEnergy(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;

    // Mở rộng ảnh kích thước tối ưu cho DFT
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Tạo plane phức số
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    // DFT
    cv::dft(complexI, complexI);

    // Tính magnitude
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    // Chuyển sang scale logarit để dễ phân tích
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    // Cắt phổ tần số (Spectrum crop) để bỏ phần DC (giữa)
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    
    // Tính trung bình năng lượng tần số cao
    cv::Scalar meanVal = cv::mean(magI);
    return meanVal.val[0];
}

bool Layer4Hybrid::verifyDeepAnalysis(const cv::Mat& frame, const cv::Rect& faceBox, float layer3Score) {
    // Cắt khuôn mặt chính xác (tight crop) để phân tích texture
    cv::Mat faceRoi = frame(faceBox).clone();
    if (faceRoi.empty()) return false;

    // --- CHECK 1: BLUR DETECTION (Laplacian) ---
    double blurScore = analyzeBlur(faceRoi);
    // Ngưỡng thực nghiệm: < 100 thường là ảnh mờ/màn hình kém chất lượng
    // Nếu Layer 3 đang nghi ngờ (0.4-0.8), mà ảnh lại mờ -> Khả năng cao là Spoof
    bool passBlur = (blurScore > 150.0); 

    // --- CHECK 2: FREQUENCY ANALYSIS ---
    // Kiểm tra xem ảnh có "độ sâu" thông tin không
    double freqEnergy = analyzeFrequencyEnergy(faceRoi);
    bool passFreq = (freqEnergy > 10.0); // Ngưỡng ví dụ, cần tune thực tế

    // --- TỔNG HỢP LOGIC ---
    // Nếu cả 2 chỉ số vật lý đều tốt, ta cộng điểm cho Layer 3
    if (passBlur && passFreq) {
        // Boost điểm lên vì đã qua vòng kiểm tra vật lý
        return true; 
    } else {
        // Nếu ảnh mờ hoặc phổ tần số thấp -> Giữ nguyên nghi ngờ hoặc đánh rớt
        // Tuy nhiên, để an toàn, nếu Layer 3 Score đã khá cao (ví dụ 0.75) 
        // mà Blur hơi thấp (do camera focus sai) thì vẫn nên châm chước.
        
        if (layer3Score > 0.75 && passFreq) return true; // Cứu vớt
        return false; // Đánh rớt
    }
}