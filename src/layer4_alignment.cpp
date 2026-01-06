// ========================== Nguyen Hien ==========================
// FILE: src/layer4_alignment.cpp
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer4_alignment.h"
#include <opencv2/imgproc.hpp>

cv::Mat Layer4Alignment::align(const cv::Mat& frame,
                               const cv::Point2f& leftEye,
                               const cv::Point2f& rightEye) {
    static const cv::Point2f dstLeftEye(38.2946f, 51.6963f);
    static const cv::Point2f dstRightEye(73.5318f, 51.5014f);

    std::vector<cv::Point2f> src = { leftEye, rightEye };
    std::vector<cv::Point2f> dst = { dstLeftEye, dstRightEye };

    cv::Mat M = cv::estimateAffinePartial2D(
        src, dst,
        cv::noArray(),
        cv::RANSAC
    );

    if (M.empty()) {
        return cv::Mat(); 
    }

    cv::Mat aligned;
    cv::warpAffine(
        frame,
        aligned,
        M,
        cv::Size(112, 112),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0)
    );

    return aligned;
}

