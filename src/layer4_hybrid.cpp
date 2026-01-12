// ========================== Nguyen Hien ==========================
// FILE: src/layer4_hybrid.cpp (BALANCED ANTI-SPOOFING)
// Developer: TRAN NGUYEN HIEN
// Email: trannguyenhien29085@gmail.com
// =================================================================
#include "layer4_hybrid.h"
#include <numeric>
#include <cmath>

Layer4Hybrid::Layer4Hybrid() {}
Layer4Hybrid::~Layer4Hybrid() {}

float Layer4Hybrid::analyzeTextureGradient(const cv::Mat& src) {
    if (src.empty()) return 0.0f;
    if (src.channels() == 3) cv::cvtColor(src, grayBuffer, cv::COLOR_BGR2GRAY);
    else src.copyTo(grayBuffer);
    
    cv::Sobel(grayBuffer, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(grayBuffer, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, magnitude);
    cv::Scalar meanGrad = cv::mean(magnitude);
    cv::Scalar stdGrad;
    cv::meanStdDev(magnitude, cv::Scalar(), stdGrad); 
    
    float ratio = stdGrad.val[0] / (meanGrad.val[0] + 1e-6);
    float gradientScore = 0.0f;

    if (ratio > 0.7f && ratio < 1.8f) {
        gradientScore += 0.20f; 
    } else if (ratio < 0.4f) { 
        gradientScore -= 0.35f;
    } else if (ratio > 2.2f) {
        gradientScore -= 0.25f; 
    } else {
        gradientScore -= 0.10f;
    }
    
    if (meanGrad.val[0] < 3.5f) {
        gradientScore -= 0.25f;
    } else if (meanGrad.val[0] > 25.0f) {
        gradientScore += 0.15f;
    } else if (meanGrad.val[0] > 8.0f) {
        gradientScore += 0.05f;
    }
    
    return gradientScore;
}

float Layer4Hybrid::detectMoirePattern(const cv::Mat& src) {
    if (src.empty()) return 0.0f;
    cv::resize(src, moireResized, cv::Size(128, 128));
    if (moireResized.channels() == 3) cv::cvtColor(moireResized, moireGray, cv::COLOR_BGR2GRAY);
    else moireGray = moireResized;
    
    cv::Laplacian(moireGray, moireLaplacian, CV_32F, 3);
    cv::convertScaleAbs(moireLaplacian, moireLaplacian);
    cv::Scalar mean, stddev;
    cv::meanStdDev(moireLaplacian, mean, stddev);
    float variance = stddev.val[0] * stddev.val[0];
    
    if (variance > 1200) return -0.45f;
    if (variance > 850) return -0.30f; 
    if (variance < 100) return -0.15f; 
    if (variance >= 150 && variance <= 500) return 0.15f; 
    return 0.0f;
}

double Layer4Hybrid::calculateHighFrequency(const cv::Mat& src) {
    if (src.empty()) return 0.0;

    cv::resize(src, resizedBuffer, cv::Size(128, 128));

    if (resizedBuffer.channels() == 3)
        cv::cvtColor(resizedBuffer, grayBuffer, cv::COLOR_BGR2GRAY);
    else
        resizedBuffer.copyTo(grayBuffer);

    int m = cv::getOptimalDFTSize(grayBuffer.rows);
    int n = cv::getOptimalDFTSize(grayBuffer.cols);
    
    cv::copyMakeBorder(grayBuffer, padded, 0, m - grayBuffer.rows, 0, n - grayBuffer.cols, 
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    if (plane0.size() != padded.size()) {
        plane0 = cv::Mat::zeros(padded.size(), CV_32F);
        plane1 = cv::Mat::zeros(padded.size(), CV_32F);
    }
    
    padded.convertTo(plane0, CV_32F);
    plane1.setTo(0);

    std::vector<cv::Mat> planes = {plane0, plane1};
    cv::merge(planes, complexI);
    
    cv::dft(complexI, complexI);
    
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    magI = planes[0];

    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    int maskSize = std::min(magI.cols, magI.rows) / 6; 
    cv::Rect centerRect(cx - maskSize, cy - maskSize, maskSize * 2, maskSize * 2);
    
    if (mask.size() != magI.size()) {
        mask = cv::Mat::ones(magI.size(), CV_8U);
        cv::rectangle(mask, centerRect, cv::Scalar(0), -1);
    } 
    
    cv::Scalar highFreqMean = cv::mean(magI, mask);
    return highFreqMean.val[0];
}

bool Layer4Hybrid::checkSkinConsistency(const cv::Mat& src, float& outScore) {
    if (src.empty()) return false;
    cv::resize(src, skinSmall, cv::Size(64, 64));
    cv::cvtColor(skinSmall, skinYCrCb, cv::COLOR_BGR2YCrCb);
    
    double sumY = 0, sumCr = 0, sumCb = 0;
    double minY = 255, maxY = 0;
    int totalPixels = skinSmall.rows * skinSmall.cols;

    if (skinYCrCb.isContinuous()) {
        const uchar* ptr = skinYCrCb.ptr<uchar>(0);
        for (int i = 0; i < totalPixels; ++i) {
            uchar y = ptr[3*i];
            uchar cr = ptr[3*i + 1];
            uchar cb = ptr[3*i + 2];
            sumY += y;
            sumCr += cr;
            sumCb += cb;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
    } else {
        cv::Scalar m = cv::mean(skinYCrCb);
        sumY = m[0] * totalPixels;
        sumCr = m[1] * totalPixels;
        sumCb = m[2] * totalPixels;
        double min, max;
        cv::minMaxLoc(skinSmall, &min, &max);
        minY = min; maxY = max;
    }

    float meanCrVal = (float)(sumCr / totalPixels);
    float meanCbVal = (float)(sumCb / totalPixels);
    float score = 0.0f;
    bool validCr = (meanCrVal >= 125 && meanCrVal <= 180);
    bool validCb = (meanCbVal >= 70 && meanCbVal <= 135);  
    
    if (validCr && validCb) {
        score += 0.25f;
    } else if (!validCr && !validCb) {
        score -= 0.40f;
    } else {
        score -= 0.15f; 
    }

    double contrast = maxY - minY;
    if (contrast < 25) {
        score -= 0.30f; 
    } else if (contrast > 140) {
        score -= 0.15f;
    } else if (contrast >= 40 && contrast <= 120) {
        score += 0.15f; 
    }
    
    cv::cvtColor(skinSmall, skinHSV, cv::COLOR_BGR2HSV);
    int from_to[] = {1, 0};
    if (plane0.size() != skinHSV.size()) plane0 = cv::Mat(skinHSV.size(), CV_8U);
    cv::mixChannels(&skinHSV, 1, &plane0, 1, from_to, 1);
    cv::Scalar satMean = cv::mean(plane0);
    
    if (satMean.val[0] >= 15 && satMean.val[0] <= 90) {
        score += 0.15f; 
    } else if (satMean.val[0] < 8 || satMean.val[0] > 110) {
        score -= 0.25f; 
    } else {
        score -= 0.08f; 
    }

    outScore = score;
    return true;
}

float Layer4Hybrid::analyzeColorTemperature(const cv::Mat& src) {
    if (src.empty()) return 0.0f;
    cv::resize(src, tempSmall, cv::Size(32, 32));
    cv::Scalar meanColor = cv::mean(tempSmall);
    float b = meanColor.val[0];
    float g = meanColor.val[1];
    float r = meanColor.val[2];
    float tempScore = 0.0f;
    
    if (r > g && g > b) {
        float rg_ratio = r / (g + 1e-6);
        float gb_ratio = g / (b + 1e-6);
        
        if (rg_ratio >= 1.05 && rg_ratio <= 1.45 && 
            gb_ratio >= 1.10 && gb_ratio <= 1.65) {
            tempScore += 0.15f;
        } 
        else if (rg_ratio >= 0.98 && rg_ratio <= 1.55 && 
                 gb_ratio >= 1.00 && gb_ratio <= 1.80) {
            tempScore += 0.05f;
        }
        else {
            tempScore -= 0.10f; 
        }
    } 
    else if (r > b && g > b) {
        tempScore += 0.0f; 
    }
    else {
        tempScore -= 0.20f; 
    }
    
    float avgBrightness = (r + g + b) / 3.0f;
    if (avgBrightness < 20 || avgBrightness > 235) {
        tempScore -= 0.15f;
    } else if (avgBrightness >= 40 && avgBrightness <= 200) {
        tempScore += 0.05f; 
    }
    
    return tempScore;
}

float Layer4Hybrid::detectScreenEdges(const cv::Mat& src) {
    if (src.empty() || src.cols < 60 || src.rows < 60) return 0.0f;
    cv::resize(src, edgeBuffer, cv::Size(120, 120)); 
    if (edgeBuffer.channels() == 3) 
        cv::cvtColor(edgeBuffer, grayBuffer, cv::COLOR_BGR2GRAY);
    else 
        edgeBuffer.copyTo(grayBuffer);
    cv::Canny(grayBuffer, edgeMap, 50, 150);
    
    int borderSize = 5;
    cv::Rect topBorder(0, 0, edgeMap.cols, borderSize);
    cv::Rect bottomBorder(0, edgeMap.rows - borderSize, edgeMap.cols, borderSize);
    cv::Rect leftBorder(0, 0, borderSize, edgeMap.rows);
    cv::Rect rightBorder(edgeMap.cols - borderSize, 0, borderSize, edgeMap.rows);
    
    int topEdges = cv::countNonZero(edgeMap(topBorder));
    int bottomEdges = cv::countNonZero(edgeMap(bottomBorder));
    int leftEdges = cv::countNonZero(edgeMap(leftBorder));
    int rightEdges = cv::countNonZero(edgeMap(rightBorder));
    int totalBorderEdges = topEdges + bottomEdges + leftEdges + rightEdges;
    int maxExpected = borderSize * edgeMap.cols * 2 + borderSize * edgeMap.rows * 2;
    float edgeRatio = (float)totalBorderEdges / maxExpected;
    
    if (edgeRatio > 0.20f) {
        return -0.25f;
    } else if (edgeRatio > 0.15f) {
        return -0.12f; 
    }
    
    return 0.0f;
}

float Layer4Hybrid::analyzeQuality(const cv::Mat& frame, const cv::Rect& faceBox) {
    cv::Rect safeBox = faceBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safeBox.area() <= 100) return -0.5f;  
    cv::Mat faceRoi = frame(safeBox);
    
    // 1. Skin consistency
    float skinScore = 0.0f;
    checkSkinConsistency(faceRoi, skinScore);   
    // 2. Texture gradient
    float textureScore = analyzeTextureGradient(faceRoi);
    // 3. Color temperature
    float tempScore = analyzeColorTemperature(faceRoi);
    // 4. Screen edge detection
    float edgeScore = detectScreenEdges(faceRoi);
    // 5. Moire + High frequency
    int centerSize = std::min(safeBox.width, safeBox.height) / 2;
    int cx = safeBox.x + safeBox.width / 2;
    int cy = safeBox.y + safeBox.height / 2;
    cv::Rect moireRect(cx - centerSize/2, cy - centerSize/2, centerSize, centerSize);
    moireRect = moireRect & cv::Rect(0, 0, frame.cols, frame.rows);
    
    float moireScore = 0.0f;
    if (moireRect.width >= 32 && moireRect.height >= 32) {
        cv::Mat moireRoi = frame(moireRect);
        double freqHigh = calculateHighFrequency(moireRoi);
        moireScore = detectMoirePattern(moireRoi);
        
        if (freqHigh > 17.0) {
            moireScore -= 0.40f; 
        } else if (freqHigh > 14.5) {
            moireScore -= 0.20f;
        } else if (freqHigh < 5.0) {
            moireScore -= 0.20f;
        } else if (freqHigh >= 7.0 && freqHigh <= 13.0) {
            moireScore += 0.20f; 
        }
    }
    
    float totalAdjustment = skinScore * 1.0f +      
                           textureScore * 0.8f +   
                           tempScore * 0.6f +      
                           edgeScore * 0.7f +      
                           moireScore * 0.9f;   
    
    return std::max(-0.60f, std::min(0.50f, totalAdjustment)); 
}