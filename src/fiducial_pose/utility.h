#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <vector> 
#include <map>

#include <opencv2/core/matx.hpp>

std::vector< std::array<float, 2> > getPosterMeasurements(std::string filepath);


cv::Matx44d matrixDot(cv::Matx44d A, cv::Matx44d B);
cv::Matx44d getTagToCenterT(std::array<float, 2> placement);
cv::Matx33f getCameraIntrinsics(std::string filepath);
cv::Mat getCameraDistortion(std::string filepath);
cv::Matx44d getHomogeneousTransformationMatrix(cv::Vec3d rvec, cv::Vec3d tvec);
// cv::Matx44d getHomogeneousTransformationMatrix(cv::Mat R, cv::Vec3d tvec);

std::map<std::string, cv::Matx44d> getModelPositions(std::string filepath);
std::vector<std::string> getModelNames(std::string filepath);

cv::Vec3d avgTransformedVector(std::vector< cv::Vec3d > rvecs);
cv::Vec3d getRvecFromT(cv::Matx44d T);
cv::Vec3d getTvecFromT(cv::Matx44d T);

// void disassembleTransformationMatrix(cv::Matx44d T, cv::Vec3d rvec, cv::Vec3d tvec);
std::string printMatrixSingleLine(cv::Matx44f T);
void printMatrix(cv::Mat T);
void printMatrix(cv::Matx33f T);
void printMatrix(cv::Matx44d T);

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

#endif