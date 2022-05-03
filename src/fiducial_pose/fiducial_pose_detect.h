#ifndef FIDUCIAL_POSE_DETECT_H
#define FIDUCIAL_POSE_DETECT_H

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

// #include "utility.h"

using namespace std;
using namespace cv;

struct fiducialPose {
    int n_fiducials;
    Matx44f T; 
};

class fiducialPoseDetector {
    public:
        fiducialPoseDetector(
            string CameraMetricsFilepath,
            string PosterMeasurementsFilepath,
            string modelPositionFilepath,
            Ptr<aruco::Dictionary> dict,
            float markerSize
            );

        // arguments
        float ARUCO_MARKER_SIZE;
        string CameraMetricsFilepath;
        string PosterMeasurementsFilepath;
        string modelPositionFilepath;
        Ptr<aruco::Dictionary> dict;

        // misc variables
        Matx33f intrinsics;
        Mat distortion;
        vector< array<float, 2> > posterMeasurements;
        map<string, Matx44d> T_om;
        vector<string> modelNames;
        array< Matx44d, 12 > T_to;

        // methods
        map<int, Matx44f>   getFiducialPoses(Mat frame);
        map<int, Matx44f>   getRawModelPoses(Mat frame, string modelName);
        fiducialPose        getCleanModelPose(Mat frame, string modelName, float outlier_coef, int min_tags);
        // pair<int, Matx44f>  getCleanModelPose(Mat frame, string modelName, float outlier_coef, int min_tags);
        // Matx44d             averageMatrix(vector<Matx44d>);
        string              printModelNames();

};


#endif