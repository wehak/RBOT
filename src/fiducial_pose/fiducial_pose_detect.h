#ifndef FIDUCIAL_POSE_DETECT_H
#define FIDUCIAL_POSE_DETECT_H

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

// #include "utility.h"

using namespace std;
using namespace cv;

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
        vector<Matx44d> getPoses(Mat frame);
        Matx44f         getPoseModel(Mat frame, string modelName);
        Matx44d         averageMatrix(vector<Matx44d>);
        string          printModelNames();

};

#endif