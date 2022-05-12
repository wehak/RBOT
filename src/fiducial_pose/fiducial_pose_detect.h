#ifndef FIDUCIAL_POSE_DETECT_H
#define FIDUCIAL_POSE_DETECT_H

#include <iostream>
#include <map>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>

// #include "utility.h"

using namespace std;
using namespace cv;

struct fiducialPose {
    int n_fiducials;
    Matx44f T; 
};


/**
 * @brief   Extracts fiducial poses from a given image and transforms them into a 
 *          new pose based on poster measurements. 
 */
class fiducialPoseDetector {
    public:
        /**
         * @brief Construct a new fiducial Pose Detector object.
         * 
         * @param CameraMetricsFilepath  Path to JSON file with the camera intrics.
         * @param PosterMeasurementsFilepath  Path to JSON file with transforms from each fiducial to the poster center.
         * @param modelPositionFilepath  Path to JSON file with transforms from the poster center to model origin.
         * @param dict  OpenCV fiducial dictionary to be used.
         * @param markerSize  Expected size of the fiducial markers [m].
         */
        fiducialPoseDetector(
            string CameraMetricsFilepath,
            string PosterMeasurementsFilepath,
            string modelPositionFilepath,
            Ptr<aruco::Dictionary> dict,
            float markerSize
            );

        /**
         * @brief Get the pose and ID of every fiducial withoth any transformations.
         * 
         * @param frame Image frame.
         * @return map<int, Matx44f> 
         */        
        map<int, Matx44f>   getFiducialPoses(Mat frame);

        /**
         * @brief Get the transformed pose and ID of every fiducial.
         * 
         * @param frame Image frame.
         * @param modelName Name of the model poses shall be transformed into.
         * @return map<int, Matx44f> 
         */
        map<int, Matx44f>   getRawModelPoses(Mat frame, string modelName);

        /**
         * @brief Get a cleaned and averaged pose of every fiducial.
         * 
         * @param frame Image frame.
         * @param modelName Name of the model poses shall be transformed into.
         * @param tvec_outlier_coef Outlier threshold in standard deviations from the mean.
         * @param min_tags Minimum number of tags necessary to produce a pose.
         * @return fiducialPose 
         */
        fiducialPose        getCleanModelPose(Mat frame, string modelName, float tvec_outlier_coef, int min_tags);
        
        /**
         * @brief Takes a vector of rotations represented as quaternions and returns an average. 
         * 
         * @param Q Vector of quaternions.
         * @return Eigen::Vector4d 
         */
        Eigen::Vector4d     averageQuaternions(vector<Eigen::Vector4d> Q);

        /**
         * @brief Translates a rotation vector into a quaternion.
         * 
         * @param rvec 
         * @return Eigen::Quaterniond 
         */
        Eigen::Quaterniond  rvec2quat(Vec3d rvec);

        /**
         * @brief Translates a quaternion into a Eigen vector.
         * 
         * @param q quaternion
         * @return Eigen::Vector4d 
         */
        Eigen::Vector4d     quat2vec(Eigen::Quaterniond q);

        /**
         * @brief Translates a quaternion into a OpenCV rotation vector.
         * 
         * @param q 
         * @return Vec3d 
         */
        Vec3d               quat2rvec(Eigen::Vector4d q);

        /**
         * @brief Prints names of all models found in "modelPositionFilepath".
         * 
         * @return string 
         */
        string              printModelNames();



        // arguments
        float ARUCO_MARKER_SIZE;
        // string CameraMetricsFilepath;
        // string PosterMeasurementsFilepath;
        // string modelPositionFilepath;
        Ptr<aruco::Dictionary> dict;
        // Ptr<aruco::DetectorParameters> parameters;

        // misc variables
        Matx33f intrinsics;
        Mat distortion;
        vector< array<float, 2> > posterMeasurements;
        map<string, Matx44d> T_om;
        vector<string> modelNames;
        array< Matx44d, 12 > T_to;
};


#endif