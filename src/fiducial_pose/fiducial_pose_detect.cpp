#include "fiducial_pose_detect.h"
#include "utility.h"

using namespace std;
using namespace cv;

// constructor
fiducialPoseDetector::fiducialPoseDetector(
    string CameraMetricsFilepath,
    string PosterMeasurementsFilepath,
    string modelPositionFilepath,
    Ptr<aruco::Dictionary> dictionary,
    float markerSize
    ) {
    ARUCO_MARKER_SIZE = markerSize; // [m]

    // read the camera properties
    intrinsics = getCameraIntrinsics(CameraMetricsFilepath);
    distortion = getCameraDistortion(CameraMetricsFilepath);

    // create homogeneous transformation matrices linking every tag to the center of the poster
    posterMeasurements = getPosterMeasurements(PosterMeasurementsFilepath);

    // get homogeneous transformation matrices linking every model base to the center
    T_om = getModelPositions(modelPositionFilepath);
    modelNames = getModelNames(modelPositionFilepath);
    
    // const int nPosterTags = posterMeasurements.size();
    int i=0;
    for (auto& element : posterMeasurements) {
        T_to[i] = getTagToCenterT(element);
        // printMatrix(T_to[i]);
        i++;
    }

    // aruco parameters
    dict = dictionary;
}


// get the pose of all fiducial tags
map<int, Matx44f> fiducialPoseDetector::getFiducialPoses(Mat frame) {
    // declare variables
    vector<int> ids;
    vector< vector<Point2f> > corners;
    map<int, Matx44f> poses;

    // detect markers
    aruco::detectMarkers(frame, dict, corners, ids);

    // if any markers are detected, perform transform and return map
    if (ids.size() > 0) {

        // estimate pose relative to camera
        vector<Vec3d> rvecs, tvecs;
        aruco::estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_SIZE,
            intrinsics,
            distortion,
            rvecs,
            tvecs
        );

        // and for each markers
        vector<Matx44d> T_co_vector;
        for (int i=0 ; i<ids.size() ; i++) {

            // convert Rodriguez vector to homogeneous transformation matrix
            Matx44d T_ct = getHomogeneousTransformationMatrix(rvecs[i], tvecs[i]);

            // save with the tag ID
            poses[ids[i]] = T_ct;
        }

        return poses;
    }

    // if not tags are detected, return empty map
    else {
        return poses;
    }
}


// return the poses of a model, transformed from each fiducial tag
map<int, Matx44f> fiducialPoseDetector::getRawModelPoses(Mat frame, string modelName) {
    // declare variables
    vector<int> ids;
    vector< vector<Point2f> > corners;
    map<int, Matx44f> poses;

    // detect markers
    aruco::detectMarkers(frame, dict, corners, ids);

    // if any markers are detected, perform transform and return map
    if (ids.size() > 0) {

        // estimate pose relative to camera
        vector<Vec3d> rvecs, tvecs;
        aruco::estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_SIZE,
            intrinsics,
            distortion,
            rvecs,
            tvecs
        );

        // and for each markers
        vector<Matx44d> T_co_vector;
        for (int i=0 ; i<ids.size() ; i++) {

            // transform pose from fiducial to poster center
            Matx44d T_ct = getHomogeneousTransformationMatrix(rvecs[i], tvecs[i]);
            Matx44d T_co = matrixDot(T_ct, T_to[ids[i]]);

            // transform pose for center to model and save with the tag ID
            poses[ids[i]] = matrixDot(T_co, T_om[modelName]);
        }

        return poses;
    }

    // if not tags are detected, return empty map
    else {
        return poses;
    }

}


// return the pose of a model, with outliers removed
fiducialPose fiducialPoseDetector::getCleanModelPose(Mat frame, string modelName, float tvec_outlier_coef=2, int min_tags=1) {
    // declare variables
    fiducialPose pose;
    map<int, Matx44f> poses;

    // detect pose
    vector<int> ids;
    vector< vector<Point2f> > corners;
    aruco::detectMarkers(frame, dict, corners, ids);

    // if any markers are detected
    if (ids.size() >= min_tags) {

        // estimate pose relative to camera
        vector<Vec3d> raw_rvecs, raw_tvecs;
        aruco::estimatePoseSingleMarkers(
            corners,
            ARUCO_MARKER_SIZE,
            intrinsics,
            distortion,
            raw_rvecs,
            raw_tvecs
        );

        // and for each markers
        for (int i=0 ; i<ids.size() ; i++) {

            // transform pose from fiducial to poster center
            Matx44d T_ct = getHomogeneousTransformationMatrix(raw_rvecs[i], raw_tvecs[i]);
            Matx44d T_co = matrixDot(T_ct, T_to[ids[i]]);

            // transform pose for center to model and save with the tag ID
            poses[ids[i]] = matrixDot(T_co, T_om[modelName]);


            // aruco::drawAxis(
            //     frame, 
            //     intrinsics,
            //     distortion,
            //     getRvecFromT(poses[ids[i]]),
            //     getTvecFromT(poses[ids[i]]),
            //     0.03
            //     );
        }

        // declare variables necessary to perform outlier detection
        vector<Vec3d> tvecs;
        vector<Vec3d> inlier_tvecs;
        vector<Vec3d> outlier_tvecs;
        Vec3d tvec_avg;
        Vec3d tvec_stdDev;
        Vec3d inlier_avg_tvec;

        // vector<Eigen::Quaterniond> quats;
        vector<Eigen::Vector4d> quats;
        vector<Eigen::Vector4d> inlier_quats;
        vector<Eigen::Vector4d> outlier_quats;
        // Eigen::Vector4d quat_stdDev;



        // create vector with rotation and translation vectors
        for (const auto& pose : poses) {
            // append pose to vector
            tvecs.push_back(getTvecFromT(pose.second)); 

            // add value to calculate average
            tvec_avg += tvecs.back();

            Eigen::Quaterniond q = rvec2quat( getRvecFromT(pose.second) );
            Eigen::Vector4d v = quat2vec(q);
            quats.push_back(v);
        }

        Eigen::Vector4d quat_avg(averageQuaternions(quats));

        // calculate average
        tvec_avg = tvec_avg / float(tvecs.size());

        // calculate standard deviation by squaring the difference from average
        for (int i=0 ; i<tvecs.size() ; i++) {
            tvec_stdDev[0] += pow(tvecs[i][0] - tvec_avg[0], 2);
            tvec_stdDev[1] += pow(tvecs[i][1] - tvec_avg[1], 2);
            tvec_stdDev[2] += pow(tvecs[i][2] - tvec_avg[2], 2);

            // quat_stdDev = (quats[i] - quat_avg).array().pow(2);
        }

        // and divide the sum by the number of samples, then square root
        tvec_stdDev[0] = sqrt(tvec_stdDev[0] / tvecs.size());
        tvec_stdDev[1] = sqrt(tvec_stdDev[1] / tvecs.size());
        tvec_stdDev[2] = sqrt(tvec_stdDev[2] / tvecs.size());

        // quat_stdDev = (quat_stdDev / quats.size()).array().sqrt();

        // remove outliers that deviate from the average by x standard deviations
        for (int i=0 ; i<tvecs.size() ; i++) {
            bool outlier = false;
            for (int j=0 ; j<3 ; j++) {
                // check tvec
                if (abs(tvecs[i][j] - tvec_avg[j]) > tvec_stdDev[j] * tvec_outlier_coef) {
                    outlier = true;
                }
            }

            // sort outliers and inliers
            if (!outlier) {
                inlier_tvecs.push_back(tvecs[i]);
                inlier_quats.push_back(quats[i]);
            }
            else {
                outlier_tvecs.push_back(tvecs[i]);
                outlier_quats.push_back(quats[i]);
                
                // aruco::drawAxis(
                //     frame, 
                //     intrinsics,
                //     distortion,
                //     quat2rvec(quats[i]),
                //     tvecs[i],
                //     0.03
                //     );
            }

        }

        // average based on inliers
        for (int i=0 ; i<inlier_tvecs.size() ; i++) {
            inlier_avg_tvec += inlier_tvecs[i];
        }

        inlier_avg_tvec = inlier_avg_tvec / float(inlier_tvecs.size());
        
        // return cleaned and averaged T
        pose.T = getHomogeneousTransformationMatrix(
            quat2rvec(averageQuaternions(inlier_quats)), 
            inlier_avg_tvec
            );
        pose.n_fiducials = ids.size();
        return pose;
        // return make_pair(ids.size(), T_cm);
    }

    // if no tags, return empty matrix
    else {
        pose.n_fiducials = ids.size();
        pose.T = Matx44f::zeros();
        return pose;
    }
}

// average quaternions, based on Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.
Eigen::Vector4d fiducialPoseDetector::averageQuaternions(vector<Eigen::Vector4d> Q) {

    // add the quaternion and its transpose to a 4x4 matrix
    Eigen::Matrix4d A;
    for (auto& q : Q) {
        A = q * q.transpose() + A;
    }

    // retun the eigenvector of A
    Eigen::EigenSolver<Eigen::Matrix4d> solver(A);
    return solver.eigenvectors().real().col(0);
}


// print the names of available models
string fiducialPoseDetector::printModelNames() {
    string models = "";
    bool first = true;
    for (auto const & x : T_om) {
        if (first) {
            first = false;
            models = x.first;
        }
        else {
            models = models + ", " + x.first;
        }
    }

    return models;
}


// convert a opencv rvec to eigen quaternion
Eigen::Quaterniond fiducialPoseDetector::rvec2quat(cv::Vec3d rvec) {
    
    // cast as opencv matrix and convert to rotation matrix
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    // convert rotation matrix to quat
    Eigen::Matrix3d R;
    cv::cv2eigen(R_cv, R);
    Eigen::Quaterniond q(R);

    return q;
}


// convert a eigen quaternion to a eigen vector of length 4
Eigen::Vector4d fiducialPoseDetector::quat2vec(Eigen::Quaterniond q) {
    Eigen::Vector4d w;
    w(0) = q.vec()[0];
    w(1) = q.vec()[1];
    w(2) = q.vec()[2];
    w(3) = q.w();
    return w;
}


// convert a eigen quaternion in vector form to opencv rvec
Vec3d fiducialPoseDetector::quat2rvec(Eigen::Vector4d v) {

    // convert from quaternion to rotation matrix
    Eigen::Quaterniond q(v);
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    Mat R_cv;
    eigen2cv(R, R_cv);

    // use rotation matrix to calculate rvec
    Vec3d rvec;
    Rodrigues(R_cv, rvec);
    return rvec;
}