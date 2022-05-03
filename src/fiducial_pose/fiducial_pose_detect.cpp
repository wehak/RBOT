#include "fiducial_pose_detect.h"
#include "utility.h"

using namespace std;
using namespace cv;

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
    printMatrix(intrinsics);

    // create homogeneous transformation matrices linking every tag to the center of the poster
    posterMeasurements = getPosterMeasurements(PosterMeasurementsFilepath);

    // get homogeneous transformation matrices linking every model base to the center
    T_om = getModelPositions(modelPositionFilepath);
    modelNames = getModelNames(modelPositionFilepath);
    
    // const int nPosterTags = posterMeasurements.size();
    int i=0;
    for (auto& element : posterMeasurements) {
        T_to[i] = getTagToCenterT(element);
        i++;
    }

    // aruco parameters
    dict = dictionary;
}

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

fiducialPose fiducialPoseDetector::getCleanModelPose(Mat frame, string modelName, float outlier_coef=2, int min_tags=1) {
    // declare variables
    fiducialPose pose;
    vector<int> ids;
    vector< vector<Point2f> > corners;
    map<int, Matx44f> poses;

    // detect pose
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
        }

        // declare variables necessary to perform outlier detection
        vector<Vec3d> rvecs;
        vector<Vec3d> tvecs;
        vector<Vec3d> inlier_rvecs;
        vector<Vec3d> inlier_tvecs;
        vector<Vec3d> outlier_rvecs;
        vector<Vec3d> outlier_tvecs;
        Vec3d rvec_avg;
        Vec3d tvec_avg;
        Vec3d rvec_stdDev;
        Vec3d tvec_stdDev;
        Vec3d inlier_avg_rvec;
        Vec3d inlier_avg_tvec;

        // create vector with rotation and translation vectors
        for (const auto& pose : poses) {
            // append pose to vector
            rvecs.push_back(getRvecFromT(pose.second)); 
            tvecs.push_back(getTvecFromT(pose.second)); 

            // add value to calculate average
            rvec_avg += rvecs.back();
            tvec_avg += tvecs.back();
        }

        // calculate average
        rvec_avg = rvec_avg / float(rvecs.size());
        tvec_avg = tvec_avg / float(tvecs.size());

        // calculate standard deviation
        for (int i=0 ; i<rvecs.size() ; i++) {
            rvec_stdDev[0] += pow(rvecs[i][0] - rvec_avg[0], 2);
            rvec_stdDev[1] += pow(rvecs[i][1] - rvec_avg[1], 2);
            rvec_stdDev[2] += pow(rvecs[i][2] - rvec_avg[2], 2);

            tvec_stdDev[0] += pow(tvecs[i][0] - tvec_avg[0], 2);
            tvec_stdDev[1] += pow(tvecs[i][1] - tvec_avg[1], 2);
            tvec_stdDev[2] += pow(tvecs[i][2] - tvec_avg[2], 2);
        }

        rvec_stdDev[0] = sqrt(rvec_stdDev[0] / rvecs.size());
        rvec_stdDev[1] = sqrt(rvec_stdDev[1] / rvecs.size());
        rvec_stdDev[2] = sqrt(rvec_stdDev[2] / rvecs.size());
        tvec_stdDev[0] = sqrt(tvec_stdDev[0] / tvecs.size());
        tvec_stdDev[1] = sqrt(tvec_stdDev[1] / tvecs.size());
        tvec_stdDev[2] = sqrt(tvec_stdDev[2] / tvecs.size());

        // remove outliers that deviate from the average by x standard deviations
        for (int i=0 ; i<rvecs.size() ; i++) {
            bool outlier = false;
            for (int j=0 ; j<3 ; j++) {
                if ((abs(rvecs[i][j] - rvec_avg[j]) > rvec_stdDev[j] * outlier_coef) || (abs(tvecs[i][j] - tvec_avg[j]) > tvec_stdDev[j] * outlier_coef)) {
                    outlier = true;
                }
            }

            // sort outliers and inliers
            if (!outlier) {
                inlier_rvecs.push_back(rvecs[i]);
                inlier_tvecs.push_back(tvecs[i]);
            }
            else {
                outlier_rvecs.push_back(rvecs[i]);
                outlier_tvecs.push_back(tvecs[i]);
            }

        }

        // average based on inliers
        for (int i=0 ; i<inlier_rvecs.size() ; i++) {
            inlier_avg_rvec += inlier_rvecs[i];
            inlier_avg_tvec += inlier_tvecs[i];
        }

        inlier_avg_rvec = inlier_avg_rvec / float(inlier_rvecs.size());
        inlier_avg_tvec = inlier_avg_tvec / float(inlier_tvecs.size());
        
        // return cleaned and averaged T
        // Matx44f T_cm = getHomogeneousTransformationMatrix(inlier_avg_rvec, inlier_avg_tvec);
        pose.T = getHomogeneousTransformationMatrix(inlier_avg_rvec, inlier_avg_tvec);
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


// map<int, Matx44d> fiducialPoseDetector::getCenterPoses(Mat frame) {
//     map<int, Matx44d> T_cm_map;

//     // create a copy of the frame
//     Mat frameCopy;
//     frame.copyTo(frameCopy);

//     // detect markers
//     vector<int> ids;
//     vector< vector<Point2f> > corners;
//     aruco::detectMarkers(frame, dict, corners, ids);

//     // if any markers are detected
//     if (ids.size() > 0) {

//         // estimate pose relative to camera
//         vector<Vec3d> rvecs, tvecs;
//         aruco::estimatePoseSingleMarkers(
//             corners,
//             ARUCO_MARKER_SIZE,
//             intrinsics,
//             distortion,
//             rvecs,
//             tvecs
//         );

//         // and for each markers
//         map<int, Matx44d> T_co_map;
//         for (int i=0 ; i<ids.size() ; i++) {

//             // transform pose from fiducial to poster center
//             Matx44d T_ct = getHomogeneousTransformationMatrix(rvecs[i], tvecs[i]);
//             Matx44d T_co = matrixDot(T_ct, T_to[ids[i]]);
//             T_co_map[ids[i]] = (T_co);
//         }

//         // average the T_co from all tags
//         Matx44d avg_T_co = averageMatrix(T_co_map);

//         // for each model
//         for (auto & model : modelNames) {
//             // Matx44d T_cm = matrixDot(avg_T_co, T_om[model]);
//             // T_cm_map.push_back(T_cm);
//             if (model == "d-handle") {
//                 // cout << model << endl;
//                 // cout << matrixDot(avg_T_co, T_om[model]) << endl;

//             }

//             // save the T_cm
//             T_cm_map.push_back( matrixDot(avg_T_co, T_om[model]) );
//         }
//         // cout << endl;
//     }
//     return T_cm_map;
// }

// average the elements in a vector of matrices
// Matx44d fiducialPoseDetector::averageMatrix(vector<Matx44d> Ts) {
//     Matx44d avgT;
//     int n = Ts.size();

//     // sum element by element
//     for (auto & T : Ts) {
//         for (int row=0 ; row<T.rows ; row++) {
//             for(int col=0 ; col<T.cols ; col++) {
//                 avgT(row, col) += T(row, col);
//             }
//         }
//     }

//     // divide by n
//     for (int row=0 ; row<avgT.rows ; row++) {
//         for(int col=0 ; col<avgT.cols ; col++) {
//             avgT(row, col) /= n;
//         }
//     }

//     return avgT;
// }

string fiducialPoseDetector::printModelNames() {
    string models = "";
    bool first = true;
    for (auto const & x : T_om) {
        // if (!first) cout << ", ";
        // if (!first) strcat(models, ", " << x.first);
        if (first) {
            first = false;
            models = x.first;
        }
        else {
            // strcat(models, strcat(", ", x.first) );
            models = models + ", " + x.first;
        }
        // cout << x.first;
    }

    return models;
}