#include "fiducial_pose_detect.h"
#include "utility.h"

using namespace std;
using namespace cv;

// class fiducialPoseDetector {
//     public:
//         fiducialPoseDetector(
//             string CameraMetricsFilepath,
//             string PosterMeasurementsFilepath,
//             string modelPositionFilepath,
//             Ptr<aruco::Dictionary> dict,
//             float markerSize
//             );

//         // arguments
//         float ARUCO_MARKER_SIZE;
//         string CameraMetricsFilepath;
//         string PosterMeasurementsFilepath;
//         string modelPositionFilepath;
//         Ptr<aruco::Dictionary> dict;

//         // misc variables
//         Matx33f intrinsics;
//         Mat distortion;
//         vector< array<float, 2> > posterMeasurements;
//         map<string, Matx44d> T_om;
//         vector<string> modelNames;
//         array< Matx44d, 12 > T_to;

//         // methods
//         vector<Matx44d> getPoses(Mat frame);
//         Matx44d averageMatrix(vector<Matx44d>);

// };

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

Matx44f fiducialPoseDetector::getPoseModel(Mat frame, string modelName) {
    // detect markers
    vector<int> ids;
    vector< vector<Point2f> > corners;
    aruco::detectMarkers(frame, dict, corners, ids);

    // if any markers are detected
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
            T_co_vector.push_back(T_co);
        }

        // average the T_co from all tags
        Matx44d avg_T_co = averageMatrix(T_co_vector);

        // return the dot product of the average with the transformation to the selected model
        return matrixDot(avg_T_co, T_om[modelName]);
    }

    else {
        cout << "No tags detected" << endl;
        return Matx44f::eye();
    }

}

vector<Matx44d> fiducialPoseDetector::getPoses(Mat frame) {
    vector<Matx44d> T_cm_vector;

    // create a copy of the frame
    Mat frameCopy;
    frame.copyTo(frameCopy);

    // detect markers
    vector<int> ids;
    vector< vector<Point2f> > corners;
    aruco::detectMarkers(frame, dict, corners, ids);

    // if any markers are detected
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
            T_co_vector.push_back(T_co);
        }

        // average the T_co from all tags
        Matx44d avg_T_co = averageMatrix(T_co_vector);

        // for each model
        for (auto & model : modelNames) {
            // Matx44d T_cm = matrixDot(avg_T_co, T_om[model]);
            // T_cm_vector.push_back(T_cm);
            if (model == "d-handle") {
                // cout << model << endl;
                // cout << matrixDot(avg_T_co, T_om[model]) << endl;

            }

            // save the T_cm
            T_cm_vector.push_back( matrixDot(avg_T_co, T_om[model]) );
        }
        // cout << endl;
    }
    return T_cm_vector;
}

// average the elements in a vector of matrices
Matx44d fiducialPoseDetector::averageMatrix(vector<Matx44d> Ts) {
    Matx44d avgT;
    int n = Ts.size();

    // sum element by element
    for (auto & T : Ts) {
        for (int row=0 ; row<T.rows ; row++) {
            for(int col=0 ; col<T.cols ; col++) {
                avgT(row, col) += T(row, col);
            }
        }
    }

    // divide by n
    for (int row=0 ; row<avgT.rows ; row++) {
        for(int col=0 ; col<avgT.cols ; col++) {
            avgT(row, col) /= n;
        }
    }

    return avgT;
}

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