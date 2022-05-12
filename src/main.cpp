/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include <time.h>
#include <bits/stdc++.h>
#include <getopt.h>

// log FPS
#include <chrono>
#include <string>
#include <cmath>
#include <fstream>
#include <map>

#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "object3d.h"
#include "pose_estimator6d.h"

#include "fiducial_pose/fiducial_pose_detect.h"
#include "fiducial_pose/utility.h"
#include "fiducial_pose/json.hpp"

using namespace std;
using namespace cv;
using json = nlohmann::json;

bool posesCorrespond(Matx44f A, Matx44f B, map<string, float> limit) {
    Vec3d A_rvec = getRvecFromT(A) * (180 / 3.14); // radians to degrees
    Vec3d B_rvec = getRvecFromT(B) * (180 / 3.14);

    Vec3d A_tvec = getTvecFromT(A);
    Vec3d B_tvec = getTvecFromT(B);
    
    // if difference is within limits, return true
    if (
        abs(A_rvec[0] - B_rvec[0]) < limit["RVEC_X_MAX"] &&
        abs(A_rvec[1] - B_rvec[1]) < limit["RVEC_Y_MAX"] &&
        abs(A_rvec[2] - B_rvec[2]) < limit["RVEC_Z_MAX"] &&

        abs(A_tvec[0] - B_tvec[0]) < limit["TVEC_X_MAX"] &&
        abs(A_tvec[1] - B_tvec[1]) < limit["TVEC_Y_MAX"] &&
        abs(A_tvec[2] - B_tvec[2]) < limit["TVEC_Z_MAX"]
    ) {
        return true;
    }
    // else return false
    else {
        return false;
    }    
}

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.0, 1.0));
    //colors.push_back(Point3f(0.2, 0.3, 1.0));
    RenderingEngine::Instance()->renderShaded(vector<Model*>(objects.begin(), objects.end()), GL_FILL, colors, true);
    
    // download the rendering to the CPU
    Mat rendering = RenderingEngine::Instance()->downloadFrame(RenderingEngine::RGB);
    
    // download the depth buffer to the CPU
    Mat depth = RenderingEngine::Instance()->downloadFrame(RenderingEngine::DEPTH);
    
    // compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
    Mat result = frame.clone();
    for(int y = 0; y < frame.rows; y++)
    {
        for(int x = 0; x < frame.cols; x++)
        {
            Vec3b color = rendering.at<Vec3b>(y,x);
            if(depth.at<float>(y,x) != 0.0f)
            {
                result.at<Vec3b>(y,x)[0] = (color[2] + frame.at<Vec3b>(y,x)[0] * 2) / 3;
                result.at<Vec3b>(y,x)[1] = (color[1] + frame.at<Vec3b>(y,x)[1] * 2) / 3;
                result.at<Vec3b>(y,x)[2] = (color[0] + frame.at<Vec3b>(y,x)[2] * 2) / 3;
            }
        }
    }
    return result;
}

cv::Mat drawBinaryMask(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    RenderingEngine::Instance()->renderSilhouette(
        objects[0], 
        GL_FILL, 
        false,
        1.0,
        1.0,
        1.0,
        true
        );
    
    // download the rendering to the CPU
    Mat mask = RenderingEngine::Instance()->downloadFrame(RenderingEngine::MASK);    
    return mask;
}



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // read arguments      
    string configFilePath;
    if (argc == 2) {
        configFilePath = argv[1];
    }
    else {
        cout << "Error: No configuration file specified." << endl;
        abort();
    }

    // read the config file
    ifstream configFile(configFilePath);
    json j;
    configFile >> j;

    // read each parameter
    const string sessionName = j["sessionName"];
    const string selectedModel = j["selectedModel"];
    const string outputFolder = j["outputFolder"];
    const string inputVideoPath = j["inputVideoPath"];

    const string PosterMeasurementsFilepath = j["PosterMeasurementsFilepath"];
    const string modelPositionFilepath = j["modelPositionFilepath"];
    const string CameraMetricsFilepath = j["CameraMetricsFilepath"];

    const string outputVideoPath = j["outputVideoPath"];

    const int MIN_FIDUCIALS = j["MIN_FIDUCIALS"];
    const float TVEC_COEF = j["TVEC_COEF"];
    const int reInitializePoseLimit = j["reInitializePoseLimit"];

    map<string, float> limit;
    limit["RVEC_X_MAX"] = j["RVEC_X_MAX"];
    limit["RVEC_Y_MAX"] = j["RVEC_Y_MAX"];
    limit["RVEC_Z_MAX"] = j["RVEC_Z_MAX"];

    limit["TVEC_X_MAX"] = j["TVEC_X_MAX"];
    limit["TVEC_Y_MAX"] = j["TVEC_Y_MAX"];
    limit["TVEC_Z_MAX"] = j["TVEC_Z_MAX"];


    // set some other parameters 
    const bool record = false;
    const bool undistortFrame = false;
    // const int reInitializePoseLimit = 10;

    // fiducials
    float fiducialTagSize = 0.14; // fiducial size [m]
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_16h5); // fiducial dictionary


    /************
    *  PROGRAM  *
    ************/

    // read the camera properties
    Matx33f intrinsics = getCameraIntrinsics(CameraMetricsFilepath);
    Mat distortion = getCameraDistortion(CameraMetricsFilepath);
    Matx14f distCoeffs;
    for (int i=0 ; i<4 ; i++) {
        distCoeffs(i) = distortion.at<float>(i, 0);
    }

    // measure FPS
    const int nMeanSamples = 5;
    vector<std::chrono::duration<double>> t_iteration;

    // camera image size
    // int width = 1920;
    // int height = 1080;
    int width = 1280;
    int height = 720;
    
    
    // near and far plane of the OpenGL view frustum
    float zNear = 10.0;
    float zFar = 10000.0;
    
    // distances for the pose detection template generation
    vector<float> distances = {200.0f, 400.0f, 800.0f};

    // create fiducial pose estimator
    fiducialPoseDetector fiducial_detector(
        CameraMetricsFilepath,
        PosterMeasurementsFilepath,
        modelPositionFilepath,
        dictionary,
        fiducialTagSize
    );

    // load 3D objects
    vector<Object3D*> objects;
    string selectedModelPath = "data/" + selectedModel + ".obj";
    cout << "Reading: " << selectedModelPath << endl;
    objects.push_back(new Object3D(selectedModelPath, 0, 0, 1000, 90, 0, 0, 1.0, 0.55f, distances));
    
    // create the RBOT pose estimator
    PoseEstimator6D* poseEstimator = new PoseEstimator6D(
        width, height,
        zNear, zFar, 
        intrinsics, 
        distCoeffs, 
        objects
        );
    
    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);
    
    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();
    
    int timeout = 0;
    bool showHelp = true;

    // initialize camera    
    Mat frame;
    VideoCapture cap(inputVideoPath);
    VideoWriter outputVideo(outputVideoPath, VideoWriter::fourcc('M','J','P','G'), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    
    // check if successful 
    if (!cap.isOpened()) {
        cout << "Error: Unable to open video file\n";
        return -1;
    }

    // record measurements
    vector< vector<string> > measurements;

    // create output data folder
    create_directory(outputFolder);
    create_directory(outputFolder + "/" + sessionName);
    create_directory(outputFolder + "/" + sessionName + "/" + selectedModel);
    create_directory(outputFolder + "/" + sessionName + "/" + selectedModel + "/mask");
    create_directory(outputFolder + "/" + sessionName + "/" + selectedModel + "/img");
    // create_directory(outputFolder + "/" + sessionName + "/result"); // DELETE

    // per frame loop
    int frame_n = 0;
    int missed_frames = 0;
    while(true)
    {
        // start timer
        auto tStart = std::chrono::high_resolution_clock::now();

        // obtain an input image
        cap.read(frame);
        if (frame.empty()) {
            cout << "Error: Blank frame grabbed\n";
            break;
        }

        // get model poses from RBOT
        poseEstimator->estimatePoses(frame, undistortFrame, true);
        
        // get model pose from fiducial
        fiducialPose pose = fiducial_detector.getCleanModelPose(frame, selectedModel, TVEC_COEF, MIN_FIDUCIALS);


        // DEBUG
        // cout << printMatrixSingleLine(objects[0]->getPose()) << endl;    
    
        // render the models with the resulting pose estimates ontop of the input image
        Mat result = drawResultOverlay(objects, frame);


        // record frame and mask
        Mat mask = drawBinaryMask(objects, frame);

        // normalize RBOT tvec (from mm to meters)
        Matx44f T_rbot = objects[0]->getPose();
        for (int i=0 ; i<3 ; i++) T_rbot(i, 3) = T_rbot(i, 3) / float(1000);

        // DEBUG
        // cout << printMatrixSingleLine(T_rbot) << endl; 

        // draw fiducial pose ontop of the input image
        if (pose.n_fiducials >= MIN_FIDUCIALS) {
            aruco::drawAxis(
                result, 
                intrinsics,
                distortion,
                getRvecFromT(pose.T),
                getTvecFromT(pose.T),
                0.10
                );
            
            // imwrite(outputFolder + "/" + sessionName + "/result/" + to_string(frame_n) + ".png", result); // DELETE 

            // check if poses correspond
            if (posesCorrespond(
                    // objects[0]->getPose() / float(1000), // RBOT pose, normalized to meters
                    T_rbot, // RBOT pose
                    pose.T, // fiducial pose
                    limit // correspondence limits
                    )
                ) {
                missed_frames = 0;
                imwrite(outputFolder + "/" + sessionName + "/" + selectedModel + "/mask/" + to_string(frame_n) + ".png", mask);
                imwrite(outputFolder + "/" + sessionName + "/" + selectedModel + "/img/" + to_string(frame_n) + ".png", frame);
            }

            // if not
            else {
                missed_frames++;
            }

            // reset RBOT is if there are successive correspondence failures
            if (missed_frames >= reInitializePoseLimit) {
                missed_frames = 0;
                cout << "#" + to_string(frame_n) + ": No correspondance for " + to_string(missed_frames) + " frames. Attempting to reset RBOT." << endl;

                poseEstimator->setModelInitialPose(0, pose.T);
                poseEstimator->reset();
                poseEstimator->toggleTracking(frame, 0, undistortFrame);
                poseEstimator->estimatePoses(frame, undistortFrame, false);
            }
        }
        else {
            if (missed_frames >= reInitializePoseLimit) {
                cout << "No correspondance for " + to_string(missed_frames) + " frames. No track on fiducials."  << endl;
            }
        }

        if(showHelp)
        {
            putText(result, "Press 's' to start, 'p' to pause", Point(width / 3, height / 2 - 20), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 0, 255), 1);
            putText(result, "or 'c' to quit", Point(width / 3, height / 2 + 20), FONT_HERSHEY_DUPLEX, 1.0, Scalar(255, 0, 255), 1);
        }

        // measure FPS
        // auto tEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - tStart;
        t_iteration.emplace_back(diff);
        std::chrono::duration<double> sum = std::chrono::duration<double>::zero();
        vector<std::chrono::duration<double>> t_recent (t_iteration.end() - nMeanSamples, t_iteration.end());
        for (auto n : t_recent) {
            sum += n;
        }
        auto FPS = 1.0 / ((1.0 / nMeanSamples) * sum.count());
        putText(result, to_string((int)round(FPS)), Point(10, 30), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 1);

        // record poses
        measurements.push_back(vector<string>{to_string(diff.count()), to_string(pose.n_fiducials), printMatrixSingleLine(pose.T), printMatrixSingleLine(objects[0]->getPose())});

        // show frame
        imshow("result", result);
        int key = waitKey(timeout);
        
        // start/stop tracking the first object
        if(key == (int)'s') {
            poseEstimator->setModelInitialPose(0, pose.T);
            poseEstimator->reset();

            poseEstimator->toggleTracking(frame, 0, undistortFrame);
            poseEstimator->estimatePoses(frame, undistortFrame, false);

            timeout = 1;
            showHelp = false;
        }
        // if(key == (int)'2') // the same for a second object
        // {
        //     //poseEstimator->toggleTracking(frame, 1, undistortFrame);
        //     //poseEstimator->estimatePoses(frame, false, undistortFrame);
        // }

        // // reset the system to the initial state
        // if(key == (int)'r') poseEstimator->reset();
        // // stop the demo
        if(key == (int)'c') break;
        // // set pose from fiducal pose detector
        // if(key == (int)'f') {
        //     poseEstimator->setModelPose(0, pose.T);
        //     // poseEstimator->reset();
        //     // timeout = 0;
        // }
        // // pause
        if(key == (int)'p') timeout = 0;

        // recording
        if (record == true) {
            outputVideo.write(result);
        }

        frame_n++;
    }
    
    // deactivate the offscreen rendering OpenGL context
    RenderingEngine::Instance()->doneCurrent();
    
    // clean up
    RenderingEngine::Instance()->destroy();
    
    for(int i = 0; i < objects.size(); i++)
    {
        delete objects[i];
    }
    objects.clear();
    
    delete poseEstimator;

    // write measurements data to file
    ofstream log;
    string output_filename = outputFolder + "/" + sessionName + "/" + selectedModel + "/measurement_log_" + getFilename(inputVideoPath) + "_" + selectedModel + ".txt";
    log.open(output_filename);
    if (log.is_open()) {
        for (auto& line : measurements) {
            for (auto& col : line) {
                log << col << ";";
            }
            log << "\n";
        }
    }
    log.close();

    // openCV cleanup
    cap.release();
    if (record == true) {
        outputVideo.release();
    }
    destroyAllWindows();

    return 0;
}
