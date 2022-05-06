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

#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "object3d.h"
#include "pose_estimator6d.h"

#include "fiducial_pose/fiducial_pose_detect.h"
#include "fiducial_pose/utility.h"

using namespace std;
using namespace cv;

cv::Mat drawResultOverlay(const vector<Object3D*>& objects, const cv::Mat& frame)
{
    // render the models with phong shading
    RenderingEngine::Instance()->setLevel(0);
    
    vector<Point3f> colors;
    colors.push_back(Point3f(1.0, 0.5, 0.0));
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



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    /*************
    * PARAMETERS *
    *************/
   
    const string PosterMeasurementsFilepath = "/home/wehak/Dropbox/ACIT master/data/poster_json/poster_tag_positions.json";
    const string modelPositionFilepath = "/home/wehak/Dropbox/ACIT master/data/poster_json/poster_model_positions.json";
    
    // const string CameraMetricsFilepath = "/home/wehak/Dropbox/ACIT master/data/cam_calibration_json/samsung_s20_h_1080.json";
    const string CameraMetricsFilepath = "/home/wehak/Dropbox/ACIT master/data/cam_calibration_json/blueye_tank_1080_nodistort.json";

    // const string inputVideo = "/home/wehak/Videos/master/input/samsung_20_horizontal/oceanlab_in_air.mp4";
    const string inputVideo = "/media/wehak/OS/Users/hweyd/Documents/My Videos/oceanlab_video/dataset/test_1_1.mp4";

    const int MIN_FIDUCIALS = 5;
    const float TVEC_COEF = 2;

    const bool record = false;
    const bool undistortFrame = true;


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


    float fiducialTagSize = 0.14; // [m]
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_APRILTAG_16h5);

    // measure FPS
    const int nMeanSamples = 5;
    vector<std::chrono::duration<double>> t_iteration;

    // camera image size
    // int width = 1138;
    // int height = 640;
    int width = 1920;
    int height = 1080;
    
    // near and far plane of the OpenGL view frustum
    float zNear = 10.0;
    float zFar = 10000.0;
    
    // camera instrinsics 
    // Matx33f K = Matx33f(650.048, 0, 324.328, 0, 647.183, 257.323, 0, 0, 1); // sample defaults
    // Matx33f K = Matx33f(893.6084327447165, 0.0, 579.7766602490716, 0.0, 895.7142289287846, 323.8682945945517, 0.0, 0.0, 1.0); // canon g5x video
    // Matx33f K = Matx33f(543.7763964666945, 0.0, 338.10640491567415, 0.0, 544.0291633358358, 222.64448193460834, 0.0, 0.0, 1.0); // dell xps webcam
    // Matx33f K = Matx33f(1742.846017065312, 0.0, 561.1918240669172, 0.0, 1747.3323386985453, 963.8549487863223, 0.0, 0.0, 1.0); // samsung s20 5g vertical
    // Matx33f K = Matx33f(1780.8593020747785, 0.0, 921.4040583220925, 0.0, 1775.352467023823, 538.4276419433833, 0.0, 0.0, 1.0); // samsung s20 5g horizontal
    // Matx33f K = Matx33f(952.3526885892495, 0.0, 939.5077453607088, 0.0, 949.6063395088117, 526.5838186301586, 0.0, 0.0, 1.0); // blueeye rov

    // distortion coefficients (k1, k2, p1, p2)
    // Matx14f distCoeffs =  Matx14f(0.0, 0.0, 0.0, 0.0);
    // Matx14f distCoeffs =  Matx14f(0.03278270430670608, -0.009389285121867291, -0.001952381467447879, -0.007213736827947127); // samsung s20 5g horizontal
    // Matx14f distCoeffs =  Matx14f(-0.25880017407368267, 0.11329539367233754, -0.000471868947826731, 0.00010304816317521514); // blueeye rov
    
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

    // get arguments
    string selectedModel;
    if (argc > 2) {
        int c;
        while( (c = getopt(argc, argv, "hm:")) != -1) {
            switch(c) {
                case 'm':
                    selectedModel = strdup(optarg);
                    // cout << optarg;
                    break;
                case 'h':
                    cout << "-m <model name>" << endl;
                    abort();
                default:
                    cout << "-m <model name>" << endl;
                    abort();
            }
        }
    }
    else {
        cout << "Error: No model selected. Use -m <model name>.\nAvailable models: "<< fiducial_detector.printModelNames() << endl;
        abort();
    }

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
        // fiducial_detector.intrinsics, 
        distCoeffs, 
        objects
        );
    // PoseEstimator6D* poseEstimator = new PoseEstimator6D(width, height, zNear, zFar, K, distCoeffs, objects);
    
    // move the OpenGL context for offscreen rendering to the current thread, if run in a seperate QT worker thread (unnessary in this example)
    //RenderingEngine::Instance()->getContext()->moveToThread(this);
    
    // active the OpenGL context for the offscreen rendering engine during pose estimation
    RenderingEngine::Instance()->makeCurrent();
    
    int timeout = 0;
    
    bool showHelp = true;

    // initialize camera    
    Mat frame;
    // VideoCapture cap("/home/wehak/Videos/master/input/samsung_20_horizontal/oceanlab_in_air.mp4");
    VideoCapture cap(inputVideo);
    VideoWriter outputVideo("/home/wehak/Videos/master/output/rbot_pose_test.avi", VideoWriter::fourcc('M','J','P','G'), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    
    // VideoCapture cap;
    // int deviceID = 0; // default camera
    // int apiID = cv::CAP_ANY; // autodetect default API
    // cap.open(deviceID, apiID); // open camera

    // check if successful 
    if (!cap.isOpened()) {
        cout << "Error: Unable to open camera\n";
        return -1;
    }

    // record measurements
    vector< vector<string> > measurements;

    while(true)
    {
        auto tStart = std::chrono::high_resolution_clock::now();

        // obtain an input image
        cap.read(frame);
        if (frame.empty()) {
            cout << "Error: Blank frame grabbed\n";
            break;
        }
        
        // get model pose from fiducial
        fiducialPose pose = fiducial_detector.getCleanModelPose(frame, selectedModel, TVEC_COEF, MIN_FIDUCIALS);

        // get model poses from RBOT
        poseEstimator->estimatePoses(frame, undistortFrame, true);
        
        // cout << objects[0]->getPose() << endl;
        cout << printMatrixSingleLine(objects[0]->getPose()) << endl;    
    
        // render the models with the resulting pose estimates ontop of the input image
        Mat result = drawResultOverlay(objects, frame);


        // draw poses on frame
        // aruco::drawAxis(
        //     result, 
        //     intrinsics,
        //     distortion,
        //     getRvecFromT(objects[0]->getPose()),
        //     getTvecFromT(objects[0]->getPose()),
        //     0.05
        //     );


        if (pose.n_fiducials >= MIN_FIDUCIALS) {
            // draw poses on frame
            aruco::drawAxis(
                result, 
                intrinsics,
                distortion,
                getRvecFromT(pose.T),
                getTvecFromT(pose.T),
                0.10
                );
        }

        if(showHelp)
        {
            putText(result, "Press '1' to initialize", Point(150, 250), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 1);
            putText(result, "or 'c' to quit", Point(205, 285), FONT_HERSHEY_DUPLEX, 1.0, Scalar(0, 255, 0), 1);
        }

        // measure FPS
        auto tEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = tEnd - tStart;
        t_iteration.emplace_back(diff);
        std::chrono::duration<double> sum = std::chrono::duration<double>::zero();
        vector<std::chrono::duration<double>> t_recent (t_iteration.end() - nMeanSamples, t_iteration.end());
        for (auto n : t_recent) {
            // cout << n.count() << ", ";
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
        if(key == (int)'1') {
            poseEstimator->setModelInitialPose(0, pose.T);
            poseEstimator->reset();

            poseEstimator->toggleTracking(frame, 0, undistortFrame);
            poseEstimator->estimatePoses(frame, undistortFrame, true);

            timeout = 1;
            showHelp = !showHelp;
        }
        if(key == (int)'2') // the same for a second object
        {
            //poseEstimator->toggleTracking(frame, 1, undistortFrame);
            //poseEstimator->estimatePoses(frame, false, undistortFrame);
        }

        // reset the system to the initial state
        if(key == (int)'r') poseEstimator->reset();
        // stop the demo
        if(key == (int)'c') break;
        // set pose from fiducal pose detector
        if(key == (int)'f') {
            poseEstimator->setModelPose(0, pose.T);
            // poseEstimator->reset();
            // timeout = 0;
        }
        // pause
        if(key == (int)'p') timeout = 0;

        // recording
        if (record == true) {
            outputVideo.write(result);
        }
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

    // write FPS data to file
    // ofstream log;
    // log.open("data/fps_log.txt");
    // if (log.is_open()) {
    //     for (auto n : t_iteration) {
    //         log << n.count() << ",";
    //     }
    // }
    // log.close();

    // write measurements data to file
    ofstream log;
    string output_filename = "./data/measurement_log_" + getFilename(inputVideo) + "_" + selectedModel + ".txt";
    // log.open("./data/measurement_log.txt");
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
