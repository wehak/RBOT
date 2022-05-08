#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "utility.h"
#include "json.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>



using json = nlohmann::json;

// create a folder if folder does not exist
void create_directory(std::string folderPath) {
    {
  
    // Creating a directory
    if (mkdir(folderPath.c_str(), 0777) == -1) return;
    else
        // std::cout << "Directory created";
        return;
    }
}

// get the filename from a file path
// courtesy of https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path
std::string getFilename(std::string filepath) {
    std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
    std::string::size_type const p(filename.find_last_of("."));
    return filename.substr(0, p);
}

// print matrix to single line
std::string printMatrixSingleLine(cv::Matx44f T) {
    std::string matrix = "[";
    for (int row=0 ; row<T.rows ; row++) {
        for (int col=0 ; col<T.cols ; col++) {
            if (row==0 && col==0) matrix = matrix + std::to_string(T(row, col));
            else matrix = matrix + " " + std::to_string(T(row, col));
        }
    }
    matrix += "]";
    return matrix;
}

// overloaded
void printMatrix(cv::Mat T) {
    for (int row=0 ; row<T.rows ; row++) {
        for (int col=0 ; col<T.cols ; col++) {
            std::cout << T.at<double>(row, col) << "\t";
        }
        std::cout << "\n";
    }
}
// overloaded
void printMatrix(cv::Matx44d T) {
    for (int row=0 ; row<T.rows ; row++) {
        for (int col=0 ; col<T.cols ; col++) {
            std::cout << T(row, col) << "\t";
        }
        std::cout << "\n";
    }
}

// overloaded
void printMatrix(cv::Matx33f T) {
    for (int row=0 ; row<T.rows ; row++) {
        for (int col=0 ; col<T.cols ; col++) {
            std::cout << T(row, col) << "\t";
        }
        std::cout << "\n";
    }
}

// // Checks if a matrix is a valid rotation matrix.
// bool isRotationMatrix(cv::Mat &R)
// {
//     cv::Mat Rt;
//     transpose(R, Rt);
//     cv::Mat shouldBeIdentity = Rt * R;
//     cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());

//     return  norm(I, shouldBeIdentity) < 1e-6;

// }

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
// cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
// {

//     assert(isRotationMatrix(R));

//     float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

//     bool singular = sy < 1e-6; // If

//     float x, y, z;
//     if (!singular)
//     {
//         x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
//         y = atan2(-R.at<double>(2,0), sy);
//         z = atan2(R.at<double>(1,0), R.at<double>(0,0));
//     }
//     else
//     {
//         x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
//         y = atan2(-R.at<double>(2,0), sy);
//         z = 0;
//     }
//     return cv::Vec3f(x, y, z);

// }


// get rotation and translation vectors from a homogeneous transfomration matrix
// void disassembleTransformationMatrix(cv::Matx44d T, cv::Vec3d rvec, cv::Vec3d tvec) {
//     cv::Matx33d R;
//     for (int i=0 ; i<3 ;  i++) {
//         for (int j=0 ; j<3 ; j++) {
//             R(i, j) = T(i, j);
//         }
//     }
//     cv::Rodrigues(R, rvec);

//     for (int i=0 ; i<3 ; i++) {
//         tvec[i] = T(i, 3);
//     }
// }


//
// cv::Vec3d avgTransformedVector(std::vector< cv::Vec3d > rvecs) {
//     int i=0;
//     double x, y, z = 0;
//     for (auto & element : rvecs) {
//         x += element[0];
//         y += element[1];
//         z += element[2];
//         i++;
//     }
//     cv::Vec3d result;
//     result[0] = x / float(i);
//     result[1] = y / float(i);
//     result[2] = z / float(i);

//     return result;
// }


// extract the rotation vector from a homogeneous transformation matrix
cv::Vec3d getRvecFromT(cv::Matx44d T) {
    cv::Matx33d R;
    cv::Vec3d rvec;
    for (int i=0 ; i<3 ;  i++) {
        for (int j=0 ; j<3 ; j++) {
            R(i, j) = T(i, j);
        }
    }
    cv::Rodrigues(R, rvec);
    return rvec;
}


// extract the translation vector from a homogeneous transformation matrix
cv::Vec3d getTvecFromT(cv::Matx44d T) {
    cv::Vec3d tvec;
    for (int i=0 ; i<3 ; i++) {
        tvec[i] = T(i, 3);
    }
    return tvec;
}


// perform matrix multiplication
cv::Matx44d matrixDot(cv::Matx44d A, cv::Matx44d B) {
    cv::Matx44d C;
    for (int i=0 ; i<A.rows ; i++) {
        for (int j=0 ; j<A.cols ; j++) {
            for (int k=0 ; k<A.cols ; k++) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }            
    }
    return C;
} 


//
cv::Matx44d getTagToCenterT(std::array<float, 2> placement) {
    // make it a identity matrix (no rotation)
    cv::Matx44d T = cv::Matx44d::eye();

    // T(0, 0) = 1;
    // T(1, 1) = 1;
    // T(2, 2) = 1;
    // T(3, 3) = 1;

    // add displacement vector to T
    T(0, 3) = placement[0]; // x displacement
    T(1, 3) = placement[1]; // y displacement

    return T;
}


std::vector<std::string> getModelNames(std::string filepath) {
    // convert string to input stream class
    std::ifstream inputData(filepath);

    // create json reader
    json j;
    inputData >> j;

    std::vector<std::string> data;
    for (json::iterator it=j.begin() ; it!=j.end() ; ++it) {
        data.push_back(it.key());
    }

    return data;
    
}

std::map<std::string, cv::Matx44d> getModelPositions(std::string filepath) {
    std::ifstream inputData(filepath);
    
    // create json reader
    json j;
    inputData >> j;

    // write data to an array
    std::map<std::string, cv::Matx44d> data;    
    for (json::iterator it=j.begin() ; it != j.end() ; ++it) {
        std::string name = it.key();
        double list[16];
        int i=0;
        for (auto & element : *it) {
            list[i] = element;
            i++;
        }

        // put array into a matx-object and place in a STL map
        cv::Matx44d m(list);
        data.insert({name, m});
    }

    // return map
    return data;
    

}


// helper function for getPosterMeasurements()
bool sortcol(const std::array<float, 3> a1, const std::array<float, 3> a2) {
    return a1[0] < a2[0];
}

// reads a json file with the distance of tag to the center of the poster
std::vector< std::array<float, 2> > getPosterMeasurements(std::string filepath) {

    // convert string to input stream class
    std::ifstream inputData(filepath);

    // create json reader
    json j;
    inputData >> j;

    // create vector to hold the data, contains arrays with the tag ID and XY translation
    std::vector< std::array<float, 3> > data;

    // iterate through the json file
    for (json::iterator it=j.begin() ; it != j.end() ; ++it) {
        
        // write each element to the array
        std::array<float, 3> row;
        row[0] = std::stof(it.key());
        int i=1;
        for (auto& element : *it) {
            row[i] = float(element) / 1000.0;
            i++;
        }

        // append array to vector
        data.push_back(row);
    }

    // sort vector based on value of first element in each array
    std::sort(data.begin(), data.end(), sortcol);

    // write XY positon to new vector of arrays and return
    std::vector< std::array<float, 2> > result;
    for (auto& row : data) {
        std::array<float, 2> pair;
        pair[0] = row[1];
        pair[1] = row[2];
        result.push_back(pair);
    }

    return result;
}


// reads a json file with camera intrinsics, return the camera matrix
cv::Matx33f getCameraIntrinsics(std::string filepath) {
    // convert string to input stream class
    std::ifstream inputData(filepath);

    // create json reader
    json j;
    inputData >> j;

    // convert json to array
    int i = 0;
    float data[9] = { };
    for (json::iterator it = j["R"].begin(); it != j["R"].end(); ++it) {
        for (auto& element : *it) {
            data[i] = element;
            i++;
        }
    }

    // convert array to opencv matrix object    
    cv::Matx33f R(data);
    return R;
}


// reads a json file with camera intrinsics, returns a vector with distortion coefficients
cv::Mat getCameraDistortion(std::string filepath) {
    std::ifstream inputData(filepath);

    json j;
    inputData >> j;

    // write the json info to an array
    int i = 0;
    float data[5] = { };
    for (json::iterator it = j["distortion"][0].begin() ; it != j["distortion"][0].end() ; ++it) {
        data[i] = it.value();
        i++;
    }

    // write the array to a opencv mat object
    cv::Mat distortion(5, 1, CV_32F, data);
    return distortion;

}


// return a homogeneous transformation matrix, given a rotation and translation vector
cv::Matx44d getHomogeneousTransformationMatrix(cv::Vec3d rvec, cv::Vec3d tvec) {
    // create a 4x4 matrix of all zeros
    cv::Matx44d T;
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // add R element by element
    for (int row=0 ; row<R.rows ; row++) {
        for (int col=0 ; col<R.cols ; col++) {
            T(row, col) = R.at<double>(row, col);
        }
    }

    // add translation vector to the fourth column
    for (int row=0 ; row<3 ; row++) {
        T(row, 3) = tvec[row];
    }

    // set bottom right corner to 1
    T(3, 3) = 1;

    return T;
}



