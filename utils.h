//  utils.h
//  SurfaceStereo
//
//  Created by Chen Liu on 9/30/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#ifndef SurfaceStereo_utils_h
#define SurfaceStereo_utils_h

#include <vector>
#include <set>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "Segment.h"


using namespace std;
using cv::Mat;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;


//read point cloud from a .obj file
vector<double> readPointCloudFromObj(const string filename, const int image_width, const int image_height, const double rotation_angle);

//save point cloud in a .ply file
void savePointCloudAsPly(const cv::Mat &image, const vector<double> &point_cloud, const char *filename);

//save point cloud as a mesh in a .ply file
void savePointCloudAsMesh(const vector<double> &point_cloud, const char *filename);

//load point cloud from a text file
vector<double> loadPointCloud(const string &filename);

//save point cloud to a text file
void savePointCloud(const vector<double> &point_cloud, const char *filename);

//draw a disp (inverse depth) image based on point cloud
Mat drawDispImage(const vector<double> &point_cloud, const int width, const MatrixXd &projection_matrix);

//draw a disp (inverse depth) image based on point cloud
Mat drawDispImage(const vector<double> &point_cloud, const int width, const int height);

//normalize point cloud on depth direction
vector<double> normalizePointCloudByZ(const vector<double> &point_cloud);

//zoom image and point cloud
void zoomScene(Mat &image, vector<double> &point_cloud, const double scale_x, const double scale_y);

//crop image and point cloud
void cropScene(Mat &image, vector<double> &point_cloud, const int start_x, const int start_y, const int end_x, const int end_y);

//inpaint empty point in a point cloud
vector<double> inpaintPointCloud(const vector<double> &point_cloud, const int image_width, const int image_height);

//read point cloud from a .ptx file
bool readPtxFile(const string &filename, cv::Mat &image, vector<double> &point_cloud, vector<double> &camera_parameters);

//unproject a pixel to 3D given depth
vector<double> unprojectPixel(const int pixel, const double depth, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);

//project a 3D point to image domain
int projectPoint(const vector<double> &point, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);

//calculate plane depth at pixel given plane parameters
double calcPlaneDepthAtPixel(const vector<double> &plane, const int pixel, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);

//normalize values based on mean and svar
double normalizeStatistically(const double value, const double mean, const double svar, const double normalized_value_for_mean, const double scale_factor);

#endif
