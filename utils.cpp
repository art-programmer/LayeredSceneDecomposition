//
//  utils.cpp
//  SurfaceStereo
//
//  Created by Chen Liu on 9/30/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#include "utils.h"

#include <stdio.h>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "cv_utils/cv_utils.h"

//#include <ceres/solver.h>
//#include <ceres/problem.h>
//#include <ceres/autodiff_cost_function.h>


double DISP_DEPTH_C = 0;

using namespace Eigen;
using namespace cv;
using namespace cv_utils;


void normalizePointCloud(vector<double> &point_cloud)
{
  const int NUM_POINTS = point_cloud.size() / 3;
    double min_x = 1000000, max_x = -1000000, min_y = 1000000, max_y = -1000000, min_z = 1000000, max_z = -1000000;
    for (int i = 0; i < NUM_POINTS; i++) {
        double x = point_cloud[i * 3 + 0];
        double y = point_cloud[i * 3 + 1];
        double z = point_cloud[i * 3 + 2];
        if (x < min_x)
            min_x = x;
        if (x > max_x)
            max_x = x;
        if (y < min_y)
            min_y = y;
        if (y > max_y)
            max_y = y;
        if (z < min_z)
            min_z = z;
        if (z > max_z)
            max_z = z;
    }
    
//    double range = max(max_x - min_x, max(max_y - min_y, max_z - min_z));
    double range = max(abs(max_z), abs(min_z));
    for (int i = 0; i < NUM_POINTS; i++) {
        point_cloud[i * 3 + 0] /= range;
        point_cloud[i * 3 + 1] /= range;
        point_cloud[i * 3 + 2] /= range;
    }
}

vector<double> loadCoordinates(const char *filename)
{
    DISP_DEPTH_C = 1;
    ifstream in_str(filename);
    int num = -1;
    in_str >> num;
    vector<double> coordinates(num * 3);
    for (int i = 0; i < num * 3; i++)
        in_str >> coordinates[i];
    in_str.close();
    return coordinates;
}

void saveCoordinates(const vector<double> &coordinates, const char *filename)
{
    ofstream out_str(filename);
    out_str << coordinates.size() / 3 << endl;
    for (int i = 0; i < coordinates.size() / 3; i++) {
        for (int j = 0; j < 3; j++)
            out_str << coordinates[i * 3 + j] << '\t';
        out_str << endl;
    }
    out_str.close();
}


Mat blendImage(const Mat &image_1, const Mat &image_2, const int type)
{
  const int IMAGE_WIDTH = image_1.cols;
  const int IMAGE_HEIGHT = image_1.rows;
  Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
  Mat image_1_color(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
  if (image_1.channels() == 1)
    cvtColor(image_1, image_1_color, CV_GRAY2BGR);
  else
    image_1.copyTo(image_1_color);
  Mat image_2_color(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
  if (image_2.channels() == 1)
    cvtColor(image_2, image_2_color, CV_GRAY2BGR);
  else
    image_2.copyTo(image_2_color);
  for (int y = 0; y < IMAGE_HEIGHT; y++) {
    uchar *data_1 = image_1_color.ptr<uchar>(y);
    uchar *data_2 = image_2_color.ptr<uchar>(y);
    uchar *data = image.ptr<uchar>(y);
    for (int x = 0; x < IMAGE_WIDTH; x++)
      for (int c = 0; c < 3; c++)
	if (type == 0)
	  data[3 * x + c] = (data_1[3 * x + c] + data_2[3 * x + c]) / 2;
	else
	  data[3 * x + c] = data_1[3 * x + c] * data_2[3 * x + c] / 255;
  }
  return image;
}

double calcFitError(VectorXd &surface_model, const double x, const double y, const double z, const int px, const int py)
{
  double true_depth = sqrt(pow(x, 2) + pow(y, 2));
  if (true_depth < 0.000001)
    return 0;

  double distance = abs(x * surface_model(0) + y * surface_model(1) + z * surface_model(2) - surface_model(3));

  return distance;
}

double calcAngle(const double x, const double y)
{
    if (x == 0) {
        if (y == 0)
            return 0;
        else if (y > 0)
            return 1.57;
        else
            return -1.57;
    }
    else if (x > 0)
        return atan(y / x);
    else {
        if (y >= 0)
            return atan(y / x) + 3.14;
        else
            return atan(y / x) - 3.14;
    }
}

//vector<double> fitLine2D(const vector<vector<double> > &points)
//{
//    vector<double> parameters;
//    if (points.size() < 3)
//        return parameters;
//    
//    CMatrix<double> A(points.size(), 2);
//    vector<double> b(points.size());
//    for (int i = 0; i < points.size(); i++) {
//        A.set(i, 0, points[i][0]);
//        A.set(i, 1, points[i][1]);
//        b[i] = 1;
//    }
//    parameters = solveLinearEquations(A, b);
//    parameters.push_back(1);
//    double length = sqrt(pow(parameters[0], 2) + pow(parameters[1], 2));
//    for (int i = 0; i < 3; i++)
//        parameters[i] /= length;
//    
//    double fit_error = 0;
//    for (int i = 0; i < points.size(); i++)
//        fit_error += abs(parameters[0] * points[i][0] + parameters[1] * points[i][1] - parameters[2]);
//    fit_error /= points.size();
//    
//    if (fit_error > 1)
//        return vector<double>(0);
//    return parameters;
//}

Mat drawSurfaceIdImage(const Mat &surface_id_image, const int type)
{
  Mat color_image = surface_id_image.clone();
  for (int y = 0; y < color_image.rows; y++) {
    uchar *data = color_image.ptr<uchar>(y);
    for (int x = 0; x < color_image.cols; x++)
      for (int c = 0; c < 3; c++)
	data[x * 3 + c] = static_cast<int>(pow(data[x * 3 + c], c + 3)) % 255;
  }
  return color_image;
}

double calcDistance(vector<double> &surface_model, const int x, const int y, const int width, const int height)
{
    double R = width / 6.28;
    double angle_1 = (width / 2 - x) / R;
    double angle_2 = (height / 2 - y) / R;
    double distance_1 = 0, distance_2 = 0;
    double v_x = cos(angle_2) * cos(angle_1);
    double v_y = cos(angle_2) * sin(angle_1);
    double v_z = sin(angle_2);
    vector<double> plane = surface_model;
    double cos_value = plane[0] * v_x + plane[1] * v_y + plane[2] * v_z;
    
    if (x == 277 && y == 208)
        bool wait = true;
    return abs(plane[3] / cos_value);
}

void initCoordinates(vector<double> &coordinates, Mat &disp_image)
{
    //    double min_depth = 1000000;
    //    double max_depth = 0;
    int num_effective_depth_values = 0;
    double *depth_values = new double[coordinates.size() / 3];
    double depth_mean = 0;
    double depth_svar = 0;
    for (int i = 0; i < coordinates.size() / 3; i++) {
        double depth = sqrt(pow(coordinates[i * 3 + 0], 2) + pow(coordinates[i * 3 + 1], 2));
        if (depth < 0.000001)
            continue;
        num_effective_depth_values++;
        //            if (depth < min_depth)
        //                min_depth = depth;
        //            if (depth > max_depth)
        //                max_depth = depth;
        depth_mean += depth;
        depth_svar += pow(depth, 2);
    }
    depth_mean /= num_effective_depth_values;
    depth_svar /= num_effective_depth_values;
    depth_svar -= pow(depth_mean, 2);
    
    num_effective_depth_values = 0;
    for (int i = 0; i < coordinates.size() / 3; i++) {
        double depth = sqrt(pow(coordinates[i * 3 + 0], 2) + pow(coordinates[i * 3 + 1], 2));
        if (depth < 0.000001)
            continue;
        if (abs(depth - depth_mean) > 5 * depth_svar) {
            coordinates[i * 3 + 0] = coordinates[i * 3 + 1] = coordinates[i * 3 + 2] = 0;
            continue;
        }
        
        depth_values[num_effective_depth_values] = depth;
        num_effective_depth_values++;
        //            if (depth < min_depth)
        //                min_depth = depth;
        //            if (depth > max_depth)
        //                max_depth = depth;
    }
    
    
    double max_ratio = 1.0  / 100;
    int k = num_effective_depth_values * (1 - max_ratio);
    nth_element(depth_values, depth_values + k, depth_values + num_effective_depth_values);
    DISP_DEPTH_C = depth_values[k] * 10;
    
    const int IMAGE_WIDTH = disp_image.cols;
    const int IMAGE_HEIGHT = disp_image.rows;
    vector<vector<int> > disp_mask(IMAGE_WIDTH, vector<int>(IMAGE_HEIGHT, -1));
    for (int i = 0; i < coordinates.size() / 3; i++) {
        double depth = sqrt(pow(coordinates[i * 3 + 0], 2) + pow(coordinates[i * 3 + 1], 2));
        if (depth < 0.000001)
            continue;
        int x = i % IMAGE_WIDTH;
        int y = i / IMAGE_HEIGHT;
        disp_mask[x][y] = min(static_cast<int>(DISP_DEPTH_C / depth + 0.5), 255);
    }
    
    bool has_change = true;
    while (has_change == true) {
        has_change = false;
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int y = 0; y < IMAGE_HEIGHT; y++) {
                if (disp_mask[x][y] != -1)
                    continue;
                int neighbor_x = x + rand() % 3 - 1;
                int neighbor_y = y + rand() % 3 - 1;
                if (neighbor_x >= 0 && neighbor_x < IMAGE_WIDTH && neighbor_y >= 0 && neighbor_y < IMAGE_HEIGHT && disp_mask[neighbor_x][neighbor_y] >= 0) {
                    disp_mask[x][y] = disp_mask[neighbor_x][neighbor_y];
                }
            }
        }
    }
    
//    for (int x = 0; x < disp_image->width; x++) {
//        for (int y = 0; y < disp_image->height; y++) {
//            if (disp_mask[x][y] != 255)
//                continue;
//            for (int neighbor_x = max(x - 1, 0); neighbor_x <= min(x + 1, disp_image->width - 1); neighbor_x++)
//                for (int neighbor_y = max(y - 1, 0); neighbor_y <= min(y + 1, disp_image->height - 1); neighbor_y++)
//                    if (disp_mask[neighbor_x][neighbor_y] != 255)
//                        disp_mask[x][y] = disp_mask[neighbor_x][neighbor_y];
//        }
//    }

    
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
      uchar *data = disp_image.ptr<uchar>(y);
        for (int x = 0; x < IMAGE_WIDTH; x++)
            data[x] = disp_mask[x][y];
    }
    medianBlur(disp_image, disp_image, 3);
}

Mat readDispFromPFM(const char *filename, const int scale)
{
    ifstream in_str(filename);
    char temp;
    in_str >> temp;
    in_str >> temp;
    int width, height;
    in_str >> width >> height;
    double scale_factor;
    in_str >> scale_factor;
    scale_factor = abs(scale_factor);
    
//    in_str >> temp;
//    if (temp == ' ')
//        bool wait = true;
    
    Mat disp_image(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
      uchar *data = disp_image.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            uchar value_1, value_2, value_3, value_4;
            in_str >> value_1 >> value_2 >> value_3 >> value_4;
//            cout << (int)value_1 << '\t' << (int)value_2 << '\t' << (int)value_3 << '\t' << (int)value_4 << endl;
            int sign = value_4 / 128;
            int exponent = value_4 % 128 * 2 + value_3 / 128;
            if (exponent > 127)
                exponent = exponent % 128 + 1;
            else
                exponent = 127 - exponent;
            int mantissa = value_3 % 128 * 256 * 256 + value_2 * 256 + value_1;
            double disp = 1.0 / (pow(2, 23 - exponent)) * mantissa;
            if (sign == 1)
                disp *= -1;

//            float disp;
//            in_str.read(reinterpret_cast<char *>(&disp), 4);
//            uchar *bytes = reinterpret_cast<uint8_t *>(&disp);
//            swap(bytes[0], bytes[3]);
//            swap(bytes[1], bytes[2]);
            
            if (disp < 0 || disp > 255)
                bool wait = true;

            disp *= scale_factor;
            data[x] = min(static_cast<int>(disp * scale + 0.5), 255);
        }
    }
    return disp_image;
}

Mat deleteBoundary(const Mat &image, const int boundary_value)
{
  Mat image_boundaryless = image.clone();
    bool has_change = true;
    int width = image_boundaryless.cols;
    int height = image_boundaryless.rows;
    while (has_change == true) {
        has_change = false;
        for (int y = 0; y < height; y++) {
	  uchar *data = image_boundaryless.ptr<uchar>(y);
            for (int x = 0; x < width; x++) {
                if (data[x * 3] != boundary_value)
                    continue;
                int neighbor_x = x;
                int neighbor_y = y;
                while ((neighbor_x == x && neighbor_y == y) || (x < 0 || x >= width || y < 0 || y >= height)) {
                    neighbor_x = x + (rand() % 3 - 1);
                    neighbor_y = y + (rand() % 3 - 1);
                }
		Vec3b color = image_boundaryless.at<Vec3b>(neighbor_y, neighbor_x);
                for (int c = 0; c < 3; c++)
		  data[x * 3 + c] = color[c];
                has_change = true;
            }
        }
    }
    return image_boundaryless;
}

vector<int> findPoints(const Mat &image, const int value)
{
  int width = image.cols;
    int height = image.rows;
    
    vector<int> points;
    for (int y = 0; y < height; y++) {
      const uchar *data = image.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (data[x * 3] == value)
                points.push_back(y * width + x);
        }
    }
    return points;
}

void drawPoints(Mat &image, const vector<int> points, const int value)
{
  int width = image.cols;
    int height = image.rows;
    
    for (int i = 0; i < points.size(); i++) {
        int x = points[i] % width;
        int y = points[i] / width;
        Vec3b color = image.at<Vec3b>(y, x);
        for (int c = 0; c < 3; c++) {
            color[c] = value;
        }
	image.at<Vec3b>(y, x) = color;
    }
}

Mat drawMask(const Mat &image, const set<int> indices)
{
    int width = image.cols;
    int height = image.rows;

    Mat mask = Mat::zeros(height, width, CV_8UC1);

    for (int y = 0; y < height; y++) {
      const uchar *data = image.ptr<uchar>(y);
      uchar *mask_data = mask.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int index = (data[x * 3 + 0] + data[x * 3 + 1] + data[x * 3 + 2]) / 3;
            if (x == 265 && y == 325)
                bool wait = true;
            if (indices.count(index) > 0)
                mask_data[x] = 255;
        }
    }
    return mask;
}

void mergeSurfaces(Mat &image, const set<int> indices, const int value)
{
  int width = image.cols;
    int height = image.rows;
    
    for (int y = 0; y < height; y++) {
      uchar *data = image.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int index = data[x * 3 + 0];
            if (indices.count(index) > 0)
                for (int c = 0; c < 3; c++)
                    data[x * 3 + c] = value;
        }
    }
}

void drawSurface(Mat &image, const Mat &mask, const int ori_value, const int new_value)
{
  int width = image.cols;
    int height = image.rows;
    
    for (int y = 0; y < height; y++) {
      uchar *data = image.ptr<uchar>(y);
      const uchar *mask_data = mask.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            if (mask_data[x] < 128)
                continue;
            if (data[x * 3 + 0] == ori_value)
                for (int c = 0; c < 3; c++)
                    data[x * 3 + c] = new_value;
        }
    }
}

//        Mat &mask = cvLoadImage("layer/mask.png", 0);
//        Mat &surface_id_image = cvCreateImage(cvGetSize(mask), 8, 3);
//        for (int y = 0; y < mask->height; y++) {
//            uchar *data = (uchar *)(surface_id_image->imageData + y * surface_id_image->widthStep);
//            uchar *mask_data = (uchar *)(mask->imageData + y * mask->widthStep);
//            for (int x = 0; x < mask->width; x++) {
//                if (mask_data[x] < 64)
//                    for (int c = 0; c < 3; c++)
//                        data[x * 3 + c] = 0;
//                else if (mask_data[x] > 192)
//                    for (int c = 0; c < 3; c++)
//                        data[x * 3 + c] = 255;
//                else
//                    for (int c = 0; c < 3; c++)
//                        data[x * 3 + c] = 128;
//            }
//        }
//        cvSaveImage("layer/surface_id_1.bmp", surface_id_image);


//        Mat &surface_id_image = cvLoadImage("layer/surface_id_1.bmp");

Mat cropImage(const Mat &image, const int start_x, const int start_y, const int width, const int height)
{
  Mat cropped_image;
  Mat(image, Rect(start_y, start_x, height, width)).copyTo(cropped_image);
  return cropped_image;
  Mat new_image(height, width, image.type());
  for (int y = start_y; y < start_y + height; y++) {
    const uchar *data = image.ptr<uchar>(y);
    uchar *new_data = new_image.ptr<uchar>(y - start_y);
    for (int x = start_x; x < start_x + width; x++)
      for (int c = 0; c < image.channels(); c++)
	new_data[(x - start_x) * image.channels() + c] = data[x * image.channels() + c];
  }
  return new_image;
}

vector<double> calcCrossLine(const vector<double> &plane_1, const vector<double> &plane_2)
{
  Vector3d normal_1;
  Vector3d normal_2;
  for (int c = 0; c < 3; c++) {
    normal_1(c) = plane_1[c];
    normal_2(c) = plane_2[c];
  }
  Vector3d direction = normal_1.cross(normal_2);

  direction.normalize();
    
    int max_direction = -1;
    if (direction(0) > direction(1) && direction(0) > direction(2))
        max_direction = 0;
    else if (direction(1) > direction(2))
        max_direction = 1;
    else
        max_direction = 2;

//    CMatrix<double> A(3, 3);
//    A.set(0, 0, plane_1->A);
//    A.set(0, 1, plane_1->B);
//    A.set(0, 2, plane_1->C);
//    A.set(1, 0, plane_2->A);
//    A.set(1, 1, plane_2->B);
//    A.set(1, 2, plane_2->C);
//    A.set(2, 0, 0);
//    A.set(2, 1, 0);
//    A.set(2, 2, 0);
//    A.set(2, max_direction, 1);
//    vector<double> b(3);
//    b[0] = plane_1->D;
//    b[1] = plane_2->D;
//    b[2] = 0;
//    vector<double> point = solveLinearEquations(A, b);
    
//    CvMat *A = cvCreateMat(3, 3, CV_32FC1);
//    CvMat *b = cvCreateMat(3, 1, CV_32FC1);
//    cvmSet(A, 0, 0, plane_1->A);
//    cvmSet(A, 0, 1, plane_1->B);
//    cvmSet(A, 0, 2, plane_1->C);
//    cvmSet(A, 1, 0, plane_2->A);
//    cvmSet(A, 1, 1, plane_2->B);
//    cvmSet(A, 1, 2, plane_2->C);
//    cvmSet(A, 2, 0, 0);
//    cvmSet(A, 2, 1, 0);
//    cvmSet(A, 2, 2, 0);
//    cvmSet(b, 0, 0, plane_1->D);
//    cvmSet(b, 1, 0, plane_2->D);
//    cvmSet(b, 2, 0, 0);
//    CvMat *result = cvCreateMat(3, 1, CV_32FC1);
//    cvSolve(A, b, result);
//    vector<double> point(3);
//    for (int c = 0; c < 3; c++)
//        point[c] = cvmGet(result, c, 0);
    
    MatrixXf A(2, 3);
    VectorXf b(2);
    for (int c = 0; c < 3; c++) {
      A(0, c) = plane_1[3];
      A(1, c) = plane_2[3];
    }
    b(0) = plane_1[3];
    b(1) = plane_2[3];
    VectorXf point = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
    
    vector<double> line(6);
    for (int c = 0; c < 3; c++) {
      line[c] = direction(c);
      line[c + 3] = point(c);
    }
    return line;
}


Mat modifySurfaceIdImage(const Mat &surface_id_image, const int type)
{
    srand(time(NULL));
    if (type == 1) {
      Mat modified_surface_id_image = surface_id_image.clone();
      int width = surface_id_image.cols;
        int height = surface_id_image.rows;
        vector<vector<int> > surface_ids(width, vector<int>(height));
        vector<vector<int> > new_surface_ids(width, vector<int>(height, -1));
        
        int small_surface_id_threshold = width * height / 10000;
        for (int y = 0; y < height; y++) {
	  const uchar *data = surface_id_image.ptr<uchar>(y);
            for (int x = 0; x < width; x++) {
                int id = data[3 * x + 0] * 256 * 256 + data[3 * x + 1] * 256 + data[3 * x + 2];
                surface_ids[x][y] = id;
            }
        }
        
        int new_id = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (new_surface_ids[x][y] >= 0)
                    continue;
                int id = surface_ids[x][y];
                
                vector<int> border_points_x;
                vector<int> border_points_y;
                border_points_x.push_back(x);
                border_points_y.push_back(y);
                vector<int> region_points_x;
                vector<int> region_points_y;
                
                while (border_points_x.size() > 0) {
                    vector<int> new_border_points_x;
                    vector<int> new_border_points_y;
                    for (int i = 0; i < border_points_x.size(); i++) {
                        if (border_points_x[i] >= 0 && border_points_x[i] < width && border_points_y[i] >= 0 && border_points_y[i] < height && surface_ids[border_points_x[i]][border_points_y[i]] == id && new_surface_ids[border_points_x[i]][border_points_y[i]] == -1) {
                            
                            new_surface_ids[border_points_x[i]][border_points_y[i]] = new_id;
                            region_points_x.push_back(border_points_x[i]);
                            region_points_y.push_back(border_points_y[i]);
                            
                            new_border_points_x.push_back(border_points_x[i] - 1);
                            new_border_points_y.push_back(border_points_y[i]);
                            new_border_points_x.push_back(border_points_x[i] + 1);
                            new_border_points_y.push_back(border_points_y[i]);
                            new_border_points_x.push_back(border_points_x[i]);
                            new_border_points_y.push_back(border_points_y[i] - 1);
                            new_border_points_x.push_back(border_points_x[i]);
                            new_border_points_y.push_back(border_points_y[i] + 1);
                            new_border_points_x.push_back(border_points_x[i] - 1);
                            new_border_points_y.push_back(border_points_y[i] - 1);
                            new_border_points_x.push_back(border_points_x[i] + 1);
                            new_border_points_y.push_back(border_points_y[i] - 1);
                            new_border_points_x.push_back(border_points_x[i] - 1);
                            new_border_points_y.push_back(border_points_y[i] + 1);
                            new_border_points_x.push_back(border_points_x[i] + 1);
                            new_border_points_y.push_back(border_points_y[i] + 1);
                        }
                    }
                    border_points_x = new_border_points_x;
                    border_points_y = new_border_points_y;
                }
                
                if (region_points_x.size() < 10) {
                    bool has_change = true;
                    while (has_change == true) {
                        has_change = false;
                        for (int i = 0; i < region_points_x.size(); i++) {
                            int ori_x = region_points_x[i];
                            int ori_y = region_points_y[i];
                            if (new_surface_ids[ori_x][ori_y] != new_id)
                                continue;
                            int new_x = ori_x;
                            int new_y = ori_y;
                            while (new_x == ori_x && new_y == ori_y) {
                                new_x = min(max(ori_x + rand() % 3 - 1, 0), width - 1);
                                new_y = min(max(ori_y + rand() % 3 - 1, 0), height - 1);
                            }
                            if (new_surface_ids[new_x][new_y] != new_id) {
                                new_surface_ids[ori_x][ori_y] = new_surface_ids[new_x][new_y];
                                has_change = true;
                                uchar *data_1 = modified_surface_id_image.ptr<uchar>(ori_y);
                                uchar *data_2 = modified_surface_id_image.ptr<uchar>(new_y);
                                for (int c = 0; c < 3; c++)
                                    data_1[ori_x * 3 + c] = data_2[new_x * 3 + c];
                            }
                        }
                    }
                }
                
                new_id++;
            }
        }
        
        map<int, int> color_table;
        for (int y = 0; y < height; y++) {
	  uchar *data = modified_surface_id_image.ptr<uchar>(y);
            for (int x = 0; x < width; x++) {
                int id = data[x * 3 + 0] * 256 * 256 + data[x * 3 + 1] * 256 + data[x * 3 + 2];
                int r = 0, g = 0, b = 0;
                if (color_table.count(id) > 0) {
                    r = color_table[id] / (256 * 256);
                    g = color_table[id] % (256 * 256) / 256;
                    b = color_table[id] % 256;
                }
                else {
                    bool invalid_color = true;
                    while (invalid_color == true) {
                        invalid_color = false;
                        r = rand() % 256;
                        g = rand() % 256;
                        b = rand() % 256;
                        for (map<int, int>::const_iterator color_it = color_table.begin(); color_it != color_table.end(); color_it++)
                            if ((r + g + b) / 3 == (color_it->second / (256 * 256) + color_it->second % (256 * 256) / 256 + color_it->second % 256) / 3)
                                invalid_color = true;
                    }
                    color_table[id] = r * 256 * 256 + g * 256 + b;
                }
                
                if (data[3 * x + 0] == data[3 * x + 1] && data[3 * x + 0] == data[3 * x + 2])
                    for (int c = 0; c < 3; c++)
                        data[3 * x + c] = (r + g + b) / 3;
                else {
                    data[3 * x + 0] = r;
                    data[3 * x + 1] = g;
                    data[3 * x + 2] = b;
                }
            }
        }
        return modified_surface_id_image;
    }
    
    set<int> spline_ids;
    int width = surface_id_image.cols;
    int height = surface_id_image.rows;
    vector<vector<int> > surface_ids(width, vector<int>(height));
    vector<vector<int> > new_surface_ids(width, vector<int>(height, -1));
    for (int y = 0; y < height; y++) {
      const uchar *data = surface_id_image.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int id = data[3 * x + 0] * 256 * 256 + data[3 * x + 1] * 256 + data[3 * x + 2];
            surface_ids[x][y] = id;
            if (data[3 * x + 0] != data[3 * x + 1] || data[3 * x + 0] != data[3 * x + 2])
                spline_ids.insert(id);
        }
    }
    
    int small_surface_id_threshold = width * height / 10000;
    set<int> small_surface_ids;
    set<int> spline_new_ids;
    int new_id = 0;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (new_surface_ids[x][y] >= 0)
                continue;
            int id = surface_ids[x][y];
            
            vector<int> border_points_x;
            vector<int> border_points_y;
            border_points_x.push_back(x);
            border_points_y.push_back(y);
            int num_nodes = 0;
            while (border_points_x.size() > 0) {
                vector<int> new_border_points_x;
                vector<int> new_border_points_y;
                for (int i = 0; i < border_points_x.size(); i++) {
                    if (border_points_x[i] >= 0 && border_points_x[i] < width && border_points_y[i] >= 0 && border_points_y[i] < height && surface_ids[border_points_x[i]][border_points_y[i]] == id && new_surface_ids[border_points_x[i]][border_points_y[i]] == -1) {
                        
                        new_surface_ids[border_points_x[i]][border_points_y[i]] = new_id;
                        num_nodes++;
                        
                        new_border_points_x.push_back(border_points_x[i] - 1);
                        new_border_points_y.push_back(border_points_y[i]);
                        new_border_points_x.push_back(border_points_x[i] + 1);
                        new_border_points_y.push_back(border_points_y[i]);
                        new_border_points_x.push_back(border_points_x[i]);
                        new_border_points_y.push_back(border_points_y[i] - 1);
                        new_border_points_x.push_back(border_points_x[i]);
                        new_border_points_y.push_back(border_points_y[i] + 1);
                        new_border_points_x.push_back(border_points_x[i] - 1);
                        new_border_points_y.push_back(border_points_y[i] - 1);
                        new_border_points_x.push_back(border_points_x[i] + 1);
                        new_border_points_y.push_back(border_points_y[i] - 1);
                        new_border_points_x.push_back(border_points_x[i] - 1);
                        new_border_points_y.push_back(border_points_y[i] + 1);
                        new_border_points_x.push_back(border_points_x[i] + 1);
                        new_border_points_y.push_back(border_points_y[i] + 1);
                    }
                }
                border_points_x = new_border_points_x;
                border_points_y = new_border_points_y;
            }
            if (spline_ids.count(id) > 0)
                spline_new_ids.insert(new_id);
            
            if (num_nodes < 10)
                small_surface_ids.insert(new_id);
            
            new_id++;
        }
    }
    
    vector<int> small_surface_point_x;
    vector<int> small_surface_point_y;
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (small_surface_ids.count(new_surface_ids[x][y]) == 0)
                continue;
            small_surface_point_x.push_back(x);
            small_surface_point_y.push_back(y);
        }
    }
    bool has_change = true;
    while (has_change == true) {
        has_change = false;
        for (int i = 0; i < small_surface_point_x.size(); i++) {
            int x = small_surface_point_x[i];
            int y = small_surface_point_y[i];
            if (small_surface_ids.count(new_surface_ids[x][y]) == 0)
                continue;
            int new_x = min(max(x + rand() % 3 - 1, 0), width - 1);
            int new_y = min(max(y + rand() % 3 - 1, 0), height - 1);
            if (small_surface_ids.count(new_surface_ids[new_x][new_y]) == 0) {
                new_surface_ids[x][y] = new_surface_ids[new_x][new_y];
                has_change = true;
            }
        }
    }
    
    Mat modified_surface_id_image = Mat::zeros(height, width, CV_8UC3);
    map<int, int> color_table;
    for (int y = 0; y < height; y++) {
      uchar *data = modified_surface_id_image.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int id = new_surface_ids[x][y];
            
            int r = 0, g = 0, b = 0;
            if (color_table.count(id) > 0) {
                r = color_table[id] / (256 * 256);
                g = color_table[id] % (256 * 256) / 256;
                b = color_table[id] % 256;
            }
            else {
                bool invalid_color = true;
                while (invalid_color == true) {
                    invalid_color = false;
                    r = rand() % 256;
                    g = rand() % 256;
                    b = rand() % 256;
                    for (map<int, int>::const_iterator color_it = color_table.begin(); color_it != color_table.end(); color_it++)
                        if ((r + g + b) / 3 == (color_it->second / (256 * 256) + color_it->second % (256 * 256) / 256 + color_it->second % 256) / 3)
                            invalid_color = true;
                }
                color_table[id] = r * 256 * 256 + g * 256 + b;
            }
            
            if (spline_new_ids.count(id) == 0)
                for (int c = 0; c < 3; c++)
                    data[3 * x + c] = (r + g + b) / 3;
            else {
                data[3 * x + 0] = r;
                data[3 * x + 1] = g;
                data[3 * x + 2] = b;
            }
        }
    }
    return modified_surface_id_image;
}

vector<double> loadPointCloud(const string &filename)
{
    ifstream in_str(filename);
    int num = -1;
    in_str >> num;
    vector<double> point_cloud(num * 3);
    for (int i = 0; i < num * 3; i++) {
        in_str >> point_cloud[i];
        if (point_cloud[i] == 0)
            bool wait = true;
    }
    in_str.close();
    return point_cloud;
}

void savePointCloud(const vector<double> &point_cloud, const char *filename)
{
    ofstream out_str(filename);
    out_str << point_cloud.size() / 3 << endl;
    for (int i = 0; i < point_cloud.size() / 3; i++) {
        for (int j = 0; j < 3; j++)
            out_str << point_cloud[i * 3 + j] << '\t';
        out_str << endl;
    }
    out_str.close();
}

void savePointCloudAsPly(const Mat &image, const vector<double> &point_cloud, const char *filename)
{
  ofstream out_str(filename);
    int num_points = point_cloud.size() / 3;
    out_str << "ply" << endl;
    out_str << "format ascii 1.0" << endl;
    out_str << "element vertex " << num_points << endl;
    out_str << "property float x" << endl;
    out_str << "property float y" << endl;
    out_str << "property float z" << endl;
    out_str << "property uchar red" << endl;
    out_str << "property uchar green" << endl;
    out_str << "property uchar blue" << endl;
    out_str << "end_header" << endl;
    for (int i = 0; i < num_points; i++) {
      Vec3b color = image.at<Vec3b>(i / image.cols, i % image.cols);
      out_str << -point_cloud[i * 3 + 0] << ' ' << -point_cloud[i * 3 + 1] << ' ' << point_cloud[i * 3 + 2] << ' ' << static_cast<int>(color[2]) << ' ' << static_cast<int>(color[1]) << ' ' << static_cast<int>(color[0]) << endl;
    }
    out_str.close();
}

void savePointCloudAsMesh(const vector<double> &point_cloud, const char *filename)
{
  ofstream out_str(filename);
  int num_valid_points = 0;
  vector<int> pixel_index_map(point_cloud.size() / 3);
  for (int pixel = 0; pixel < point_cloud.size() / 3; pixel++) {
    if (point_cloud[pixel * 3 + 2] < 0)
      continue;
    pixel_index_map[pixel] = num_valid_points;
    num_valid_points++;
  }
  out_str << "ply" << endl;
  out_str << "format ascii 1.0" << endl;
  out_str << "element vertex " << num_valid_points << endl;
  out_str << "property float x" << endl;
  out_str << "property float y" << endl;
  out_str << "property float z" << endl;
  out_str << "end_header" << endl;
  for (int pixel = 0; pixel < point_cloud.size() / 3; pixel++) {
    if (point_cloud[pixel * 3 + 2] < 0)
      continue;
    out_str << -point_cloud[pixel * 3 + 0] << ' ' << -point_cloud[pixel * 3 + 1] << ' ' << point_cloud[pixel * 3 + 2] << endl;
  }
  
  out_str.close();
}

Mat drawDispImage(const vector<double> &point_cloud, const int width, const MatrixXd &projection_matrix)
{
    const int height = point_cloud.size() / 3 / width;
    Mat disp(height, width, CV_8UC1);
    for (int y = 0; y < height; y++) {
      uchar *data = disp.ptr<uchar>(y);
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            //            double depth = sqrt(pow(point_cloud[index * 3 + 0], 2) + pow(point_cloud[index * 3 + 1], 2));
	    //            double depth = point_cloud[index * 3 + 2];
	    VectorXd point(4, 1);
	    point << point_cloud[index * 3 + 0], point_cloud[index * 3 + 1], point_cloud[index * 3 + 2], 1;
	    Vector3d point_2D = projection_matrix * point;
	    double depth = point_2D(2);
	    //	    if ((x % 10 == 0 && y % 10 == 0) || depth < 0)
	      //	      cout << x << '\t' << y << '\t' << depth << endl;
            data[x] = min(static_cast<int>(100 / depth + 0.5), 255);
        }
    }
    return disp;
}

Mat drawDispImage(const vector<double> &point_cloud, const int width, const int height)
{
  Mat disp_real = Mat::zeros(height, width, CV_64FC1);
  Mat disp_uchar = Mat::zeros(height, width, CV_8UC1);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (checkPointValidity(getPoint(point_cloud, y * width + x)) == false)
      	continue;
      double X = point_cloud[(y * width + x) * 3 + 0];
      double Y = point_cloud[(y * width + x) * 3 + 1];
      double Z = point_cloud[(y * width + x) * 3 + 2];
      double depth = sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2));
      //double depth = abs(Y);
      //cout << depth << '\t';
      disp_real.at<double>(y, x) = 1.0 / depth;
      disp_uchar.at<uchar>(y, x) = max(min(300 / depth, 255.0), 0.0);
      //cout << depth << endl;
    }
  }
  return disp_uchar;
  normalize(disp_real, disp_real);
  Mat disp;
  disp_real.convertTo(disp, CV_8UC3, 25600 * 5);
  return disp;
}


/*************Planes*************/
vector<double> fitPlane(const vector<double> &points, double &error_per_pixel)
{
    int num_points = points.size() / 3;
    if (num_points < 3) {
        error_per_pixel = 1000000;
        return vector<double>();
    }
    
    //    CMatrix<double> A(num_points, 3);
    //    vector<double> b(num_points, 1);
    //    for (int i = 0; i < num_points; i++)
    //        for (int c = 0; c < 3; c++)
    //            A.set(i, c, points[i * 3 + c]);
    //
    //    vector<double> plane = solveLinearEquations(A, b);
    
//    CvMat *A = cvCreateMat(num_points, 3, CV_32FC1);
//    CvMat *b = cvCreateMat(num_points, 1, CV_32FC1);
//    for (int i = 0; i < num_points; i++) {
//        for (int c = 0; c < 3; c++)
//            cvmSet(A, i, c, points[i * 3 + c]);
//        cvmSet(b, i, 0, 1);
//    }
//    CvMat *result = cvCreateMat(3, 1, CV_32FC1);
//    cvSolve(A, b, result);
//    vector<double> plane(3);
//    for (int c = 0; c < 3; c++)
//        plane[c] = cvmGet(result, c, 0);
    
//    MatrixXf A(num_points, 3);
//    VectorXf b(num_points);
//    for (int i = 0; i < num_points; i++) {
//        for (int c = 0; c < 3; c++)
//            A(i, c) = points[i * 3 + c];
//        b(i) = 1;
//    }
//    Vector3f result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
//    vector<double> plane(3);
//    for (int c = 0; c < 3; c++)
//        plane[c] = result(c);
//    
//    double norm = 0;
//    for (int c = 0; c < 3; c++)
//        norm += pow(plane[c], 2);
//    norm = sqrt(norm);
//    for (int c = 0; c < 3; c++)
//        plane[c] /= norm;
//    plane.push_back(1 / norm);
    
    
    vector<double> center(3, 0);
    for (int i = 0; i < num_points; i++)
        for (int c = 0; c < 3; c++)
            center[c] += points[i * 3 + c];
    for (int c = 0; c < 3; c++)
        center[c] /= num_points;
    MatrixXf A(3, num_points);
    for (int i = 0; i < num_points; i++)
        for (int c = 0; c < 3; c++)
            A(c, i) = points[i * 3 + c] - center[c];
    
    JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeThinV);
//    MatrixXf S = svd.singularValues();
//    cout << S << endl;
    MatrixXf U = svd.matrixU();
//    cout << U << endl;
    vector<double> plane(4, 0);
    for (int c = 0; c < 3; c++)
        plane[c] = U(c, 2);
    for (int c = 0; c < 3; c++)
        plane[3] += plane[c] * center[c];
    
    
//    vector<double> plane(4, 0);
//    ceres::Problem problem;
//    for (int i = 0; i < num_points; ++i) {
//        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<PlaneFittingResidual, 1, 1, 1, 1, 1>(new PlaneFittingResidual(points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]));
//        problem.AddResidualBlock(cost_function, NULL, &plane[0], &plane[1], &plane[2], &plane[3]);
//    }
//    
//    ceres::Solver::Options options;
//    options.max_num_iterations = 25;
//    options.linear_solver_type = ceres::DENSE_QR;
//    options.minimizer_progress_to_stdout = true;
//    
//    ceres::Solver::Summary summary;
//    Solve(options, &problem, &summary);
    
    
    double error = 0;
    for (int i = 0; i < num_points; i++)
        error += abs(points[i * 3 + 0] * plane[0] + points[i * 3 + 1] * plane[1] + points[i * 3 + 2] * plane[2] - plane[3]);
    error_per_pixel = error / num_points;
    return plane;
}

vector<double> fitPlaneRobust(const vector<double> &points, const double plane_error_threshold)
{
    srand(time(NULL));
    const int NUM_POINTS = points.size() / 3;
    if (NUM_POINTS < 3)
        return vector<double>();
    vector<double> max_num_inliers_plane;
    int max_num_inliers = 0;
    for (int iteration = 0; iteration < 50; iteration++) {
        set<int> three_indices;
        while (three_indices.size() < 3)
            three_indices.insert(rand() % NUM_POINTS);
        vector<double> three_points(9);
        int three_point_index = 0;
        for (set<int>::const_iterator index_it = three_indices.begin(); index_it != three_indices.end(); index_it++) {
            for (int c = 0; c < 3; c++)
                three_points[three_point_index * 3 + c] = points[*index_it * 3 + c];
            three_point_index++;
        }
        double dummy;
        vector<double> plane = fitPlane(three_points, dummy);
        
        int num_inliers = 0;
        for (int i = 0; i < NUM_POINTS; i++) {
            double error = plane[3];
            for (int c = 0; c < 3; c++) {
                error -= points[i * 3 + c] * plane[c];
            }
            if (abs(error) < plane_error_threshold)
                num_inliers++;
        }
        if (num_inliers > max_num_inliers) {
            max_num_inliers_plane = plane;
            max_num_inliers = num_inliers;
        }
    }
    if (max_num_inliers == NUM_POINTS)
        return fitPlaneRobust(points, plane_error_threshold / 2);
    else if (max_num_inliers < NUM_POINTS / 2)
        return fitPlaneRobust(points, plane_error_threshold * 2);
    else
        return max_num_inliers_plane;
}

vector<double> calcCenter(const vector<double> &points)
{
    const int NUM_POINTS = points.size() / 3;
    vector<double> center(3, 0);
    for (int i = 0; i < NUM_POINTS; i++)
        for (int c = 0; c < 3; c++)
            center[c] += points[i * 3 + c];
    for (int c = 0; c < 3; c++)
        center[c] /= NUM_POINTS;
    return center;
}

vector<double> calcRange(const vector<double> &points)
{
    const int NUM_POINTS = points.size() / 3;
    double min_x = 1000000, max_x = -1000000, min_y = 1000000, max_y = -1000000, min_z = 1000000, max_z = -1000000;
    for (int i = 0; i < NUM_POINTS; i++) {
        double x = points[i * 3 + 0];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        if (x < min_x)
            min_x = x;
        if (x > max_x)
            max_x = x;
        if (y < min_y)
            min_y = y;
        if (y > max_y)
            max_y = y;
        if (z < min_z)
            min_z = z;
        if (z > max_z)
            max_z = z;
    }
    vector<double> range;
    range.push_back(min_x);
    range.push_back(max_x);
    range.push_back(min_y);
    range.push_back(max_y);
    range.push_back(min_z);
    range.push_back(max_z);
    return range;
}

bool checkInRange(const vector<double> &point, const vector<double> &range)
{
    if (point[0] < range[0] || point[0] > range[1] || point[1] < range[2] || point[1] > range[3] || point[2] < range[4] || point[2] > range[5])
        return false;
    else
        return true;
}

vector<double> calcCameraParameters(const vector<double> &point_cloud, const int width, const bool use_panorama)
{
    cout << "calculate camera parameters...\t";
    const int height = point_cloud.size() / 3 / width;
    
    MatrixXf A_x(width * height, 2);
    VectorXf b_x(width * height, 1);
    MatrixXf A_y(width * height, 2);
    MatrixXf b_y(width * height, 1);
    for (int i = 0; i < width * height; i++) {
        int x_2D = i % width;
        int y_2D = i / width;
        double x_3D = point_cloud[i * 3 + 0];
        double y_3D = point_cloud[i * 3 + 1];
        double z_3D = point_cloud[i * 3 + 2];
        
        if (z_3D == 0)
            bool wait = true;
        A_x(i, 0) = x_3D / z_3D;
        A_x(i, 1) = 1;
        b_x(i) = x_2D;

        A_y(i, 0) = y_3D / z_3D;
        A_y(i, 1) = 1;
        b_y(i) = y_2D;
    }
    VectorXf parameters_x = A_x.jacobiSvd(ComputeThinU | ComputeThinV).solve(b_x);
    double f_x = parameters_x[0];
    double c_x = parameters_x[1];
    
    VectorXf parameters_y = A_y.jacobiSvd(ComputeThinU | ComputeThinV).solve(b_y);
    double f_y = parameters_y[0];
    double c_y = parameters_y[1];
//
    vector<double> parameters;
    parameters.push_back(f_x);
    parameters.push_back(c_x);
    parameters.push_back(f_y);
    parameters.push_back(c_y);
    cout << "done" << endl;
    return parameters;
}

vector<double> calc3DPointOnPlane(const double image_x, const double image_y, const vector<double> &surface_model, const vector<double> &camera_parameters)
{
    Matrix3f A;
    Vector3f b;
    for (int c = 0; c < 3; c++)
      A(0, c) = surface_model[c];
    A(1, 0) = camera_parameters[0];
    A(1, 1) = 0;
    A(1, 2) = (camera_parameters[1] - image_x);
    A(2, 0) = 0;
    A(2, 1) = camera_parameters[2];
    A(2, 2) = (camera_parameters[3] - image_y);
    b(0) = surface_model[3];
    b(1) = 0;
    b(2) = 0;
    Vector3f result = A.colPivHouseholderQr().solve(b);
    vector<double> point_3D(3);
    for (int i = 0; i < 3; i++)
        point_3D[i] = result(i);
    return point_3D;
}

vector<double> calc3DPointOnImage(const double image_x, const double image_y, const vector<double> &camera_parameters)
{
    vector<double> point_3D(3);
    point_3D[0] = (image_x - camera_parameters[1]) / camera_parameters[0];
    point_3D[1] = (image_y - camera_parameters[3]) / camera_parameters[2];
    point_3D[2] = 1;
    return point_3D;
}

double calcPlaneDistance(const double image_x, const double image_y, const vector<double> &surface_model_1, const vector<double> &surface_model_2, const vector<double> &camera_parameters)
{
    vector<double> point_3D_1 = calc3DPointOnPlane(image_x, image_y, surface_model_1, camera_parameters);
    vector<double> point_3D_2 = calc3DPointOnPlane(image_x, image_y, surface_model_2, camera_parameters);
    double distance = sqrt(pow(point_3D_1[0] - point_3D_2[0], 2) + pow(point_3D_1[1] - point_3D_2[1], 2) + pow(point_3D_1[2] - point_3D_2[2], 2));
    return distance;
}

int findPlaneRelation(const vector<double> &points, const vector<double> &constraint_surface_model, const double opposite_distance_threshold, const double num_outliers_threshold_ratio)
{
  vector<double> constraint_plane = constraint_surface_model;
    int num_outliers_1 = 0, num_outliers_2 = 0;
    const int NUM_POINTS = points.size() / 3;
    for (int i = 0; i < NUM_POINTS; i++) {
        vector<double> point(points.begin() + i * 3, points.begin() + (i + 1) * 3);
        double distance = point[0] * constraint_plane[0] + point[1] * constraint_plane[1] + point[2] * constraint_plane[2] - constraint_plane[3];
        if (constraint_plane[3] < 0)
            distance *= -1;
        if (distance < -opposite_distance_threshold)
            num_outliers_1++;
        if (distance > opposite_distance_threshold)
            num_outliers_2++;
    }
    const int NUM_OUTLIERS_THRESHOLD = static_cast<int>(NUM_POINTS * num_outliers_threshold_ratio + 0.5);
    int relation = -1;
    if (num_outliers_1 <= NUM_OUTLIERS_THRESHOLD)
        relation = 1;
    else if (num_outliers_2 <= NUM_OUTLIERS_THRESHOLD)
        relation = 0;
    else
        relation = -1;
    return relation;
}

bool checkRelationValid(const vector<double> &point, const vector<double> &constraint_surface_model, const int relation, const double opposite_distance_threshold)
{
    //no relation constraint if relation == -1
    if (relation == -1)
        return true;
    vector<double> constraint_plane = constraint_surface_model;
    double distance = point[0] * constraint_plane[0] + point[1] * constraint_plane[1] + point[2] * constraint_plane[2] - constraint_plane[3];
    if (constraint_plane[3] < 0)
        distance *= -1;
    if (distance < -opposite_distance_threshold && relation == 1)
        return false;
    if (distance > opposite_distance_threshold && relation == 0)
        return false;
    return true;
}

vector<int> findOutmostSurfaces(const vector<vector<double> > &ranges, const vector<double> &extreme_values, const vector<vector<double> > &id_surface_models)
{
    vector<int> outmost_surfaces;
    for (int i = 0; i < ranges.size(); i++) {
        vector<double> range = ranges[i];
        for (int c = 0; c < 3; c++) {
            if ((c != 2 && range[c * 2 + 0] < extreme_values[c * 2 + 0]) || range[c * 2 + 1] > extreme_values[c * 2 + 1]) {
                vector<double> plane = id_surface_models[i];
                if ((c == 0 && abs(plane[1]) < abs(plane[0]) && abs(plane[2]) < abs(plane[0])) || (c == 1 && abs(plane[2]) < abs(plane[1]) && abs(plane[0]) < abs(plane[1])) || (c == 2 && abs(plane[0]) < abs(plane[2]) && abs(plane[1]) < abs(plane[2])))
                    outmost_surfaces.push_back(i);
//                double extreme_direction_range = range[c * 2 + 1] - range[c * 2 + 0];
//                bool horizontal = true;
//                for (int direction = 0; direction < 3; direction++)
//                    if (direction != c && range[direction * 2 + 1] - range[direction * 2 + 0] < extreme_direction_range)
//                        horizontal = false;
//                if (horizontal == true)
//                    outmost_surfaces.push_back(i);
            }
        }
    }
    return outmost_surfaces;
}

vector<double> calcExtremeValues(const vector<double> &point_cloud)
{
    const int NUM_POINTS = point_cloud.size() / 3;
    vector<double> xs(NUM_POINTS), ys(NUM_POINTS), zs(NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) {
        xs[i] = point_cloud[i * 3 + 0];
        ys[i] = point_cloud[i * 3 + 1];
        zs[i] = point_cloud[i * 3 + 2];
    }
    
    vector<double> extreme_values(6);
    const int NUM_OUTLIERS = NUM_POINTS * 0.01;

    nth_element(xs.begin(), xs.begin() + NUM_OUTLIERS, xs.end());
    extreme_values[0] = xs[NUM_OUTLIERS];
    nth_element(xs.begin(), xs.begin() + NUM_POINTS - 1 - NUM_OUTLIERS, xs.end());
    extreme_values[1] = xs[NUM_POINTS - 1 - NUM_OUTLIERS];
    
    nth_element(ys.begin(), ys.begin() + NUM_OUTLIERS, ys.end());
    extreme_values[2] = ys[NUM_OUTLIERS];
    nth_element(ys.begin(), ys.begin() + NUM_POINTS - 1 - NUM_OUTLIERS, ys.end());
    extreme_values[3] = ys[NUM_POINTS - 1 - NUM_OUTLIERS];
    
    nth_element(zs.begin(), zs.begin() + NUM_OUTLIERS, zs.end());
    extreme_values[4] = zs[NUM_OUTLIERS];
    nth_element(zs.begin(), zs.begin() + NUM_POINTS - 1 - NUM_OUTLIERS, zs.end());
    extreme_values[5] = zs[NUM_POINTS - 1 - NUM_OUTLIERS];
    
    return extreme_values;
}

map<int, int> calcSurfaceObjectMap(const Mat &surface_id_image, const Mat &label_image)
{
    const int WIDTH = surface_id_image.cols;
    const int HEIGHT = surface_id_image.rows;
    map<int, map<int, int> > surface_object_counter;
    for (int y = 0; y < HEIGHT; y++) {
      const uchar *surface_id_data = surface_id_image.ptr<uchar>(y);
      const uchar *label_data = label_image.ptr<uchar>(y);
        for (int x = 0; x < WIDTH; x++) {
            int surface_id = (surface_id_data[x * 3 + 0] + surface_id_data[x * 3 + 1] + surface_id_data[x * 3 + 2]) / 3;
            int label = label_data[x];
            surface_object_counter[surface_id][label]++;
        }
    }
    map<int, int> surface_object_map;
    for (map<int, map<int, int> >::const_iterator surface_it = surface_object_counter.begin(); surface_it != surface_object_counter.end(); surface_it++) {
        int max_num = 0;
        int max_num_object = -1;
        for (map<int, int>::const_iterator object_it = surface_it->second.begin(); object_it != surface_it->second.end(); object_it++) {
            if (object_it->second > max_num) {
                max_num_object = object_it->first;
                max_num = object_it->second;
            }
        }
        surface_object_map[surface_it->first] = max_num_object;
    }
    return surface_object_map;
}

vector<double> normalizePointCloudByZ(const vector<double> &point_cloud)
{
  vector<double> extreme_values(6);
  for (int c = 0; c < 3; c++) {
    extreme_values[c * 2 + 0] = 1000000;
    extreme_values[c * 2 + 1] = -1000000;
  }
  for (int i = 0; i < point_cloud.size() / 3; i++) {
    for (int c = 0; c < 3; c++) {
      double value = point_cloud[i * 3 + c];
      if (value < extreme_values[c * 2 + 0])
	extreme_values[c * 2 + 0] = value;
      if (value > extreme_values[c * 2 + 1])
	extreme_values[c * 2 + 1] = value;
    }
  }
  vector<double> new_point_cloud = point_cloud;
  for (int i = 0; i < point_cloud.size() / 3; i++)
    for (int c = 0; c < 3; c++)
      new_point_cloud[i * 3 + c] = new_point_cloud[i * 3 + c] / (extreme_values[5] - extreme_values[4]);
  return new_point_cloud;
}

void cropRegion(Mat &image, vector<double> &point_cloud, vector<int> &segmentation, const int start_x, const int start_y, const int new_width, const int new_height)
{
  int ori_width = image.cols;
  Mat cropped_image;
  Mat(image, Rect(start_x, start_y, new_width, new_height)).copyTo(cropped_image);
  vector<double> new_point_cloud(new_width * new_height * 3);
  vector<int> new_segmentation(new_width * new_height);
  for (int y = start_y; y < start_y + new_height; y++) {
    for (int x = start_x; x < start_x + new_width; x++) {
      int new_index = (y - start_y) * new_width + (x - start_x);
      int ori_index = y * ori_width + x;
      new_segmentation[new_index] = segmentation[ori_index];
      for (int c = 0; c < 3; c++)
        new_point_cloud[new_index * 3 + c] = point_cloud[ori_index * 3 + c];
    }
  }
  image = cropped_image;
  point_cloud = new_point_cloud;
  segmentation = new_segmentation;
  int new_index = 0;
  map<int, int> index_map;
  for (int i = 0; i < segmentation.size(); i++) {
    int ori_segment_id = segmentation[i];
    if (index_map.count(ori_segment_id) == 0)
      index_map[ori_segment_id] = new_index++;
    new_segmentation[i] = index_map[ori_segment_id];
  }
  segmentation = new_segmentation;
}

void zoomScene(Mat &image, vector<double> &point_cloud, const double scale_x, const double scale_y)
{
  int ori_width = image.cols;
  int ori_height = image.rows;
  int new_width = static_cast<int>(ori_width * scale_x + 0.5);
  int new_height = static_cast<int>(ori_height * scale_y + 0.5);
  Mat zoomed_image;
  resize(image, zoomed_image, Size(new_width, new_height), 0, 0, INTER_AREA);
  vector<double> new_point_cloud(new_width * new_height * 3);
  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      int new_index = y * new_width + x;
      int ori_index = max(min(static_cast<int>(y / scale_y + 0.5), ori_height - 1), 0) * ori_width + max(min(static_cast<int>(x / scale_x + 0.5), ori_width - 1), 0);
      for (int c = 0; c < 3; c++)
        new_point_cloud[new_index * 3 + c] = point_cloud[ori_index * 3 + c];
    }
  }
  image = zoomed_image;
  point_cloud = new_point_cloud;
  
  // int new_index = 0;
  // map<int, int> index_map;
  // for (int i = 0; i < segmentation.size(); i++) {
  //   int ori_segment_id = segmentation[i];
  //   if (index_map.count(ori_segment_id) == 0)
  //     index_map[ori_segment_id] = new_index++;
  //   new_segmentation[i] = index_map[ori_segment_id];
  // }
  // segmentation = new_segmentation;
}

void cropScene(Mat &image, vector<double> &point_cloud, const int start_x, const int start_y, const int end_x, const int end_y)
{
  int ori_width = image.cols;
  int ori_height = image.rows;
  int new_width = end_x - start_x + 1;
  int new_height = end_y - start_y + 1;
  Mat cropped_image;
  Mat(image, Rect(start_x, start_y, new_width, new_height)).copyTo(cropped_image);
  vector<double> cropped_point_cloud(new_width * new_height * 3);
  for (int y = 0; y < new_height; y++) {
    for (int x = 0; x < new_width; x++) {
      int new_index = y * new_width + x;
      int ori_index = (y + start_y) * ori_width + (x + start_x);
      for (int c = 0; c < 3; c++)
        cropped_point_cloud[new_index * 3 + c] = point_cloud[ori_index * 3 + c];
    }
  }
  image = cropped_image;
  point_cloud = cropped_point_cloud;
  
  // int new_index = 0;
  // map<int, int> index_map;
  // for (int i = 0; i < segmentation.size(); i++) {
  //   int ori_segment_id = segmentation[i];
  //   if (index_map.count(ori_segment_id) == 0)
  //     index_map[ori_segment_id] = new_index++;
  //   new_segmentation[i] = index_map[ori_segment_id];
  // }
  // segmentation = new_segmentation;
}

void drawMaskImage(const vector<bool> &mask, const int width, const int height, const string filename)
{
  Mat mask_image = Mat::zeros(height, width, CV_8UC1);
  for (int y = 0; y < height; y++) {
    uchar *data = mask_image.ptr<uchar>(y);
    for (int x = 0; x < width; x++) {
      int index = y * width + x;
      if (mask[index] == true)
	data[x] = 255;
    }
  }
  imwrite(filename, mask_image);
}

vector<vector<int> > getCombinations(const vector<int> &candidates, const int num_elements)
{
  if (num_elements == 0)
    return vector<vector<int> >(1, vector<int>());
  vector<vector<int> > combinations;
  int num_candidates = candidates.size();
  if (num_candidates < num_elements)
    return combinations;

  vector<bool> selected_element_mask(num_candidates, false);
  for (int index = num_candidates - num_elements; index < num_candidates; index++)
    selected_element_mask[index] = true;
  while (true) {
    vector<int> combination;
    for (int index = 0; index < num_candidates; index++) {
      if (selected_element_mask[index] == true) {
	combination.push_back(candidates[index]);
      }
    }
    combinations.push_back(combination);
    if (next_permutation(selected_element_mask.begin(), selected_element_mask.end()) == false)
      break;
  }
  return combinations;
  
  for (int configuration = 0; configuration < pow(2, num_candidates); configuration++) {
    vector<bool> selected_element_mask(num_candidates, false);
    int num_selected_elements = 0;
    int configuration_temp = configuration;
    for (int j = 0; j < num_candidates; j++) {
      if (configuration_temp % 2 == 1) {
	selected_element_mask[j] = true;
	num_selected_elements++;
	if (num_selected_elements > num_elements)
	  break;
      }
      configuration_temp /= 2;
    }
    if (num_selected_elements != num_elements)
      continue;
    vector<int> combination;
    for (int j = 0; j < num_candidates; j++)
      if (selected_element_mask[j] == true)
	combination.push_back(candidates[j]);
    combinations.push_back(combination);
  }
  return combinations;
}

vector<vector<int> > fillWithNewElement(const vector<int> &current_values, const int new_element, const int num_elements)
{
  vector<vector<int> > combinations;
  if (current_values.size() > num_elements)
    return combinations;
  else if (current_values.size() == num_elements) {
    combinations.push_back(current_values);
    return combinations;
  } else {
    for (int configuration = 0; configuration < pow(2, num_elements); configuration++) {
      vector<bool> selected_position_mask(num_elements, false);
      int num_selected_positions = 0;
      int configuration_temp = configuration;
      for (int j = 0; j < num_elements; j++) {
        if (configuration_temp % 2 == 1) {
          selected_position_mask[j] = true;
          num_selected_positions++;
          if (num_selected_positions > current_values.size())
            break;
        }
        configuration_temp /= 2;
      }
      if (num_selected_positions != current_values.size())
        continue;
      vector<int> combination(num_elements, new_element);
      int index = 0;
      for (int j = 0; j < num_elements; j++)
        if (selected_position_mask[j] == true)
          combination[j] = current_values[index++];
      combinations.push_back(combination);
    }
  }  
}

vector<int> calcDistanceToBoundaries(const vector<int> segmentation, const int image_width, const int max_distance)
{
  const int NUM_PIXELS = segmentation.size();
  if (max_distance < 0)
    return vector<int>(NUM_PIXELS, -1);
  
  const int image_height = NUM_PIXELS / image_width;
  vector<int> distance_to_boundaries(segmentation.size(), -1);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int segment_id = segmentation[pixel];
    if (distance_to_boundaries[pixel] != -1)
      continue;
    vector<int> neighbor_pixels;
    int x = pixel % image_width;
    int y = pixel / image_width;
    if (x < image_width - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < image_height - 1)
      neighbor_pixels.push_back(pixel + image_width);
    if (x > 0 && y < image_height - 1)
      neighbor_pixels.push_back(pixel - 1 + image_width);
    if (x < image_width - 1 && y < image_height - 1)
      neighbor_pixels.push_back(pixel + 1 + image_width);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      int neighbor_segment_id = segmentation[neighbor_pixel];
      if (neighbor_segment_id != segment_id) {
        distance_to_boundaries[pixel] = 0;
        distance_to_boundaries[neighbor_pixel] = 0;
      }
    }
  }
  for (int distance = 0; distance < max_distance; distance++) {
    vector<int> border_pixels;
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      if (distance_to_boundaries[pixel] != distance)
        continue;
      vector<int> neighbor_pixels;
      int x = pixel % image_width;
      int y = pixel / image_width;
      if (x > 0)
        neighbor_pixels.push_back(pixel - 1);
      if (x < image_width - 1)
        neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
        neighbor_pixels.push_back(pixel - image_width);
      if (y < image_height - 1)
        neighbor_pixels.push_back(pixel + image_width);
      if (x > 0 && y > 0)
        neighbor_pixels.push_back(pixel - 1 - image_width);
      if (x > 0 && y < image_height - 1)
        neighbor_pixels.push_back(pixel - 1 + image_width);
      if (x < image_width - 1 && y > 0)
        neighbor_pixels.push_back(pixel + 1 - image_width);
      if (x < image_width - 1 && y < image_height - 1)
        neighbor_pixels.push_back(pixel + 1 + image_width);

      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        int neighbor_pixel = *neighbor_pixel_it;
        if (distance_to_boundaries[neighbor_pixel] == -1)
          border_pixels.push_back(neighbor_pixel);
      }
    }
    for (int i = 0; i < border_pixels.size(); i++)
      distance_to_boundaries[border_pixels[i]] = distance + 1;
  }

  for (int x = 0; x < image_width; x++) {
    for (int i = 0; i < max_distance; i++) {
      int pixel_1 = i * image_width + x;
      if (distance_to_boundaries[pixel_1] == -1)
	distance_to_boundaries[pixel_1] = i + 1;
      int pixel_2 = (image_height - 1 - i) * image_width + x;
      if (distance_to_boundaries[pixel_2] == -1)
        distance_to_boundaries[pixel_2] = i + 1;
    }
  }
  for (int y = 0; y < image_height; y++) {
    for (int i = 0; i < max_distance; i++) {
      int pixel_1 = y * image_width + i;
      if (distance_to_boundaries[pixel_1] == -1)
        distance_to_boundaries[pixel_1] = i + 1;
      int pixel_2 = y * image_width + (image_width - 1 - i);
      if (distance_to_boundaries[pixel_2] == -1)
        distance_to_boundaries[pixel_2] = i + 1;
    }
  }
  return distance_to_boundaries;
}

Mat drawArrayImage(const vector<double> &array, const int width, const int scale)
{
  Mat image(array.size() / width, width, CV_8UC1);
  for (int i = 0; i < array.size(); i++) {
    int x = i % width;
    int y = i / width;
    int value = array[i] * scale + 0.5;
    image.at<uchar>(y, x) = value;
  }
  return image;
}


map<int, int> calcSurfaceColors(const Mat &image, const vector<int> &segmentation)
{
  map<int, int> surface_sum_r;
  map<int, int> surface_sum_g;
  map<int, int> surface_sum_b;
  map<int, int> surface_num_pixels_counter;
  for (int y = 0; y < image.rows; y++) {
    for (int x = 0; x < image.cols; x++) {
      int segment_id = segmentation[y * image.cols + x];
      Vec3b color = image.at<Vec3b>(y, x);
      surface_sum_b[segment_id] += color[0];
      surface_sum_g[segment_id] += color[1];
      surface_sum_r[segment_id] += color[2];
      surface_num_pixels_counter[segment_id]++;
    }
  }
  map<int, int> surface_colors;
  for (map<int, int>::const_iterator surface_it = surface_num_pixels_counter.begin(); surface_it != surface_num_pixels_counter.end(); surface_it++) {
    int b = surface_sum_b[surface_it->first] / surface_it->second;
    int g = surface_sum_g[surface_it->first] / surface_it->second;
    int r = surface_sum_r[surface_it->first] / surface_it->second;
    surface_colors[surface_it->first] = r * 256 * 256 + g * 256 + b;
  }
  return surface_colors;
}

vector<double> normalizeValues(const vector<double> &values, const double range)
{
  double sum = 0;
  double sum2 = 0;
  for (int index = 0; index < values.size(); index++) {
    sum += values[index];
    sum2 += pow(values[index], 2);
  }
  double mean = sum / values.size();
  double svar = sqrt(sum2 / values.size() - pow(mean, 2));
  vector<double> normalized_values(values.size());
  for (int index = 0; index < values.size(); index++)
    normalized_values[index] = (values[index] - mean) / svar;

  if (range > 0)
    for (int index = 0; index < values.size(); index++)
      normalized_values[index] = max(min(normalized_values[index], range), -range) / range;
  return normalized_values;
}

void calcStatistics(const vector<double> &values, double &mean, double &svar)
{
  double sum = 0, sum2 = 0;
  for (int value_index = 0; value_index < values.size(); value_index++) {
    double value = values[value_index];
    sum += value;
    sum2 += pow(value, 2);
  }
  mean = sum / values.size();
  svar = sqrt(sum2 / values.size() - pow(mean, 2));
}

void writeSurfaceDepthsImage(const vector<int> &segmentation, const map<int, vector<double> > &surface_depths, const int image_width, const int image_height, const string filename)
{
  Mat surface_depths_image(image_height, image_width, CV_8UC1);
  for (int y = 0; y < image_height; y++) {
    for (int x = 0; x < image_width; x++) {
      int segment_id = segmentation[y * image_width + x];
      double depth = surface_depths.at(segment_id)[y * image_width + x];
      surface_depths_image.at<uchar>(y, x) = 100 / depth;
    }
  }
  imwrite(filename, surface_depths_image);
}

void writeDispImageFromSegments(const vector<int> &labels, const int num_surfaces, const map<int, Segment> &segments, const int num_layers, const int image_width, const int image_height, const string filename)
{
  Mat disp_image(image_height, image_width, CV_8UC1);
  for (int y = 0; y < image_height; y++) {
    for (int x = 0; x < image_width; x++) {
      int pixel = y * image_width + x;
      int label = labels[pixel];
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
	int surface_id = label / static_cast<int>(pow(num_surfaces + 1, num_layers - 1 - layer_index)) % (num_surfaces + 1);
	if (surface_id != num_surfaces) {
	  double depth = segments.at(surface_id).getDepth(pixel);
	  disp_image.at<uchar>(y, x) = 100 / depth;
	  break;
	}
      }
    }
  }
  imwrite(filename, disp_image);
}

double normalizeStatistically(const double value, const double mean, const double svar, const double normalized_value_for_mean, const double scale_factor)
{
  double normalized_value = normalized_value_for_mean + (value - mean) / svar * scale_factor;
  normalized_value = max(min(normalized_value, 1.0), 0.0);
  return normalized_value;
}

int calcNumOneBits(const int value)
{
  int num_one_bits = 0;
  int current_value = value;
  while (current_value > 0) {
    int bit_value = current_value % 2;
    num_one_bits += bit_value;
    current_value /= 2;
  }
  return num_one_bits;
}

vector<double> readPointCloudFromObj(const string filename, const int image_width, const int image_height, const double rotation_angle)
{
  //  double max_depth = 0;
  vector<double> point_cloud(image_width * image_height * 3);
  ifstream in_str(filename);
  for (int pixel = 0; pixel < image_width * image_height; pixel++) {
    vector<double> point(3);
    char dummy_char;
    double dummy;
    in_str >> dummy_char >> point[0] >> point[1] >> point[2] >> dummy >> dummy >> dummy;
    if (abs(point[0]) < 0.000001 && abs(point[1]) < 0.000001 && abs(point[2]) < 0.000001) {
      point_cloud[pixel * 3 + 0] = 0;
      point_cloud[pixel * 3 + 1] = 0;
      point_cloud[pixel * 3 + 2] = 0;
      continue;
    }
    
    double X = -point[1];
    double Y = -point[2];
    double Z = point[0];

    double new_X = X * cos(rotation_angle) + Z * sin(rotation_angle);
    double new_Y = Y;
    double new_Z = Z * cos(rotation_angle) - X * sin(rotation_angle);
    
    point_cloud[pixel * 3 + 0] = new_X;
    point_cloud[pixel * 3 + 1] = new_Y;
    point_cloud[pixel * 3 + 2] = new_Z;

    // if (point[0] > max_depth)
    //   max_depth = point[0];
    //cout << -point[0] << endl;
  }
  // cout << max_depth << endl;
  // exit(1);
  in_str.close();
  return point_cloud;
}

vector<double> inpaintPointCloud(const vector<double> &point_cloud, const int image_width, const int image_height)
{
  const double DATA_WEIGHT = 10000;
  const int WINDOW_SIZE = 3;
  
  SparseMatrix<double> A(image_width * image_height, image_width * image_height);
  VectorXd b(image_width * image_height);
  
  vector<Eigen::Triplet<double> > triplets;
  
  for (int y = 0; y < image_height; ++y) {
    for (int x = 0; x < image_width; ++x) {
      int pixel = y * image_width + x;
      int count = 0;
      for (int delta_y = -WINDOW_SIZE / 2; delta_y <= WINDOW_SIZE / 2; ++delta_y) {
        for (int delta_x = -WINDOW_SIZE / 2; delta_x <= WINDOW_SIZE / 2; ++delta_x) {
          if (x + delta_x < 0 || x + delta_x >= image_width || y + delta_y < 0 || y + delta_y >= image_height)
            continue;
          if (delta_x == 0 && delta_y == 0)
            continue;
          triplets.push_back(Eigen::Triplet<double>(pixel, pixel + delta_y * image_width + delta_x, -1));
          ++count;
        }
      }
      
      if (checkPointValidity(getPoint(point_cloud, pixel)) == true) {
	triplets.push_back(Eigen::Triplet<double>(pixel, pixel, DATA_WEIGHT + count));
	b[pixel] = DATA_WEIGHT * point_cloud[pixel * 3 + 2];
      } else {
        triplets.push_back(Eigen::Triplet<double>(pixel, pixel, count));
        b[pixel] = 0;
      }
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());
  SimplicialCholesky<SparseMatrix<double> > chol(A);
  VectorXd solution = chol.solve(b);
  
  vector<double> inpainted_point_cloud = point_cloud;
  for (int pixel = 0; pixel < image_width * image_height; ++pixel) {
    if (inpainted_point_cloud[pixel * 3 + 2] <= 0)
      inpainted_point_cloud[pixel * 3 + 2] = solution[pixel];
  }
  return inpainted_point_cloud;
}

bool readPtxFile(const string &filename, Mat &image, vector<double> &point_cloud, vector<double> &camera_parameters)
{
  ifstream in_str(filename);
  if (!in_str)
    return false;
  int image_width, image_height;
  in_str >> image_width >> image_height;
  image = Mat(image_height, image_width, CV_8UC3);
  point_cloud.assign(image_width * image_height * 3, 0);
  double temp;
  for (int c = 0; c < 3; c++)
    for (int r = 0; r < 4; r++)
      in_str >> temp;
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < 4; r++)
      in_str >> temp;
  for (int pixel = 0; pixel < image_width * image_height; pixel++) {
    double X, Y, Z, confidence;
    int R, G, B;
    in_str >> X >> Y >> Z >> confidence;
    //cout << X << '\t' << Y << '\t' << Z << endl;
    in_str >> R >> G >> B;
    int traversed_pixel = (image_height - 1 - pixel % image_height) * image_width + pixel / image_height;
    point_cloud[traversed_pixel * 3 + 0] = X;
    point_cloud[traversed_pixel * 3 + 1] = Y;
    point_cloud[traversed_pixel * 3 + 2] = Z;
    image.at<Vec3b>(traversed_pixel / image_width, traversed_pixel % image_width) = Vec3b(B, G, R);
    //image.at<Vec3b>(pixel / image_width, pixel % image_width) = Vec3b(B, G, R);
  }
  in_str.close();
  return true;
}


vector<double> unprojectPixel(const int pixel, const double depth, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA)
{
  int x = pixel % IMAGE_WIDTH;
  int y = pixel / IMAGE_WIDTH;
  if (USE_PANORAMA == false) {
    double X_Z_ratio = (x - CAMERA_PARAMETERS[1]) / CAMERA_PARAMETERS[0];
    double Y_Z_ratio = (y - CAMERA_PARAMETERS[2]) / CAMERA_PARAMETERS[0];
    double Z = depth / sqrt(1 + pow(X_Z_ratio, 2) + pow(Y_Z_ratio, 2));
    double X = Z * X_Z_ratio;
    double Y = Z * Y_Z_ratio;
    vector<double> point(3);
    point[0] = X;
    point[1] = Y;
    point[2] = Z;
    return point;
  } else {
    double angle_1 = -M_PI * (y - CAMERA_PARAMETERS[2]) / CAMERA_PARAMETERS[0];
    double angle_2 = (2 * M_PI) * (x - CAMERA_PARAMETERS[1]) / IMAGE_WIDTH;
    double Z = depth * sin(angle_1);
    double X = depth * cos(angle_1) * sin(angle_2);
    double Y = depth * cos(angle_1) * cos(angle_2);
    vector<double> point(3);
    point[0] = X;
    point[1] = Y;
    point[2] = Z;
    return point;
  }
}

int projectPoint(const vector<double> &point, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA)
{
  if (calcNorm(point) < 0.000001)
    return -1;
  if (USE_PANORAMA == false) {
    int x = round(point[0] / point[2] * CAMERA_PARAMETERS[0] + CAMERA_PARAMETERS[1]);
    int y = round(point[1] / point[2] * CAMERA_PARAMETERS[0] + CAMERA_PARAMETERS[2]);
    if (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT)
      return -1;
    else
      return y * IMAGE_WIDTH + x;
  } else {
    double depth = sqrt(pow(point[0], 2) + pow(point[1], 2));
    double angle_1 = atan(point[2] / depth);
    double angle_2 = atan2(point[0], point[1]);
    int x = static_cast<int>(round(angle_2 / (2 * M_PI) * IMAGE_WIDTH + CAMERA_PARAMETERS[1])) % IMAGE_WIDTH;
    int y = round(-angle_1 / (M_PI) * CAMERA_PARAMETERS[0] + CAMERA_PARAMETERS[2]);
    //cout << x << '\t' << y << '\t' << angle_1 << '\t' << angle_2 << endl;
    //exit(1);
    if (x < 0 || x >= IMAGE_WIDTH || y < 0 || y >= IMAGE_HEIGHT)
      return -1;
    else
      return y * IMAGE_WIDTH + x;
  }
}

double calcPlaneDepthAtPixel(const vector<double> &plane, const int pixel, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA)
{
  if (USE_PANORAMA == false) {
    double X_Z_ratio = (pixel % IMAGE_WIDTH - CAMERA_PARAMETERS[1]) / CAMERA_PARAMETERS[0];
    double Y_Z_ratio = (pixel / IMAGE_WIDTH - CAMERA_PARAMETERS[2]) / CAMERA_PARAMETERS[0];
    
    double Z = plane[3] / (plane[0] * X_Z_ratio + plane[1] * Y_Z_ratio + plane[2]);
    if (Z < 0)
      return -1;
    return sqrt(pow(X_Z_ratio * Z, 2) + pow(Y_Z_ratio * Z, 2) + pow(Z, 2));
  } else {
    int x = pixel % IMAGE_WIDTH;
    int y = pixel / IMAGE_WIDTH;
    double angle_1 = -M_PI * (y - CAMERA_PARAMETERS[2]) / CAMERA_PARAMETERS[0];
    double angle_2 = (2 * M_PI) * (x - CAMERA_PARAMETERS[1]) / IMAGE_WIDTH;
    double Z_ratio = sin(angle_1);
    double X_ratio = cos(angle_1) * sin(angle_2);
    double Y_ratio = cos(angle_1) * cos(angle_2);
    if (plane[0] * X_ratio + plane[1] * Y_ratio + plane[2] * Z_ratio >= 0) {
      //cout << pixel << '\t' << X_ratio << '\t' << Y_ratio << '\t' << Z_ratio << endl;
      return -1;
    }
    double depth = plane[3] / (plane[0] * X_ratio + plane[1] * Y_ratio + plane[2] * Z_ratio);
    assert(depth > 0);
    return depth;
  }
}
