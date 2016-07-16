#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

#include "ImageMask.h"
#include "Histogram.h"
//#include "FusionSpaceSolver.h"


namespace cv_utils
{
  //class Histogram<int>;
  //class Histogram<double>;
  //  class ImageMask;
  
  /**********Image Operations**********/
  
  //complete image with holes
  cv::Mat completeImage(const cv::Mat &input_image, const std::vector<bool> &input_source_mask, const std::vector<bool> &input_target_mask, const int WINDOW_SIZE = 5, const Eigen::Matrix3d &unwarp_transform = Eigen::MatrixXd::Identity(3, 3));
  
  //complete image with holes
  cv::Mat completeImage(const cv::Mat &input_image, const ImageMask &input_source_mask, const ImageMask &input_target_mask, const int WINDOW_SIZE = 5, const Eigen::Matrix3d &unwarp_transform = Eigen::MatrixXd::Identity(3, 3));
  
  //complete image with holes using fusion space approace
  cv::Mat completeImageUsingFusionSpace(const cv::Mat &image, const ImageMask &source_mask, const ImageMask &target_mask, const int WINDOW_SIZE);

  //fast computation box integration
  std::vector<double> calcBoxIntegrationMask(const std::vector<double> &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT);
  double calcBoxIntegration(const std::vector<double> &mask, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int x_1, const int y_1, const int x_2, const int y_2);
  
  //calculate means and vars for all windows
  void calcWindowMeansAndVars(const std::vector<double> &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int WINDOW_SIZE, std::vector<double> &means, std::vector<double> &vars);
  void calcWindowMeansAndVars(const std::vector<std::vector<double> > &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int WINDOW_SIZE, std::vector<std::vector<double> > &means, std::vector<std::vector<double> > &vars);
  
  //guided image filtering
  void guidedFilter(const cv::Mat &guidance_image, const cv::Mat &input_image, cv::Mat &output_image, const double radius, const double epsilon);
  
  
  
  /**********Common Operations**********/
  
  //find corresponding pixel on another image
  inline int convertPixel(const int pixel, const int WIDTH, const int HEIGHT, const int NEW_WIDTH, const int NEW_HEIGHT)
  {
    int new_pixel_x = std::min(static_cast<int>(round(1.0 * (pixel % WIDTH) * NEW_WIDTH / WIDTH)), NEW_WIDTH - 1);
    int new_pixel_y = std::min(static_cast<int>(round(1.0 * (pixel / WIDTH) * NEW_HEIGHT / HEIGHT)), NEW_HEIGHT - 1);
    return new_pixel_y * NEW_WIDTH + new_pixel_x;
  }
  
  //find neighbors of a pixel on image
  std::vector<int> findNeighbors(const int pixel, const int WIDTH, const int HEIGHT, const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8);
  
  //find neighbors for all pixels
  std::vector<std::vector<int> > findNeighborsForAllPixels(const int WIDTH, const int HEIGHT, const int NEIGHBOR_SYSTEM = 8);

  //find pixels in a neighboring window
  std::vector<int> findWindowPixels(const int pixel, const int WIDTH, const int HEIGHT, const int WINDOW_SIZE, const bool USE_PANORAMA = false);
  
  inline double calcColorDiff(const cv::Mat &image, const int &pixel_1, const int &pixel_2)
  {
    if (image.channels() == 1)
      return abs(1.0 * image.at<uchar>(pixel_1 / image.cols, pixel_1 % image.cols) - image.at<uchar>(pixel_2 / image.cols, pixel_2 % image.cols));
    else {
      cv::Vec3b color_1 = image.at<cv::Vec3b>(pixel_1 / image.cols, pixel_1 % image.cols);
      cv::Vec3b color_2 = image.at<cv::Vec3b>(pixel_2 / image.cols, pixel_2 % image.cols);
      double difference = 0;
      for (int c = 0; c < 3; c++)
	difference += pow(1.0 * color_1[c] - color_2[c], 2);
      return sqrt(difference);
    }
  }
  
  template<typename T> inline T getMin(const std::vector<T> &values)
    {
      T min_value = std::numeric_limits<T>::max();
      for (typename std::vector<T>::const_iterator value_it = values.begin(); value_it != values.end(); value_it++)
	if (*value_it < min_value)
	  min_value = *value_it;
      return min_value;
    }
  
  template<typename T> inline T getMax(const std::vector<T> &values)
    {
      T max_value = std::numeric_limits<T>::lowest();
      for (typename std::vector<T>::const_iterator value_it = values.begin(); value_it != values.end(); value_it++)
        if (*value_it > max_value)
          max_value = *value_it;
      return max_value;
    }
  
  template<typename T> inline std::vector<T> randomSampleValues(const std::vector<T> &values, const int NUM_SAMPLES)
  {
    assert(values.size() >= NUM_SAMPLES);
    std::vector<T> sampled_values;
    std::vector<bool> used_mask(values.size(), false);
    while (sampled_values.size() < NUM_SAMPLES) {
      int value_index = rand() % values.size();
      if (used_mask[value_index] == true)
	continue;
      sampled_values.push_back(values[value_index]);
      used_mask[value_index] = true;
    }
    return sampled_values;
  }
  
  template<typename T> inline std::vector<T> getVec(const T &value_1, const T &value_2)
  {
    std::vector<T> values(2);
    values[0] = value_1;
    values[1] = value_2;
    return values;
  }
  
  template<typename T> inline std::vector<T> getVec(const T &value_1, const T &value_2, const T &value_3)
  {
    std::vector<T> values(3);
    values[0] = value_1;
    values[1] = value_2;
    values[3] = value_3;
    return values;
  }
  
  
  /**********Statistics Calculation**********/
  
  //generate a random probablity
  inline double randomProbability()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
  }
  
  //calculate the number of bits with specified bit value
  inline int calcNumBits(const int value, const int denoted_bit_value)
  {
    int num_bits = 0;
    int current_value = value;
    while (current_value > 0) {
      int bit_value = current_value % 2;
      num_bits += bit_value == denoted_bit_value ? 1 : 0;
      current_value /= 2;
    }
    return num_bits;
  }
  
  //calculate the mean and svar of a set of values
  std::vector<double> calcMeanAndSVar(const std::vector<double> &values);
  //calculate the mean and svar of a set of values with more than one dimension
  void calcMeanAndSVar(const std::vector<std::vector<double> > &values, std::vector<double> &mean, std::vector<std::vector<double> > &var);
  
  //find all possible combinations of selecting num_elements elements from candidates
  std::vector<std::vector<int> > findAllCombinations(const std::vector<int> &candidates, const int num_elements);
  
  //calculate the number of distinct values
  int calcNumDistinctValues(const std::vector<int> &values);
  
  
  /**********Geometry Calculation**********/
  
  //fit a plane based on points.
  std::vector<double> fitPlane(const std::vector<double> &points);
  
  //estimate camera parameters
  void estimateCameraParameters(const std::vector<double> &point_cloud, const int image_width, const int image_height, double &focal_length);
  void estimateCameraParameters(const std::vector<double> &point_cloud, const int image_width, const int image_height, std::vector<double> &camera_parameters, const bool USE_PANORAMA = false);
  
  //calculate surface normals for given point cloud based on local windows
  std::vector<double> calcNormals(const std::vector<double> &point_cloud, const int image_width, const int image_height, const int K = 30);
  
  //calculate surface normal at pixel for given point cloud based on local windows
  std::vector<double> calcNormals(const std::vector<double> &point_cloud, const int pixel, const int image_width, const int image_height, const int K = 30);
  
  //calculate surface curvatures for given point cloud based on local windows
  std::vector<double> calcCurvatures(const std::vector<double> &point_cloud, const int image_width, const int image_height, const int K = 30);
  
  //calculate 2D geodesic distance between two pixels
  double calcGeodesicDistance(const std::vector<std::vector<double> > &distance_map, const int width, const int height, const int start_pixel, const int end_pixel, const double distance_2D_weight);
  
  //calculate 2D geodesic distance from a start pixel to all end pixels
  std::vector<double> calcGeodesicDistances(const std::vector<std::vector<double> > &distance_map, const int width, const int height, const int start_pixel, const std::vector<int> end_pixels, const double distance_2D_weight);
  
  
  //calculate the dot product of two vectors
  inline double calcDotProduct(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
  {
    assert(vec_1.size() == vec_2.size());
    double dot_product = 0;
    for (int i = 0; i < vec_1.size(); i++)
      dot_product += vec_1[i] * vec_2[i];
    return dot_product;
  }
  
  //calculate the cross product of two vectors
  inline std::vector<double> calcCrossProduct(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
  {
    assert(vec_1.size() == vec_2.size() && vec_1.size() == 3);
    std::vector<double> cross_product(3);
    cross_product[0] = vec_1[1] * vec_2[2] - vec_1[2] * vec_2[1];
    cross_product[1] = vec_1[2] * vec_2[0] - vec_1[0] * vec_2[2];
    cross_product[2] = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0];
    return cross_product;
  }
  
  //calculate the L2 norm of a vector
  inline double calcNorm(const std::vector<double> &vec)
  {
    double sum = 0;
    for (int c = 0; c < vec.size(); c++)
      sum += pow(vec[c], 2);
    return sqrt(sum);
  }

  //calculate the distance from a point to a plane
  inline double calcPointPlaneDistance(const std::vector<double> &point, const std::vector<double> &plane)
  {
    double distance = plane[3];
    for (int c = 0; c < 3; c++)
      distance -= plane[c] * point[c];
    return distance;
  }

  //calculate the euclidean distance between two vectors
  inline double calcDistance(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
  {
    assert(vec_1.size() == vec_2.size());
    double distance = 0;
    for (int c = 0; c < vec_1.size(); c++)
      distance += pow(vec_1[c] - vec_2[c], 2);
    return sqrt(distance);
  }

  //calculate the angle between two vectors
  inline double calcAngle(const std::vector<double> &vec_1, const std::vector<double> &vec_2)
  {
    assert(vec_1.size() == vec_2.size());
    double cos_value = 0;
    for (int c = 0; c < vec_1.size(); c++)
      cos_value += vec_1[c] * vec_2[c];
    return acos(std::max(std::min(cos_value, 1.0), -1.0));
  }

  //fit a 2D line
  std::vector<double> fitLine2D(const std::vector<double> &points);
  
  /**********Point Cloud Operations**********/

  inline std::vector<double> getPoint(const std::vector<double> &point_cloud, const int pixel)
  {
    return std::vector<double>(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
  }
  
  inline std::vector<double> getPoints(const std::vector<double> &point_cloud, const std::vector<int> &pixels)
  {
    std::vector<double> points;
    for (std::vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      points.insert(points.end(), point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    return points;
  }
  
  inline bool checkPointValidity(const std::vector<double> &point)
  {
    assert(point.size() == 3);
    if (calcNorm(std::vector<double>(point.begin(), point.begin() + 3)) < 0.000001)
      return false;
    else
      return true;
  }
  
  inline ImageMask getInvalidPointMask(const std::vector<double> &point_cloud, const int WIDTH, const int HEIGHT)
  {
    std::vector<bool> invalid_mask(WIDTH * HEIGHT, false);
    for (int pixel = 0; pixel < WIDTH * HEIGHT; pixel++)
      invalid_mask[pixel] = !checkPointValidity(getPoint(point_cloud, pixel));
    return ImageMask(invalid_mask, WIDTH, HEIGHT);
  }
  
  bool writePointCloud(const std::string &filename, const std::vector<double> &point_cloud, const int IMAGE_WIDTH, const int IMAGE_HEIGHT);
  bool readPointCloud(const std::string &filename, std::vector<double> &point_cloud);
  
  /**********Matrix Calculation**********/
  std::vector<std::vector<double> > calcInverse(const std::vector<std::vector<double> > &matrix);
}

#endif
