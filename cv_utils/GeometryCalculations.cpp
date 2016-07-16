#include "cv_utils.h"

#include <iostream>

//#include <gsl/gsl_spline.h>

#include <set>
//#include <einspline/bspline.h>

#include <Eigen/Sparse>
#include <opencv2/imgproc/imgproc.hpp>
#include <limits>

#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>


using namespace Eigen;
using namespace std;
using namespace cv;
using namespace pcl;


namespace cv_utils
{
VectorXd calcProjectionRay(const MatrixXd projection_matrix, const double image_x, const double image_y)
  {
    Matrix3d R = projection_matrix.block(0, 0, 3, 3);
    Vector3d X;
    X(0) = image_x;
    X(1) = image_y;
    X(2) = 1;
    
    Vector3d direction = R.inverse() * X;
    direction.normalize();
    Vector3d point = -R.inverse() * projection_matrix.col(3);
    VectorXd line(6, 1);
    line << direction, 
      point;
    return line;
  }
  
  //fit a plane based on points.
  vector<double> fitPlane(const vector<double> &points)
  {
    const int NUM_POINTS = points.size() / 3;
    assert(NUM_POINTS >= 3);
    
    VectorXd center(3);
    center << 0, 0, 0;
    for (int i = 0; i < NUM_POINTS; i++)
      for (int c = 0; c < 3; c++)
        center[c] += points[i * 3 + c];
    center /= NUM_POINTS;
    
    MatrixXd A(3, NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++)
      for (int c = 0; c < 3; c++)
        A(c, i) = points[i * 3 + c] - center[c];
    
    // MatrixXd AA = A * A.transpose();
    // Vector3d normal_test = AA.
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //    MatrixXf S = svd.singularValues();
    //    cout << S << endl;
    MatrixXd U = svd.matrixU();
    //    cout << U << endl;
    Vector3d normal = U.col(2);
    vector<double> plane(4);
    for (int c = 0; c < 3; c++)
      plane[c] = normal[c];
    plane[3] = normal.dot(center);
    return plane;
  }
  
  Vector3d calcIntersectionPointOnFace(const vector<double> &four_corners, const VectorXd &line, bool &on_face)
  {
    vector<double> plane = fitPlane(four_corners);
    Vector3d normal;
    for (int c = 0; c < 3; c++)
      normal[c] = plane[c];
    double d = plane[3];
    Vector3d origin = line.tail(3);
    Vector3d direction = line.head(3);
    if (normal.dot(direction) == 0) {
      on_face = false;
      Vector3d point;
      point << 1000000, 1000000, 1000000;
      return point;
    }
    double s = (d - normal.dot(origin)) / (normal.dot(direction));
    Vector3d intersection_point = origin + direction * s;
  
    VectorXd range(6);
    for (int c = 0; c < 3; c++) {
      range(c) = 1000000;
      range(c + 3) = -1000000;
    }
    for (int i = 0; i < four_corners.size() / 3; i++) {
      for (int c = 0; c < 3; c++) {
	if (four_corners[i * 3 + c] < range(c))
	  range(c) = four_corners[i * 3 + c];
	if (four_corners[i * 3 + c] > range(c + 3))
	  range(c + 3) = four_corners[i * 3 + c];
      }
    }
    on_face = true;
    int dominant_direction = -1;
    double max_normal_value = 0;
    for (int c = 0; c < 3; c++) {
      double normal_value = abs(normal(c));
      if (normal_value > max_normal_value) {
	dominant_direction = c;
	max_normal_value = normal_value;
      }
    }
    for (int c = 0; c < 3; c++) {
      if (c == dominant_direction)
	continue;
      if (intersection_point(c) < range(c))
	on_face = false;
      if (intersection_point(c) > range(c + 3))
	on_face = false;
    }
    return intersection_point;
  }
  
  Vector3d calcIntersectionPointOnPlane(const vector<double> &plane, const VectorXd &line)
  {
    Vector3d normal;
    for (int c = 0; c < 3; c++)
      normal(c) = plane[c];
    double d = plane[3];
    Vector3d origin = line.tail(3);
    Vector3d direction = line.head(3);
    if (normal.dot(direction) == 0) {
      Vector3d point;
      point << 1000000, 1000000, 1000000;
      return point;
    }
    double s = (d - normal.dot(origin)) / (normal.dot(direction));
    Vector3d intersection_point = origin + direction * s;
  
    return intersection_point;
  }
  
  
  vector<double> fitLine2D(const vector<double> &points)
  {
    const int NUM_POINTS = points.size() / 2;
    assert(NUM_POINTS >= 2);

    VectorXd center(2);
    center << 0, 0;
    for (int i = 0; i < NUM_POINTS; i++)
      for (int c = 0; c < 2; c++)
	center[c] += points[i * 2 + c];
    center /= NUM_POINTS;

    MatrixXd A(2, NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++)
      for (int c = 0; c < 2; c++)
	A(c, i) = points[i * 2 + c] - center[c];

    // MatrixXd AA = A * A.transpose();
    // Vector3d normal_test = AA.
    Eigen::JacobiSVD<MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //    MatrixXf S = svd.singularValues();
    //    cout << S << endl;
    MatrixXd U = svd.matrixU();
    //    cout << U << endl;
    VectorXd normal = U.col(1);
    vector<double> line(3);
    for (int c = 0; c < 2; c++)
      line[c] = normal[c];
    line[2] = normal.dot(center);
    return line;
  }
  
  
  
  
  void estimateCameraParameters(const vector<double> &point_cloud, const int image_width, const int image_height, double &focal_length)
  {
    double cx = image_width * 0.5;
    double cy = image_height * 0.5;
    
    int num_valid_depth_values = 0;
    for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
      //double X = point_cloud[point_index * 3 + 0];
      //double Y = point_cloud[point_index * 3 + 1];
      double Z = point_cloud[point_index * 3 + 2];
      if (Z <= 0)
	continue;
      num_valid_depth_values++;
    }
    
    double sum_aa = 0;
    double sum_ab = 0;
    for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
      double X = point_cloud[point_index * 3 + 0];
      double Y = point_cloud[point_index * 3 + 1];
      double Z = point_cloud[point_index * 3 + 2];
      if (Z <= 0)
	continue;
      int x = point_index % image_width;
      int y = point_index / image_width;
      double a = X / Z;
      double b = x - cx;
      sum_aa += pow(a, 2);
      sum_ab += a * b;
      a = Y / Z;
      b = y - cy;
      sum_aa += pow(a, 2);
      sum_ab += a * b;
    }
    focal_length = sum_ab / sum_aa;
  }
  
  void estimateCameraParameters(const vector<double> &point_cloud, const int image_width, const int image_height, vector<double> &camera_parameters, const bool USE_PANORAMA)
  {
    if (USE_PANORAMA == false) {
      int num_valid_depth_values = 0;
      for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
	//double X = point_cloud[point_index * 3 + 0];
	//double Y = point_cloud[point_index * 3 + 1];
	double Z = point_cloud[point_index * 3 + 2];
	if (abs(Z) < 0.000001)
	  continue;
	num_valid_depth_values++;
      }
      
      MatrixXd A(num_valid_depth_values * 2, 3);
      VectorXd b(num_valid_depth_values * 2);
      int valid_point_index = 0;
      for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
	double X = point_cloud[point_index * 3 + 0];
	double Y = point_cloud[point_index * 3 + 1];
	double Z = point_cloud[point_index * 3 + 2];
	if (abs(Z) < 0.000001)
	  continue;
	int x = point_index % image_width;
	int y = point_index / image_width;
	A(valid_point_index * 2 + 0, 0) = X / Z;
	A(valid_point_index * 2 + 0, 1) = 1;
	A(valid_point_index * 2 + 0, 2) = 0;
	b(valid_point_index * 2 + 0) = x;
	A(valid_point_index * 2 + 1, 0) = Y / Z;
	A(valid_point_index * 2 + 1, 1) = 0;
	A(valid_point_index * 2 + 1, 2) = 1;
	b(valid_point_index * 2 + 1) = y;
	valid_point_index++;
      }
      VectorXd solution = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
      
      camera_parameters.assign(3, 0);
      for (int c = 0; c < 3; c++)
	camera_parameters[c] = solution(c);
    } else {
      int num_valid_depth_values = 0;
      for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
	//double X = point_cloud[point_index * 3 + 0];
	//double Y = point_cloud[point_index * 3 + 1];
	double depth = sqrt(pow(point_cloud[point_index * 3 + 0], 2) + pow(point_cloud[point_index * 3 + 1], 2));
	if (depth < 0.000001)
	  continue;
	num_valid_depth_values++;
      }
      
      MatrixXd A(num_valid_depth_values * 2, 3);
      VectorXd b(num_valid_depth_values * 2);
      int valid_point_index = 0;
      for (int point_index = 0; point_index < point_cloud.size() / 3; point_index++) {
	double X = point_cloud[point_index * 3 + 0];
	double Y = point_cloud[point_index * 3 + 1];
	double Z = point_cloud[point_index * 3 + 2];
	double depth = sqrt(pow(X, 2) + pow(Y, 2));
	if (depth < 0.000001)
	  continue;
	int x = point_index % image_width;
	int y = point_index / image_width;
	double projected_cx = x - atan2(X, Y) / (2 * M_PI) * image_width;
	if (projected_cx < 0)
	  projected_cx += image_width;
	A(valid_point_index * 2 + 0, 0) = 0;
	A(valid_point_index * 2 + 0, 1) = 1;
	A(valid_point_index * 2 + 0, 2) = 0;
	b(valid_point_index * 2 + 0) = projected_cx;
	A(valid_point_index * 2 + 1, 0) = atan(-Z / depth) / (M_PI);
	A(valid_point_index * 2 + 1, 1) = 0;
	A(valid_point_index * 2 + 1, 2) = 1;
	b(valid_point_index * 2 + 1) = y;
	//cout << X << '\t' << Y << '\t' << Z << '\t' << x << '\t' << y << endl;
	valid_point_index++;
	//sum_b += y - image_height / 2;
	//sum_A += -Z / depth;
      }
      VectorXd solution = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
      
      camera_parameters.assign(3, 0);
      for (int c = 0; c < 3; c++)
        camera_parameters[c] = solution(c);
    }
  }
  
  
  
  
  // vector<double> inpaintPointCloud(const vector<double> &point_cloud, const int image_width, const int image_height)
  // {
  //   const double DATA_WEIGHT = 10000;
  //   const int WINDOW_SIZE = 3;
  
  //   SparseMatrix<double> A(image_width * image_height, image_width * image_height);
  //   VectorXd b(image_width * image_height);
  
  //   vector<Eigen::Triplet<double> > triplets;
  
  //   for (int y = 0; y < image_height; ++y) {
  //     for (int x = 0; x < image_width; ++x) {
  //       int pixel = y * image_width + x;
  //       int count = 0;
  //       for (int delta_y = -WINDOW_SIZE / 2; delta_y <= WINDOW_SIZE / 2; ++delta_y) {
  //         for (int delta_x = -WINDOW_SIZE / 2; delta_x <= WINDOW_SIZE / 2; ++delta_x) {
  //           if (x + delta_x < 0 || x + delta_x >= image_width || y + delta_y < 0 || y + delta_y >= image_height)
  //             continue;
  //           if (delta_x == 0 && delta_y == 0)
  //             continue;
  //           triplets.push_back(Eigen::Triplet<double>(pixel, pixel + delta_y * image_width + delta_x, -1));
  //           ++count;
  //         }
  //       }
  
  //       if (point_cloud[pixel * 3 + 2] > 0) {
  //         triplets.push_back(Eigen::Triplet<double>(pixel, pixel, DATA_WEIGHT + count));
  //         b[pixel] = DATA_WEIGHT * point_cloud[pixel * 3 + 2];
  //       } else {
  //         triplets.push_back(Eigen::Triplet<double>(pixel, pixel, count));
  //         b[pixel] = 0;
  //       }
  //     }
  //   }
  //   A.setFromTriplets(triplets.begin(), triplets.end());
  //   SimplicialCholesky<SparseMatrix<double> > chol(A);
  //   VectorXd solution = chol.solve(b);
  
  //   vector<double> inpainted_point_cloud = point_cloud;
  //   for (int pixel = 0; pixel < image_width * image_height; ++pixel) {
  //     if (inpainted_point_cloud[pixel * 3 + 2] <= 0)
  //       inpainted_point_cloud[pixel * 3 + 2] = solution[pixel];
  //   }
  //   return inpainted_point_cloud;
  // }
  
  
  vector<double> calcNormals(const vector<double> &point_cloud, const int image_width, const int image_height, const int K)
  {
    // vector<double> normals;
    // for (int y = 0; y < image_height; y++) {
    //   for (int x = 0; x < image_width; x++) {
    //     vector<double> points;
    //     for (int delta_y = -(window_size - 1) / 2; delta_y <= (window_size - 1) / 2; delta_y++)
    // 	for (int delta_x = -(window_size - 1) / 2; delta_x <= (window_size - 1) / 2; delta_x++)
    // 	  if (x + delta_x >= 0 && x + delta_x < image_width && y + delta_y >= 0 && y + delta_y < image_height)
    // 	    points.insert(points.end(), point_cloud.begin() + ((y + delta_y) * image_width + (x + delta_x)) * 3, point_cloud.begin() + ((y + delta_y) * image_width + (x + delta_x) + 1) * 3);
    //     vector<double> plane = fitPlane(points);
    //     normals.insert(normals.end(), plane.begin(), plane.begin() + 3);
    //   }
    // }
    // return normals;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Fill in the cloud data
    cloud->width  = image_width;
    cloud->height = image_height;
    cloud->points.resize(cloud->width * cloud->height);
    // Generate the data
    for (size_t i = 0; i < cloud->points.size (); ++i) {
      if (checkPointValidity(getPoint(point_cloud, i)) == false)
	continue;
      cloud->points[i].x = point_cloud[i * 3 + 0];
      cloud->points[i].y = point_cloud[i * 3 + 1];
      cloud->points[i].z = point_cloud[i * 3 + 2];
    }
    
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals_pcl(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    //normal_estimator.setKSearch (params_.normal_estimation_K_1);
    normal_estimator.setKSearch(K);
    normal_estimator.compute(*normals_pcl);
    
    vector<double> normals(image_width * image_height * 3, 0);
    for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < image_width; x++) {
	int pixel = y * image_width + x;
	if (checkPointValidity(getPoint(point_cloud, pixel)) == false)
          continue;
	Normal normal = normals_pcl->points[pixel];
	normals[pixel * 3 + 0] = normal.normal_x;
	normals[pixel * 3 + 1] = normal.normal_y;
	normals[pixel * 3 + 2] = normal.normal_z;
      }
    }
    // for (size_t i = 0; i < cloud->points.size (); ++i)
    //   if (point_cloud[i * 3 + 2] > 0)
    //     cout << normals[i * 3 + 0] << '\t' << normals[i * 3 + 1] << '\t' << normals[i * 3 + 2] << endl;
    // exit(1);
    return normals;
  }
  
  vector<double> calcCurvatures(const vector<double> &point_cloud, const int image_width, const int image_height, const int K)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Fill in the cloud data
    cloud->width  = image_width;
    cloud->height = image_height;
    cloud->points.resize(cloud->width * cloud->height);
    // Generate the data
    for (size_t i = 0; i < cloud->points.size (); ++i) {
      if (checkPointValidity(getPoint(point_cloud, i)) == false)
        continue;
      cloud->points[i].x = point_cloud[i * 3 + 0];
      cloud->points[i].y = point_cloud[i * 3 + 1];
      cloud->points[i].z = point_cloud[i * 3 + 2];
    }
    
    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud <pcl::Normal>::Ptr normals_pcl(new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    //normal_estimator.setKSearch (params_.normal_estimation_K_1);
    normal_estimator.setKSearch(K);
    normal_estimator.compute(*normals_pcl);
    
    vector<double> curvatures(image_width * image_height, 0);
    for (int y = 0; y < image_height; y++) {
      for (int x = 0; x < image_width; x++) {
        int pixel = y * image_width + x;
	if (checkPointValidity(getPoint(point_cloud, pixel)) == false)
          continue;
        Normal normal = normals_pcl->points[pixel];
        curvatures[pixel] = normal.curvature;
      }
    }
    return curvatures;
  }
  
  
  double calcGeodesicDistance(const vector<vector<double> > &distance_map, const int width, const int height, const int start_pixel, const int end_pixel, const double distance_2D_weight)
  {
    int start_x = start_pixel % width;
    int start_y = start_pixel / width;
    int end_x = end_pixel % width;
    int end_y = end_pixel / width;
    int delta_x = start_x < end_x ? 1 : -1;
    int delta_y = start_y < end_y ? 1 : -1;
    vector<double> distances(width * height, 1000000);
    distances[start_pixel] = 0;
    vector<double> start_points(width * height);
    for (int step = 1; step <= abs(end_x - start_x) + abs(end_y - start_y); step++) {
      for (int i = 0; i <= step; i++) {
        int x = start_x + delta_x * i;
        int y = start_y + delta_y * (step - i);
        if ((x - start_x) * (x - end_x) > 0 || (y - start_y) * (y - end_y) > 0)
          continue;
        int pixel = y * width + x;
        if ((x - delta_x - start_x) * (x - delta_x - end_x) <= 0) {
          double distance = distances[pixel - delta_x] + distance_2D_weight + distance_map[pixel - delta_x][(0 + 1) * 3 + (delta_x + 1)];
          if (distance < distances[pixel]) {
            distances[pixel] = distance;
            start_points[pixel] = pixel - delta_x;
          }
        }
        if ((y - delta_y - start_y) * (y - delta_y - end_y) <= 0) {
          double distance = distances[pixel - delta_y * width] + distance_2D_weight + distance_map[pixel - delta_y * width][(delta_y + 1) * 3 + (0 + 1)];
          if (distance < distances[pixel]) {
            distances[pixel] = distance;
            start_points[pixel] = pixel - delta_y * width;
          }
        }
        if ((x - delta_x - start_x) * (x - delta_x - end_x) <= 0 && (y - delta_y - start_y) * (y - delta_y - end_y) <= 0) {
          double distance = distances[pixel - delta_y * width - delta_x] + sqrt(2) * distance_2D_weight + distance_map[pixel - delta_y * width - delta_x][(delta_y + 1) * 3 + (delta_x + 1)];
          if (distance < distances[pixel]) {
            distances[pixel] = distance;
            start_points[pixel] = pixel - delta_y * width - delta_x;
          }
        }
        // cout << distance_map[pixel - delta_x][(0 + 1) * 3 + (delta_x + 1)] << '\t' << distance_map[pixel - delta_y * width][(delta_y + 1) * 3 + (0 + 1)] << endl;
        // cout << x << '\t' << y << '\t' << distances[pixel] << endl;
        // exit(1);
      }
    }
    // for (int x = start_x; x != end_x + delta_x; x += delta_x)
    //   for (int y = start_y; y != end_y + delta_y; y += delta_y)
    //     cout << y * width + x << '\t' << distances[y * width + x] << '\t' << start_points[y * width + x] << endl;
    return distances[end_pixel];
  }
  
  vector<double> calcGeodesicDistances(const vector<vector<double> > &distance_map, const int width, const int height, const int start_pixel, const vector<int> end_pixels, const double distance_2D_weight)
  {
    int start_x = start_pixel % width;
    int start_y = start_pixel / width;
    
    int min_x = width;
    int max_x = -1;
    int min_y = height;
    int max_y = -1;
    for (vector<int>::const_iterator pixel_it = end_pixels.begin(); pixel_it != end_pixels.end(); pixel_it++) {
      int x = *pixel_it % width;
      int y = *pixel_it / width;
      if (x < min_x)
        min_x = x;
      if (x > max_x)
        max_x = x;
      if (y < min_y)
        min_y = y;
      if (y > max_y)
        max_y = y;
    }
    
    vector<double> distances(width * height, 1000000);
    distances[start_pixel] = 0;
    vector<double> start_points(width * height);  
    
    vector<int> end_xs;
    end_xs.push_back(min_x);
    end_xs.push_back(max_x);
    vector<int> end_ys;
    end_ys.push_back(min_y);
    end_ys.push_back(max_y);
    for (vector<int>::const_iterator end_x_it = end_xs.begin(); end_x_it != end_xs.end(); end_x_it++) {
      for (vector<int>::const_iterator end_y_it = end_ys.begin(); end_y_it != end_ys.end(); end_y_it++) {
        int end_x = *end_x_it;
        int end_y = *end_y_it;
	
        int delta_x = start_x < end_x ? 1 : -1;
        int delta_y = start_y < end_y ? 1 : -1;
	
        for (int step = 1; step <= abs(end_x - start_x) + abs(end_y - start_y); step++) {
          for (int i = 0; i <= step; i++) {
            int x = start_x + delta_x * i;
            int y = start_y + delta_y * (step - i);
            if ((x - start_x) * (x - end_x) > 0 || (y - start_y) * (y - end_y) > 0)
              continue;
            int pixel = y * width + x;
            if ((x - delta_x - start_x) * (x - delta_x - end_x) <= 0) {
              double distance = distances[pixel - delta_x] + distance_2D_weight + distance_map[pixel - delta_x][(0 + 1) * 3 + (delta_x + 1)];
              if (distance < distances[pixel]) {
                distances[pixel] = distance;
                start_points[pixel] = pixel - delta_x;
              }
            }
            if ((y - delta_y - start_y) * (y - delta_y - end_y) <= 0) {
              double distance = distances[pixel - delta_y * width] + distance_2D_weight + distance_map[pixel - delta_y * width][(delta_y + 1) * 3 + (0 + 1)];
              if (distance < distances[pixel]) {
                distances[pixel] = distance;
                start_points[pixel] = pixel - delta_y * width;
              }
            }
            if ((x - delta_x - start_x) * (x - delta_x - end_x) <= 0 && (y - delta_y - start_y) * (y - delta_y - end_y) <= 0) {
              double distance = distances[pixel - delta_y * width - delta_x] + sqrt(2) * distance_2D_weight + distance_map[pixel - delta_y * width - delta_x][(delta_y + 1) * 3 + (delta_x + 1)];
              if (distance < distances[pixel]) {
                distances[pixel] = distance;
                start_points[pixel] = pixel - delta_y * width - delta_x;
              }
            }
            // cout << distance_map[pixel - delta_x][(0 + 1) * 3 + (delta_x + 1)] << '\t' << distance_map[pixel - delta_y * width][(delta_y + 1) * 3 + (0 + 1)] << endl;
            // cout << x << '\t' << y << '\t' << distances[pixel] << endl;
            // exit(1);
          }
        }
      }
    }
    // for (int x = start_x; x != end_x + delta_x; x += delta_x)
    //   for (int y = start_y; y != end_y + delta_y; y += delta_y)
    //     cout << y * width + x << '\t' << distances[y * width + x] << '\t' << start_points[y * width + x] << endl;
    vector<double> end_pixel_distances(end_pixels.size());
    for (vector<int>::const_iterator pixel_it = end_pixels.begin(); pixel_it != end_pixels.end(); pixel_it++)
      end_pixel_distances[pixel_it - end_pixels.begin()] = distances[*pixel_it];
    return end_pixel_distances;
  }
}
