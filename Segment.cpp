#include "Segment.h"

#include <Eigen/Dense>

#include "utils.h"

#include <iostream>

#include "TRW_S/MRFEnergy.h"
#include "BSplineSurface.h"

#include "cv_utils/cv_utils.h"

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace Eigen;
using namespace cv_utils;

Segment::Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const vector<double> &camera_parameters, const vector<int> &pixels, const RepresenterPenalties &penalties, const DataStatistics &input_statistics, const int segment_type) : IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), input_statistics_(input_statistics), segment_type_(segment_type)
{
  if (segment_type == 0)
    fitDepthPlane(image, point_cloud, normals, deleteInvalidPixels(point_cloud, pixels));
  else if (segment_type > 0) {
    if (pixels.size() > input_statistics_.bspline_surface_num_pixels_threshold) {
      segment_pixels_ = pixels;
      return;
    }
    fitBSplineSurface(image, point_cloud, normals, deleteInvalidPixels(point_cloud, pixels));
  }
  
  calcColorStatistics(image, segment_pixels_);
  calcSegmentMaskInfo();
}

Segment::Segment(const int image_width, const int image_height, const vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &input_statistics) : IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(image_width * image_height), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), input_statistics_(input_statistics)
{
}

Segment::Segment()
{
}


void Segment::fitDepthPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
{
  if (pixels.size() < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }

  Mat blurred_image;
  GaussianBlur(image, blurred_image, cv::Size(3, 3), 0, 0);
  Mat blurred_hsv_image;
  blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);

  segment_type_ = 0;
  
  const int NUM_ITERATIONS = min(static_cast<int>(pixels.size() / 3), 300);
  
  int max_num_inliers = 0;
  vector<double> max_num_inliers_depth_plane;
  //Vec3f max_num_inliers_mean_color;
  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
    set<int> initial_point_indices;
    while (initial_point_indices.size() < 3)
      initial_point_indices.insert(rand() % pixels.size());
    
    Vec3f sum_color(0, 0, 0);
    for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
      sum_color += blurred_hsv_image.at<Vec3f>(pixels[*point_index_it] / IMAGE_WIDTH_, pixels[*point_index_it] % IMAGE_WIDTH_);
    
    vector<double> initial_points;
    for (set<int>::const_iterator point_index_it = initial_point_indices.begin(); point_index_it != initial_point_indices.end(); point_index_it++)
      initial_points.insert(initial_points.end(), point_cloud.begin() + pixels[*point_index_it] * 3, point_cloud.begin() + (pixels[*point_index_it] + 1) * 3);

    vector<double> depth_plane = fitPlane(initial_points);
    if (depth_plane.size() == 0)
      continue;
    
    int num_inliers = 0;
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
      assert(point_cloud[*pixel_it * 3 + 2] > 0);
      
      if (point_cloud[*pixel_it * 3 + 2] < 0)
	continue;
      vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
      double distance = depth_plane[3];
      for (int c = 0; c < 3; c++)
	distance -= depth_plane[c] * point[c];
      distance = abs(distance);
      if (distance > input_statistics_.pixel_fitting_distance_threshold)
	continue;
      
      vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
      double cos_value = 0;
      for (int c = 0; c < 3; c++)
      	cos_value += normal[c] * depth_plane[c];
      double angle = acos(min(abs(cos_value), 1.0));
      if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
        angle = 0;
      if (angle > input_statistics_.pixel_fitting_angle_threshold)
      	continue;

      num_inliers++;
    }
    //cout << num_inliers << '\t' << points.size() / 3 << endl;
    if (num_inliers > max_num_inliers) {
      max_num_inliers_depth_plane = depth_plane;
      //max_num_inliers_mean_color = mean_color;
      max_num_inliers = num_inliers;
    }
  }
  //exit(1);
  
  if (max_num_inliers < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }

  depth_plane_ = max_num_inliers_depth_plane;
  segment_pixels_.clear();
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    double distance = depth_plane_[3];
    for (int c = 0; c < 3; c++)
      distance -= depth_plane_[c] * point[c];
    distance = abs(distance);
    if (distance > input_statistics_.pixel_fitting_distance_threshold)
      continue;
      
    vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
    double cos_value = 0;
    for (int c = 0; c < 3; c++)
      cos_value += normal[c] * depth_plane_[c];
    double angle = acos(min(abs(cos_value), 1.0));
    if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
      angle = 0;
    if (angle > input_statistics_.pixel_fitting_angle_threshold)
      continue;

    segment_pixels_.push_back(*pixel_it);
  }
  
  segment_pixels_ = findLargestConnectedComponent(point_cloud, segment_pixels_);
  if (segment_pixels_.size() < 3) {
    fitParallelSurface(point_cloud, normals, pixels);
    return;
  }
  
  vector<double> fitted_points;
  for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++) {
    vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    fitted_points.insert(fitted_points.end(), point.begin(), point.end());
  }
  depth_plane_ = fitPlane(fitted_points);
  
  calcDepthMap(point_cloud, segment_pixels_);
}

void Segment::fitParallelSurface(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels)
{
  segment_type_ = -1;
  
  disp_plane_ = vector<double>(3, 0);
  depth_plane_ = vector<double>(4, 0);
  if (pixels.size() > 0) {
    double disp_sum = 0;
    double depth_sum = 0;
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
      if (point_cloud[*pixel_it * 3 + 2] < 0)
	continue;
      disp_sum += 1 / point_cloud[*pixel_it * 3 + 2];
      depth_sum += point_cloud[*pixel_it * 3 + 2];
      segment_pixels_.push_back(*pixel_it);
    }
    double disp_mean = disp_sum / segment_pixels_.size();
    disp_plane_[2] = round(disp_mean * 10000) / 10000;
    double depth_mean = depth_sum / segment_pixels_.size();
    depth_plane_[2] = 1;
    depth_plane_[3] = round(depth_mean * 10000) / 10000;
  }
  
  calcDepthMap(point_cloud, pixels);
}

void Segment::calcColorStatistics(const Mat &image, const vector<int> &pixels)
{
  if (pixels.size() < 3) {
    return;
  }
  Mat segment_samples(pixels.size(), 2, CV_32FC1);
  Mat blurred_image;
  GaussianBlur(image, blurred_image, cv::Size(3, 3), 0, 0);
  Mat blurred_hsv_image;
  blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);
  
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    Vec3f color = blurred_hsv_image.at<Vec3f>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
    segment_samples.at<float>(pixel_it - pixels.begin(), 0) = color[1] * cos(color[0] * M_PI / 180);
    segment_samples.at<float>(pixel_it - pixels.begin(), 1) = color[1] * sin(color[0] * M_PI / 180);
  }
  
  GMM_ = EM::create();
  GMM_->setClustersNumber(2);
  Mat log_likelihoods(pixels.size(), 1, CV_64FC1);
  GMM_->trainEM(segment_samples, log_likelihoods, noArray(), noArray());
  double likelihood_sum = 0;
  double likelihood2_sum = 0;
  for (int i = 0; i < pixels.size(); i++) {
    double likelihood = log_likelihoods.at<double>(i, 0);
    likelihood_sum += likelihood;
    likelihood2_sum += pow(likelihood, 2);
  }
}

void Segment::calcDepthMap(const vector<double> &point_cloud, const vector<int> &fitted_pixels)
{
  disp_plane_ = vector<double>(3);
  disp_plane_[0] = depth_plane_[0] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
  disp_plane_[1] = depth_plane_[1] / (CAMERA_PARAMETERS_[0] * depth_plane_[3]);
  disp_plane_[2] = depth_plane_[2] / depth_plane_[3];
  
  depth_map_ = vector<double>(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    double u = pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1];
    double v = pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2];
    //double depth = 1 / ((plane(0) * u + plane(1) * v + plane(2)) / plane(3));
    
    double disp = disp_plane_[0] * u + disp_plane_[1] * v + disp_plane_[2];
    double depth = disp != 0 ? 1 / disp : 0;
    // if (depth <= 0)
    //   depth = -1;
    if (depth > 10)
      depth = 10;
    depth_map_[pixel] = depth;
  }
}

vector<double> Segment::getDepthMap() const
{
  return depth_map_;
}

double Segment::getDepth(const int pixel) const
{
  return depth_map_[pixel];
}

double Segment::getDepth(const double x_ratio, const double y_ratio) const
{
  double x = IMAGE_WIDTH_ * x_ratio;
  double y = IMAGE_HEIGHT_ * y_ratio;
  int lower_x = max(static_cast<int>(floor(x)), 0);
  int upper_x = min(static_cast<int>(ceil(x)), IMAGE_WIDTH_ - 1);
  int lower_y = max(static_cast<int>(floor(y)), 0);
  int upper_y = min(static_cast<int>(ceil(y)), IMAGE_HEIGHT_ - 1);
  if (lower_x == upper_x && lower_y == upper_y)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x];
  else if (lower_x == upper_x)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_y - y) + depth_map_[upper_y * IMAGE_WIDTH_ + lower_x] * (y - lower_y);
  else if (lower_y == upper_y)
    return depth_map_[lower_y * IMAGE_WIDTH_ + lower_x] * (upper_x - x) + depth_map_[lower_y * IMAGE_WIDTH_ + upper_x] * (x - lower_x);
  else {
    double area_1 = (x - lower_x) * (y - lower_y);
    double area_2 = (x - lower_x) * (upper_y - y);
    double area_3 = (upper_x - x) * (y - lower_y);
    double area_4 = (upper_x - x) * (upper_y - y);
    double depth_1 = depth_map_[lower_y * IMAGE_WIDTH_ + lower_x];
    double depth_2 = depth_map_[upper_y * IMAGE_WIDTH_ + lower_x];
    double depth_3 = depth_map_[lower_y * IMAGE_WIDTH_ + upper_x];
    double depth_4 = depth_map_[upper_y * IMAGE_WIDTH_ + upper_x];

    return depth_1 * area_4 + depth_2 * area_3 + depth_3 * area_2 + depth_4 * area_1;
  }
}

vector<double> Segment::getDepthPlane() const
{
  return depth_plane_;
}

int Segment::getType() const
{
  return segment_type_;
}

void Segment::calcSegmentMaskInfo()
{
  segment_mask_.assign(NUM_PIXELS_, false);
  int min_x = IMAGE_WIDTH_;
  int max_x = -1;
  int min_y = IMAGE_HEIGHT_;
  int max_y = -1;
  double sum_x = 0;
  double sum_y = 0;
  for (vector<int>::const_iterator pixel_it = segment_pixels_.begin(); pixel_it != segment_pixels_.end(); pixel_it++) {
    segment_mask_[*pixel_it] = true;
    int x = *pixel_it % IMAGE_WIDTH_;
    int y = *pixel_it / IMAGE_WIDTH_;
    if (x < min_x)
      min_x = x;
    if (x > max_x)
      max_x = x;
    if (y < min_y)
      min_y = y;
    if (y > max_y)
      max_y = y;
    sum_x += x;
    sum_y += y;
  }
  //segment_radius_ = sqrt((max_x - min_x + 1) * (max_y - min_y + 1));
  segment_radius_ = sqrt(segment_pixels_.size()) * 3;
  segment_center_x_ = sum_x / segment_pixels_.size();
  segment_center_y_ = sum_y / segment_pixels_.size();

  calcDistanceMap();
}

Segment &Segment::operator =(const Segment &segment)
{
  IMAGE_WIDTH_ = segment.IMAGE_WIDTH_;
  IMAGE_HEIGHT_ = segment.IMAGE_HEIGHT_;
  NUM_PIXELS_ = segment.NUM_PIXELS_;
  CAMERA_PARAMETERS_ = segment.CAMERA_PARAMETERS_;
  penalties_ = segment.penalties_;
  segment_pixels_ = segment.segment_pixels_;
  segment_mask_ = segment.segment_mask_;
  segment_radius_ = segment.segment_radius_;
  segment_center_x_ = segment.segment_center_x_;
  segment_center_y_ = segment.segment_center_y_;
  distance_map_ = segment.distance_map_;
  segment_type_ = segment.segment_type_;
  disp_plane_ = segment.disp_plane_;
  depth_plane_ = segment.depth_plane_;
  input_statistics_ = segment.input_statistics_;
  depth_map_ = segment.depth_map_;
  normals_ = segment.normals_;
  GMM_ = segment.GMM_;
  segment_confidence_ = segment.segment_confidence_;
  
  
  return *this;
}

ostream & operator <<(ostream &out_str, const Segment &segment)
{
  out_str << segment.segment_type_ << endl;
  out_str << segment.segment_pixels_.size() << endl;
  for (vector<int>::const_iterator pixel_it = segment.segment_pixels_.begin(); pixel_it != segment.segment_pixels_.end(); pixel_it++)
    out_str << *pixel_it << '\t';
  out_str << endl;
  if (segment.segment_type_ == 0) {
    for (int c = 0; c < 4; c++)
      out_str << segment.depth_plane_[c] << '\t';
    out_str << endl;
  } else if (segment.segment_type_ > 0) {
    for (vector<double>::const_iterator depth_it = segment.depth_map_.begin(); depth_it != segment.depth_map_.end(); depth_it++)
      out_str << *depth_it << '\t';
    out_str << endl;
  }
  
  return out_str;
}

istream & operator >>(istream &in_str, Segment &segment)
{
  in_str >> segment.segment_type_;
  int num_segment_pixels;
  in_str >> num_segment_pixels;
  segment.segment_pixels_ = vector<int>(num_segment_pixels);
  for (int pixel_index = 0; pixel_index < num_segment_pixels; pixel_index++)
    in_str >> segment.segment_pixels_[pixel_index];
  segment.calcSegmentMaskInfo();
  if (segment.segment_type_ == 0) {
    segment.depth_plane_.assign(4, 0);
    for (int c = 0; c < 4; c++)
      in_str >> segment.depth_plane_[c];
    segment.calcDepthMap();
  } else if (segment.segment_type_ > 0) {
    vector<double> depth_map(segment.NUM_PIXELS_, 0);
    for (int pixel = 0; pixel < segment.NUM_PIXELS_; pixel++)
      in_str >> depth_map[pixel];
    segment.depth_map_ = depth_map;
    segment.normals_ = calcNormals(segment.calcPointCloud(), segment.IMAGE_WIDTH_, segment.IMAGE_HEIGHT_);
  }
  return in_str;
}

double Segment::predictColorLikelihood(const int pixel, const Vec3f hsv_color) const
{
  if (segment_type_ == -1)
    return 0;
  Mat sample(1, 2, CV_64FC1);
  sample.at<double>(0, 0) = hsv_color[1] * cos(hsv_color[0] * M_PI / 180);
  sample.at<double>(0, 1) = hsv_color[1] * sin(hsv_color[0] * M_PI / 180);
  
  
  Vec2d prediction = GMM_->predict2(sample, noArray());
  Mat weights = GMM_->getWeights();
  return prediction[0] + log(weights.at<double>(0, prediction[1]));
}

void Segment::setGMM(const cv::FileNode GMM_file_node)
{
  GMM_ = EM::create();
  GMM_->read(GMM_file_node);
}

Ptr<EM> Segment::getGMM() const
{
  return GMM_;
}

vector<int> Segment::getSegmentPixels() const
{
  return segment_pixels_;
}

bool Segment::checkPixelFitting(const Mat &hsv_image, const vector<double> &point_cloud, const vector<double> &normals, const int pixel) const
{
  // if (use_sub_segment_ && segment_mask_[pixel])
  //   return true;
  
  if (segment_type_ == -1)
    return false;

  if (depth_map_[pixel] < 0)
    return false;

  Vec3f color = hsv_image.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  if (predictColorLikelihood(pixel, color) < input_statistics_.pixel_fitting_color_likelihood_threshold) {
    return false;
  }
  
  if (segment_type_ > 0) {
    double distance = abs(depth_map_[pixel] - point_cloud[pixel * 3 + 2]);
    if (point_cloud[pixel * 3 + 2] < 0)
      distance = 0;
    if (distance > input_statistics_.pixel_fitting_distance_threshold)
      return false;
    
    vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
    vector<double> surface_normal(normals_.begin() + pixel * 3, normals_.begin() + (pixel + 1) * 3);
    double cos_value = 0;
    for (int c = 0; c < 3; c++)
      cos_value += normal[c] * surface_normal[c];
    double angle = acos(min(abs(cos_value), 1.0));
    if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
      angle = 0;
    if (angle > input_statistics_.pixel_fitting_angle_threshold)
      return false;
    
    return true;
  }  
  
  vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
  double distance = depth_plane_[3];
  for (int c = 0; c < 3; c++)
    distance -= depth_plane_[c] * point[c];
  distance = abs(distance);
  if (point_cloud[pixel * 3 + 2] < 0)
    distance = 0;
  if (distance > input_statistics_.pixel_fitting_distance_threshold)
    return false;
      
  vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  double cos_value = 0;
  for (int c = 0; c < 3; c++)
    cos_value += normal[c] * depth_plane_[c];
  double angle = acos(min(abs(cos_value), 1.0));
  if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
    angle = 0;
  if (angle > input_statistics_.pixel_fitting_angle_threshold)
    return false;
    
  return true;
}

vector<int> Segment::findLargestConnectedComponent(const vector<double> &point_cloud, const vector<int> &pixels)
{
  vector<int> new_segment_pixels;
  vector<bool> segment_mask(NUM_PIXELS_, false);
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
    segment_mask[*pixel_it] = true;

  vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
  map<int, vector<int> > connected_components;
  int connected_component_index = 0;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    if (visited_pixel_mask[*pixel_it] == true)
      continue;
    
    vector<int> connected_component;
    vector<int> border_pixels;
    border_pixels.push_back(*pixel_it);
    visited_pixel_mask[*pixel_it] = true;
    while (true) {
      vector<int> new_border_pixels;
      for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
	connected_component.push_back(*border_pixel_it);
	//	double depth = point_cloud[*border_pixel_it * 3 + 2];
        vector<int> neighbor_pixels;
	int x = *border_pixel_it % IMAGE_WIDTH_;
	int y = *border_pixel_it / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(*border_pixel_it - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + 1);
	if (y > 0)
	  neighbor_pixels.push_back(*border_pixel_it - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(*border_pixel_it - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(*border_pixel_it + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(*border_pixel_it + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (segment_mask[*neighbor_pixel_it] == true && visited_pixel_mask[*neighbor_pixel_it] == false) {
	    new_border_pixels.push_back(*neighbor_pixel_it);
	    visited_pixel_mask[*neighbor_pixel_it] = true;
	  }
	}
      }
      if (new_border_pixels.size() == 0)
	break;
      border_pixels = new_border_pixels;
    }
    connected_components[connected_component_index] = connected_component;
    connected_component_index++;
  }  

  int max_num_pixels = 0;
  int max_num_pixels_component_index = -1;
  for (map<int, vector<int> >::const_iterator component_it = connected_components.begin(); component_it != connected_components.end(); component_it++) {
    if (component_it->second.size() > max_num_pixels) {
      max_num_pixels_component_index = component_it->first;
      max_num_pixels = component_it->second.size();
    }
  }
  
  return connected_components[max_num_pixels_component_index];
}

double Segment::calcAngle(const vector<double> &normals, const int pixel)
{
  if (segment_type_ == -1)
    return M_PI / 2;
  vector<double> normal(normals.begin() + pixel * 3, normals.begin() + (pixel + 1) * 3);
  if (sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)) < 0.000001)
    return 0;
  vector<double> surface_normal = segment_type_ == 0 ? vector<double>(depth_plane_.begin(), depth_plane_.begin() + 3) : vector<double>(normals_.begin() + pixel * 3, normals_.begin() + (pixel + 1) * 3);
  double cos_value = 0;
  for (int c = 0; c < 3; c++)
    cos_value += normal[c] * surface_normal[c];
  double angle = acos(min(abs(cos_value), 1.0));
  return angle;
}


void Segment::calcDistanceMap()
{
  vector<double> distances(NUM_PIXELS_, 1000000);
  distance_map_ = vector<int>(NUM_PIXELS_);

  vector<int> border_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (segment_mask_[pixel] == false)
      continue;
    distance_map_[pixel] = pixel;
    distances[pixel] = 0;
    
    vector<int> neighbor_pixels;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    if (x > 0)
      neighbor_pixels.push_back(pixel - 1);
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y > 0)
      neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y > 0)
      neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y > 0)
      neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (segment_mask_[*neighbor_pixel_it] == false) {
	border_pixels.push_back(pixel);
	break;
      }
    }
  }
  
  while (border_pixels.size() > 0) {
    vector<int> new_border_pixels;
    for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
      int pixel = *border_pixel_it;
      double distance = distances[pixel];
      vector<int> neighbor_pixels;
      int x = pixel % IMAGE_WIDTH_;
      int y = pixel / IMAGE_WIDTH_;
      // if (distance_map_[IMAGE_HEIGHT_ / 2 * IMAGE_WIDTH_ + IMAGE_WIDTH_ / 2] == 0)
      //   cout << x << '\t' << y << '\t' << border_pixel_distance << endl;
      if (x > 0)
	neighbor_pixels.push_back(pixel - 1);
      if (x < IMAGE_WIDTH_ - 1)
	neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
	neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
      if (y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
      if (x > 0 && y > 0)
	neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
      if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y > 0)
	neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
      if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);      
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	int neighbor_pixel = *neighbor_pixel_it;
	double distance_delta = sqrt(pow(*neighbor_pixel_it % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_, 2) + pow(*neighbor_pixel_it / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_, 2));
	if (distance + distance_delta < distances[neighbor_pixel]) {
	  distance_map_[neighbor_pixel] = pixel;
	  distances[neighbor_pixel] = distance + distance_delta;
	  new_border_pixels.push_back(neighbor_pixel);
	}
      }
    }
    border_pixels = new_border_pixels;
  }
}

int Segment::calcDistanceOffset(const int pixel_1, const int pixel_2)
{
  if (distance_map_[pixel_1] == pixel_2)
    return 1;
  if (distance_map_[pixel_2] == pixel_1)
    return -1;
  return 0;
}
void Segment::fitBSplineSurface(const Mat &image, const vector<double> &point_cloud, const std::vector<double> &normals, const vector<int> &pixels)
{
  segment_pixels_ = findLargestConnectedComponent(point_cloud, pixels);
  if (segment_pixels_.size() < 3) {
    fitParallelSurface(point_cloud, normals, segment_pixels_);
    return;
  }
  
  //segment_type_ = 2;
  BSplineSurface surface(point_cloud, segment_pixels_, IMAGE_WIDTH_, IMAGE_HEIGHT_, 5, 5, segment_type_);
  depth_map_ = surface.getDepthMap();
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (depth_map_[pixel] < 0 || depth_map_[pixel] > 10)
  //     cout << pixel << '\t' << depth_map_[pixel] << endl;
  normals_ = calcNormals(calcPointCloud(), IMAGE_WIDTH_, IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double scale = 0;
    for (int c = 0; c < 3; c++)
      scale += pow(normals_[pixel * 3 + c], 2);
    // if (abs(scale - 1) > 0.000001)
    //   cout << pixel << '\t' << scale << endl;
  }
}

vector<int> Segment::projectToOtherViewpoints(const int pixel, const double viewpoint_movement)
{
  vector<int> projected_pixels;
  int x = pixel % IMAGE_WIDTH_;
  int y = pixel / IMAGE_WIDTH_;
  // double u = x - CAMERA_PARAMETERS_[1];
  // double v = y - CAMERA_PARAMETERS_[2];
  // //double depth = 1 / ((plane(0) * u + plane(1) * v + plane(2)) / plane(3));
    
  // double disp = disp_plane_[0] * u + disp_plane_[1] * v + disp_plane_[2];
  //double depth = disp != 0 ? 1 / disp : 0;

  double depth = depth_map_[pixel];
  if (depth <= 0)
    return projected_pixels;
  int delta = round(viewpoint_movement / depth * CAMERA_PARAMETERS_[0]);
  if (x - delta >= 0)
    projected_pixels.push_back(pixel - delta);
  if (x + delta < IMAGE_WIDTH_)
    projected_pixels.push_back(pixel + delta + NUM_PIXELS_);
  if (y - delta >= 0)
    projected_pixels.push_back(pixel - delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 2);
  if (y + delta < IMAGE_HEIGHT_)
    projected_pixels.push_back(pixel + delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 3);
  return projected_pixels;
}

vector<double> Segment::calcPointCloud()
{
  vector<double> point_cloud(NUM_PIXELS_ * 3);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    double depth = depth_map_[pixel];
    point_cloud[pixel * 3 + 0] = (pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1]) / CAMERA_PARAMETERS_[0] * depth;
    point_cloud[pixel * 3 + 1] = (pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2]) / CAMERA_PARAMETERS_[0] * depth;
    point_cloud[pixel * 3 + 2] = depth;
  }
  return point_cloud;
}

int Segment::getSegmentType() const
{
  return segment_type_;
}


vector<int> Segment::deleteInvalidPixels(const vector<double> &point_cloud, const vector<int> &pixels)
{
  vector<int> new_pixels;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
    if (point_cloud[*pixel_it * 3 + 2] > 0)
      new_pixels.push_back(*pixel_it);
  return new_pixels;
}

double Segment::calcDistance(const vector<double> &point_cloud, const int pixel)
{
  if (point_cloud[pixel * 3 + 2] < 0)
    return input_statistics_.pixel_fitting_distance_threshold;
    
  if (segment_type_ == 0) {
    double distance = depth_plane_[3];
    for (int c = 0; c < 3; c++)
      distance -= depth_plane_[c] * point_cloud[pixel * 3 + c];
    return abs(distance);
  } else {
    double depth = depth_map_[pixel];
    if (depth <= 0)
      return input_statistics_.pixel_fitting_distance_threshold;
    vector<double> point(3);
    point[0] = (pixel % IMAGE_WIDTH_ - CAMERA_PARAMETERS_[1]) / CAMERA_PARAMETERS_[0] * depth;
    point[1] = (pixel / IMAGE_WIDTH_ - CAMERA_PARAMETERS_[2]) / CAMERA_PARAMETERS_[0] * depth;
    point[2] = depth;
    double distance = 0;
    for (int c = 0; c < 3; c++)
      distance += pow((point_cloud[pixel * 3 + c] - point[c]) * normals_[pixel * 3 + c], 2);
    return sqrt(distance);
  }
}
