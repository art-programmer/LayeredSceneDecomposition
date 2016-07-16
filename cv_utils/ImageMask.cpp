#include "cv_utils.h"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

namespace cv_utils
{
  ImageMask::ImageMask() : mask_(vector<bool>()), width_(0), height_(0)
  {
  }
  
  ImageMask::ImageMask(const vector<bool> &mask, const int width, const int height) : mask_(mask), width_(width), height_(height)
  {
  }

  ImageMask::ImageMask(const bool value, const int width, const int height) : mask_(vector<bool>(width * height, value)), width_(width), height_(height)
  {
  }
  
  ImageMask::ImageMask(const vector<int> &pixels, const int width, const int height) : width_(width), height_(height)
  {
    mask_.assign(width * height, false);
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      mask_[*pixel_it] = true;
  }
  
  ImageMask::ImageMask(const Mat &image)
  {
    readMaskImage(image);
  }
  
  ImageMask &ImageMask::operator = (const ImageMask &image_mask)
  {
    mask_ = image_mask.mask_;
    width_ = image_mask.width_;
    height_ = image_mask.height_;
    return *this;
  }
  
  void ImageMask::setMask(const vector<bool> &mask, const int width, const int height)
  {
    mask_ = mask;
    width_ = width;
    height_ = height;
  }
  
  void ImageMask::resizeByRatio(const double x_ratio, const double y_ratio)
  {
    const int new_width = round(width_ * x_ratio);
    const int new_height = round(height_ * y_ratio);
    resize(new_width, new_height);
  }
  
  void ImageMask::resize(const int new_width, const int new_height)
  {
    vector<bool> new_mask(new_width * new_height, false);
    for (int new_x = 0; new_x < new_width; new_x++) {
      for (int new_y = 0; new_y < new_height; new_y++) {
	const int ori_x = min(static_cast<int>(1.0 * new_x * width_ / new_width + 0.5), width_ - 1);
	const int ori_y = min(static_cast<int>(1.0 * new_y * height_ / new_height + 0.5), height_ - 1);
	new_mask[new_y * new_width + new_x] = mask_[ori_y * width_ + ori_x];
      }
    }
    
    mask_ = new_mask;
    width_ = new_width;
    height_ = new_height;
  }
  
  void ImageMask::resizeWithBias(const int new_width, const int new_height, const bool desired_value)
  {
    vector<bool> new_mask(new_width * new_height, false);
    for (int new_x = 0; new_x < new_width; new_x++) {
      for (int new_y = 0; new_y < new_height; new_y++) {
	const double ori_x = 1.0 * new_x * width_ / new_width;
	const double ori_y = 1.0 * new_y * height_ / new_height;
	int ori_x_1 = static_cast<int>(ori_x);
	int ori_x_2 = ori_x_1 + 1;
	int ori_y_1 = static_cast<int>(ori_y);
        int ori_y_2 = ori_y_1 + 1;
        if (mask_[ori_y_1 * width_ + ori_x_1] == desired_value || mask_[ori_y_1 * width_ + ori_x_2] == desired_value || mask_[ori_y_2 * width_ + ori_x_1] == desired_value || mask_[ori_y_2 * width_ + ori_x_2] == desired_value)
	  new_mask[new_y * new_width + new_x] = desired_value;
        else
	  new_mask[new_y * new_width + new_x] = !desired_value;
      }
    }
    
    mask_ = new_mask;
    width_ = new_width;
    height_ = new_height;
  }
  
  void ImageMask::dilate(const int num_iterations, const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM)
  {
    for (int iteration = 0; iteration < num_iterations; iteration++) {
      vector<bool> new_mask = mask_;
      for (int pixel = 0; pixel < width_ * height_; pixel++) {
	if (mask_[pixel] == false)
	  continue;
        vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	  new_mask[*neighbor_pixel_it] = true;
      }
      mask_ = new_mask;
    }
  }
  
  void ImageMask::erode(const int num_iterations, const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM)
  {
    for (int iteration = 0; iteration < num_iterations; iteration++) {
      vector<bool> new_mask = mask_;
      for (int pixel = 0; pixel < width_ * height_; pixel++) {
        if (mask_[pixel] == false)
          continue;
        vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (mask_[*neighbor_pixel_it] == false) {
	    new_mask[pixel] = false;
	    break;
	  }
	}
      }
      mask_ = new_mask;
    }
  }
  
  vector<int> ImageMask::getPixels() const
  {
    vector<int> pixels;
    for (int pixel = 0; pixel < width_ * height_; pixel++)
      if (mask_[pixel] == true)
	pixels.push_back(pixel);
    return pixels;
  }
  
  int ImageMask::getNumPixels() const
  {
    int num_pixels = 0;
    for (int pixel = 0; pixel < width_ * height_; pixel++)
      if (mask_[pixel] == true)
	num_pixels++;
    return num_pixels;
  }
  
  vector<double> ImageMask::getCenter() const
  {
    double x_sum = 0, y_sum = 0;
    int num_pixels = 0;
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (mask_[pixel] == false)
	continue;
      x_sum += pixel % width_;
      y_sum += pixel / width_;
      num_pixels++;
    }
    vector<double> center(2);
    center[0] = x_sum / num_pixels;
    center[1] = y_sum / num_pixels;
    return center;
  }
  
  // vector<bool> ImageMask::getMaskVec()
  // {
  //   return mask_;
  // }
  
  Mat ImageMask::drawMaskImage(const int num_channels) const
  {
    Mat mask_image = num_channels == 1 ? Mat::zeros(height_, width_, CV_8UC1) : Mat::zeros(height_, width_, CV_8UC3);
    for (int pixel = 0; pixel < width_ * height_; pixel++)
      if (mask_[pixel] == true)
	if (num_channels == 1)
	  mask_image.at<uchar>(pixel / width_, pixel % width_) = 255;
        else
	  mask_image.at<Vec3b>(pixel / width_, pixel % width_) = Vec3b(255, 255, 255);
    return mask_image;
  }
  
  cv::Mat ImageMask::drawImageWithMask(const cv::Mat &image, const bool use_mask_color, const Vec3b mask_color, const bool use_outside_color, const Vec3b outside_color) const
  {
    assert(image.cols == width_ && image.rows == height_);
    
    Mat image_with_mask = image.clone();
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (use_mask_color && mask_[pixel])
	image_with_mask.at<Vec3b>(pixel / width_, pixel % width_) = mask_color;
      if (use_outside_color && mask_[pixel] == false)
        image_with_mask.at<Vec3b>(pixel / width_, pixel % width_) = outside_color;
    }
    return image_with_mask;
  }
  
  bool ImageMask::at(const int &pixel) const
  {
    return mask_[pixel];
  }
  
  void ImageMask::set(const int &pixel, const bool value)
  {
    mask_[pixel] = value;
  }
  
  vector<double> ImageMask::calcDistanceMapOutside(const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM) const
  {
    vector<double> distance_map(width_ * height_, 1000000);
    vector<int> border_pixels;
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (mask_[pixel] == false)
	continue;
      distance_map[pixel] = 0;
      
      vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	if (mask_[*neighbor_pixel_it] == false) {
	  border_pixels.push_back(pixel);
	  break;
	}
      }
    }
    
    while (border_pixels.size() > 0) {
      vector<int> new_border_pixels;
      for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
	int pixel = *border_pixel_it;
	double distance = distance_map[pixel];
	vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  int neighbor_pixel = *neighbor_pixel_it;
	  double distance_delta = sqrt(pow(*neighbor_pixel_it % width_ - pixel % width_, 2) + pow(*neighbor_pixel_it / width_ - pixel / width_, 2));
	  if (distance + distance_delta < distance_map[neighbor_pixel]) {
	    distance_map[neighbor_pixel] = distance + distance_delta;
	    new_border_pixels.push_back(neighbor_pixel);
	  }
	}
      }
      border_pixels = new_border_pixels;
    }
    
    return distance_map;
  }
  
  vector<double> ImageMask::calcDistanceMapInside(const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM) const
  {
    vector<double> distance_map(width_ * height_, 1000000);
    vector<int> border_pixels;
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (mask_[pixel] == true)
	continue;
      distance_map[pixel] = 0;
      
      vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	if (mask_[*neighbor_pixel_it] == true) {
	  border_pixels.push_back(pixel);
	  break;
	}
      }
    }
    
    while (border_pixels.size() > 0) {
      vector<int> new_border_pixels;
      for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
	int pixel = *border_pixel_it;
	double distance = distance_map[pixel];
	vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  int neighbor_pixel = *neighbor_pixel_it;
	  double distance_delta = sqrt(pow(*neighbor_pixel_it % width_ - pixel % width_, 2) + pow(*neighbor_pixel_it / width_ - pixel / width_, 2));
	  if (distance + distance_delta < distance_map[neighbor_pixel]) {
	    distance_map[neighbor_pixel] = distance + distance_delta;
	    new_border_pixels.push_back(neighbor_pixel);
	  }
	}
      }
      border_pixels = new_border_pixels;
    }
    
    return distance_map;
  }
  
  void ImageMask::calcBoundaryDistanceMap(vector<int> &boundary_map, vector<double> &distance_map, const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM) const
  {
    boundary_map.assign(width_ * height_, -1);
    distance_map.assign(width_ * height_, 1000000);
    
    vector<int> border_pixels;
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (mask_[pixel] == false)
        continue;
      
      vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        if (mask_[*neighbor_pixel_it] == false) {
          border_pixels.push_back(pixel);
	  boundary_map[pixel] = pixel;
	  distance_map[pixel] = 0;
          break;
        }
      }
    }
    
    while (border_pixels.size() > 0) {
      vector<int> new_border_pixels;
      for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
        int pixel = *border_pixel_it;
        double distance = distance_map[pixel];
        vector<int> neighbor_pixels = findNeighbors(pixel, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          int neighbor_pixel = *neighbor_pixel_it;
          double distance_delta = sqrt(pow(*neighbor_pixel_it % width_ - pixel % width_, 2) + pow(*neighbor_pixel_it / width_ - pixel / width_, 2));
          if (distance + distance_delta < distance_map[neighbor_pixel]) {
	    boundary_map[neighbor_pixel] = boundary_map[pixel];
            distance_map[neighbor_pixel] = distance + distance_delta;
            new_border_pixels.push_back(neighbor_pixel);
          }
        }
      }
      border_pixels = new_border_pixels;
    }
  }
  
  std::vector<std::vector<int> > ImageMask::findConnectedComponents(const bool USE_PANORAMA, const int NEIGHBOR_SYSTEM)
  {
    vector<bool> visited_pixel_mask(width_ * height_, false);
    vector<vector<int> > connected_components;
    for (int pixel = 0; pixel < width_ * height_; pixel++) {
      if (mask_[pixel] == false || visited_pixel_mask[pixel] == true)
        continue;
      
      vector<int> connected_component;
      vector<int> border_pixels;
      border_pixels.push_back(pixel);
      visited_pixel_mask[pixel] = true;
      while (true) {
        vector<int> new_border_pixels;
        for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
          connected_component.push_back(*border_pixel_it);
          vector<int> neighbor_pixels = findNeighbors(*border_pixel_it, width_, height_, USE_PANORAMA, NEIGHBOR_SYSTEM);
          for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
            if (mask_[*neighbor_pixel_it] == true && visited_pixel_mask[*neighbor_pixel_it] == false) {
              new_border_pixels.push_back(*neighbor_pixel_it);
              visited_pixel_mask[*neighbor_pixel_it] = true;
            }
          }
        }
        if (new_border_pixels.size() == 0)
          break;
        border_pixels = new_border_pixels;
      }
      connected_components.push_back(connected_component);
    }  
    
    sort(connected_components.begin(), connected_components.end(), [](const vector<int> &vec_1, const vector<int> &vec_2) { return vec_1.size() > vec_2.size(); });
    return connected_components;
  }
  
  void ImageMask::addPixels(const vector<int> &pixels)
  {
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      mask_[*pixel_it] = true;
  }
  
  void ImageMask::subtractPixels(const vector<int> &pixels)
  {
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      mask_[*pixel_it] = true;
  }
  
  ImageMask &ImageMask::operator +=(const ImageMask &image_mask)
  {
    vector<int> pixels = image_mask.getPixels();
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      mask_[*pixel_it] = true;
    return *this;
  }
  
  ImageMask &ImageMask::operator -=(const ImageMask &image_mask)
  {
    vector<int> pixels = image_mask.getPixels();
    for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
      mask_[*pixel_it] = false;
    return *this;
  }
  
  ImageMask operator +(const ImageMask &image_mask_1, const ImageMask &image_mask_2)
  {
    ImageMask result = image_mask_1;
    result += image_mask_2;
    return result;
  }
  
  ImageMask operator -(const ImageMask &image_mask_1, const ImageMask &image_mask_2)
  {
    ImageMask result = image_mask_1;
    result -= image_mask_2;
    return result;
  }
  
  ostream & operator <<(ostream &out_str, const ImageMask &image_mask)
  {
    out_str << image_mask.width_ << '\t' << image_mask.height_ << '\t' << image_mask.getNumPixels() << endl;
    for (int y = 0; y < image_mask.height_; y++) {
      for (int x = 0; x < image_mask.width_; x++)
	if (image_mask.mask_[y * image_mask.width_ + x])
	  out_str << y * image_mask.width_ + x << ' ';
      //	out_str << image_mask.mask_[y * image_mask.width_ + x] << ' ';
      //out_str << endl;
    }
    return out_str;
  }
  
  istream & operator >>(istream &in_str, ImageMask &image_mask)
  {
    int num_pixels = -1;
    in_str >> image_mask.width_ >> image_mask.height_ >> num_pixels;
    image_mask.mask_.assign(image_mask.width_ * image_mask.height_, false);
    for (int i = 0; i < num_pixels; i++) {
      int pixel;
      in_str >> pixel;
      image_mask.mask_[pixel] = true;
    }
    // for (int y = 0; y < image_mask.height_; y++) {
    //   for (int x = 0; x < image_mask.width_; x++) {
    // 	int value;
    // 	in_str >> value;
    //     image_mask.mask_[y * image_mask.width_ + x] = value != 0;
    //   }
    // }
    return in_str;
  }
  
  void ImageMask::smooth(const string type, const int window_size, const double sigma)
  {
    Mat mask_image = drawMaskImage();
    if (type == "median")
      medianBlur(mask_image, mask_image, window_size);
    else if (type == "Gaussian")
      GaussianBlur(mask_image, mask_image, Size(window_size, window_size), sigma);
    readMaskImage(mask_image);
  }
  
  void ImageMask::readMaskImage(const Mat &mask_image)
  {
    Mat mask_image_gray;
    if (mask_image.channels() == 1)
      mask_image_gray = mask_image.clone();
    else
      cvtColor(mask_image, mask_image_gray, CV_BGR2GRAY);
    
    width_ = mask_image.cols;
    height_ = mask_image.rows;
    mask_.assign(width_ * height_, false);
    for (int pixel = 0; pixel < width_ * height_; pixel++)
      mask_[pixel] = mask_image_gray.at<uchar>(pixel / width_, pixel % width_) >= 128;
  }

  std::vector<int> ImageMask::findMaskWindowPixels(const int pixel, const int WINDOW_SIZE, const int USE_PANORAMA) const
  {
    vector<int> window_pixels = findWindowPixels(pixel, width_, height_, WINDOW_SIZE, USE_PANORAMA);
    vector<int> mask_window_pixels;
    for (vector<int>::const_iterator window_pixel_it = window_pixels.begin(); window_pixel_it != window_pixels.end(); window_pixel_it++)
      if (mask_[*window_pixel_it])
	mask_window_pixels.push_back(*window_pixel_it);
    return mask_window_pixels;
  }
  
}

