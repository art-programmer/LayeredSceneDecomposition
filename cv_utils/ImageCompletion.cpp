#include <map>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "ImageMask.h"
#include "cv_utils.h"

//#include "ImageCompletionCostFunctor.h"
//#include "ImageCompletionProposalGenerator.h"
//#include "FusionSpaceSolver.h"

using namespace std;
using namespace cv;
using namespace Eigen;
namespace cv_utils
{
  
namespace
{
  vector<int> calcCommonWindowOffsets(const ImageMask &mask_1, const ImageMask &mask_2, const int pixel_1, const int pixel_2, const int image_width, const int image_height, const int WINDOW_SIZE)
  {
    vector<int> window_offsets;
    for (int offset_x = -(WINDOW_SIZE - 1) / 2; offset_x <= (WINDOW_SIZE - 1) / 2; offset_x++)
      for (int offset_y = -(WINDOW_SIZE - 1) / 2; offset_y <= (WINDOW_SIZE - 1) / 2; offset_y++)
        if (pixel_1 % image_width + offset_x >= 0 && pixel_1 % image_width + offset_x < image_width && pixel_1 / image_width + offset_y >= 0 && pixel_1 / image_width + offset_y < image_height
            && pixel_2 % image_width + offset_x >= 0 && pixel_2 % image_width + offset_x < image_width && pixel_2 / image_width + offset_y >= 0 && pixel_2 / image_width + offset_y < image_height
            && mask_1.at(pixel_1 + offset_y * image_width + offset_x) == true && mask_2.at(pixel_2 + offset_y * image_width + offset_x) == true)
          window_offsets.push_back((offset_y + (WINDOW_SIZE - 1) / 2) * WINDOW_SIZE + (offset_x + (WINDOW_SIZE - 1) / 2));
    return window_offsets;
  }
  
  double calcPatchDistance(const Mat &source_image, const Mat &target_image, const ImageMask &source_mask, const ImageMask &target_mask, const int source_pixel, const int target_pixel, const int WINDOW_SIZE, const vector<double> &source_distance_map)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    const double CONFIDENT_PIXEL_WEIGHT = 100;
    
    vector<int> common_window_offsets = calcCommonWindowOffsets(target_mask, target_mask, source_pixel, target_pixel, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW_SIZE);
    double SSD = 0;
    double sum_confidence = 0;
    vector<bool> used_offsets(WINDOW_SIZE * WINDOW_SIZE, false);
    for (vector<int>::const_iterator window_offset_it = common_window_offsets.begin(); window_offset_it != common_window_offsets.end(); window_offset_it++) {
      used_offsets[*window_offset_it] = true;
      int offset_x = *window_offset_it % WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      int offset_y = *window_offset_it / WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      Vec3b color_1 = target_image.at<Vec3b>(source_pixel / IMAGE_WIDTH + offset_y, source_pixel % IMAGE_WIDTH + offset_x);
      Vec3b color_2 = target_image.at<Vec3b>(target_pixel / IMAGE_WIDTH + offset_y, target_pixel % IMAGE_WIDTH + offset_x);
      
      double confidence = 1; //pow(1.3, -source_distance_map[(target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x]);
      if (source_mask.at((target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x))
	confidence = CONFIDENT_PIXEL_WEIGHT;
      //double confidence = source_mask.at((target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x) ? CONFIDENT_PIXEL_WEIGHT : 1;
      
      for (int c = 0; c < 3; c++)
        SSD += pow(1.0 * (color_1[c] - color_2[c]) / 255, 2) * confidence;
      sum_confidence += confidence;
    }
    
    vector<int> target_window_offsets = calcCommonWindowOffsets(target_mask, target_mask, target_pixel, target_pixel, IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW_SIZE);
    for (vector<int>::const_iterator window_offset_it = target_window_offsets.begin(); window_offset_it != target_window_offsets.end(); window_offset_it++) {
      if (used_offsets[*window_offset_it] == true)
        continue;
      
      int offset_x = *window_offset_it % WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      int offset_y = *window_offset_it / WINDOW_SIZE - (WINDOW_SIZE - 1) / 2;
      
      double confidence = 1; //pow(1.3, -source_distance_map[(target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x]);
      if (source_mask.at((target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x))
        confidence = CONFIDENT_PIXEL_WEIGHT;
      //double confidence = source_mask.at((target_pixel / IMAGE_WIDTH + offset_y) * IMAGE_WIDTH + target_pixel % IMAGE_WIDTH + offset_x) ? CONFIDENT_PIXEL_WEIGHT : 1;
      
      SSD += 3 * confidence;
      sum_confidence += confidence;
    }
    
    if (sum_confidence == 0)
      return 1;
    
    double distance = SSD / (sum_confidence * 3);
    return distance;
  }
  
  void findBetterNearestNeighbor(const Mat &source_image, const Mat &target_image, const ImageMask &source_mask, const ImageMask &target_mask, vector<int> &nearest_neighbor_field, vector<double> &distance_field, const int pixel, const int direction, const int WINDOW_SIZE, const std::vector<double> &source_distance_map)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    
    int x = pixel % IMAGE_WIDTH;
    int y = pixel / IMAGE_WIDTH;
    
    int current_nearest_neighbor = nearest_neighbor_field[pixel];
    int current_nearest_neighbor_x = current_nearest_neighbor % IMAGE_WIDTH;
    int current_nearest_neighbor_y = current_nearest_neighbor / IMAGE_WIDTH;
    double current_distance = distance_field[pixel];
    
    int best_nearest_neighbor = current_nearest_neighbor;
    double min_distance = current_distance;
    
    if (x + direction >= 0 && x + direction < IMAGE_WIDTH && target_mask.at(pixel + direction)) {
      int neighbor = nearest_neighbor_field[pixel + direction];
      if (neighbor != -1 && neighbor % IMAGE_WIDTH - direction >= 0 && neighbor % IMAGE_WIDTH - direction < IMAGE_WIDTH) {
        if (source_mask.at(neighbor - direction) == true) {
          double distance = calcPatchDistance(source_image, target_image, source_mask, target_mask, neighbor - direction, pixel, WINDOW_SIZE, source_distance_map);
          if (distance < min_distance) {
            best_nearest_neighbor = neighbor - direction;
            min_distance = distance;
          }
        } else {
          double distance = calcPatchDistance(source_image, target_image, source_mask, target_mask, neighbor, pixel, WINDOW_SIZE, source_distance_map);
          if (distance < min_distance) {
            best_nearest_neighbor = neighbor;
            min_distance = distance;
          }
        }       
      }
    }
    
    if (y + direction >= 0 && y + direction < IMAGE_HEIGHT && target_mask.at(pixel + direction * IMAGE_WIDTH)) {
      int neighbor = nearest_neighbor_field[pixel + direction * IMAGE_WIDTH];
      if (neighbor != -1 && neighbor / IMAGE_WIDTH - direction >= 0 && neighbor / IMAGE_WIDTH - direction < IMAGE_HEIGHT) {
        if (source_mask.at(neighbor - direction * IMAGE_WIDTH) == true) {
          double distance = calcPatchDistance(source_image, target_image, source_mask, target_mask, neighbor - direction * IMAGE_WIDTH, pixel, WINDOW_SIZE, source_distance_map);
          if (distance < min_distance) {
            best_nearest_neighbor = neighbor - direction * IMAGE_WIDTH;
            min_distance = distance;
          }
        } else {
          double distance = calcPatchDistance(source_image, target_image, source_mask, target_mask, neighbor, pixel, WINDOW_SIZE, source_distance_map);
          if (distance < min_distance) {
            best_nearest_neighbor = neighbor;
            min_distance = distance;
          }
        }
      }
    }
    
    int radius = max(IMAGE_WIDTH, IMAGE_HEIGHT);
    int num_attempts = 0;
    while (radius > 0) {
      int x = max(min(current_nearest_neighbor_x + (rand() % (radius * 2 + 1) - radius), IMAGE_WIDTH - 1), 0);
      int y = max(min(current_nearest_neighbor_y + (rand() % (radius * 2 + 1) - radius), IMAGE_HEIGHT - 1), 0);
      int neighbor = y * IMAGE_WIDTH + x;
      if (source_mask.at(neighbor) == false) {
        num_attempts++;
        radius--;
        // if (num_attempts > radius * radius)
        //      break;
        continue;
      }
      double distance = calcPatchDistance(source_image, target_image, source_mask, target_mask, neighbor, pixel, WINDOW_SIZE, source_distance_map);
      if (distance < min_distance) {
        best_nearest_neighbor = neighbor;
        min_distance = distance;
      }
      radius /= 2;
    }
    if (best_nearest_neighbor != current_nearest_neighbor)
      //cout << pixel % IMAGE_WIDTH << ' ' << pixel / IMAGE_WIDTH << '\t' << best_nearest_neighbor % IMAGE_WIDTH << ' ' << best_nearest_neighbor / IMAGE_WIDTH << '\t' << min_distance << '\t' << current_nearest_neighbor % IMAGE_WIDTH << ' ' << current_nearest_neighbor / IMAGE_WIDTH << '\t' << distance_field[pixel] << endl;
    nearest_neighbor_field[pixel] = best_nearest_neighbor;
    distance_field[pixel] = min_distance;
  }
  
  void calcNearestNeighborField(const Mat &source_image, const Mat &target_image, const ImageMask &source_mask, const ImageMask &target_mask, vector<int> &nearest_neighbor_field, vector<double> &distance_field, const int WINDOW_SIZE)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    
    const int NUM_ITERATIONS = 5;
    
    //vector<double> previous_distance_field = distance_field;
    const double DISTANCE_THRESHOLD = 0.000001;
    
    const vector<double> source_distance_map = source_mask.calcDistanceMapOutside();
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
      int direction = 1;
      for (int step = 0; step <= max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; step++) {
        for (int i = 0; i < max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; i++) {
          int target_x = IMAGE_WIDTH - 1 - (step - 1 - i);
          int target_y = IMAGE_HEIGHT - 1 - i;
          if (target_x < 0 || target_x >= IMAGE_WIDTH || target_y < 0 || target_y >= IMAGE_HEIGHT)
            continue;
          //cout << target_x << '\t' << target_y << '\t' << IMAGE_WIDTH << '\t' << IMAGE_HEIGHT << '\t' << step << '\t' << i << endl;
          int target_pixel = target_y * IMAGE_WIDTH + target_x;
          if (target_mask.at(target_pixel) == true && distance_field[target_pixel] > DISTANCE_THRESHOLD)
            findBetterNearestNeighbor(source_image, target_image, source_mask, target_mask, nearest_neighbor_field, distance_field, target_pixel, direction, WINDOW_SIZE, source_distance_map);
        }
      }
      
      direction = -1;
      for (int step = 0; step <= max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; step++) {
        for (int i = 0; i < max(IMAGE_WIDTH, IMAGE_HEIGHT) * 2; i++) {
          int target_x = step - 1 - i;
          int target_y = i;
          //    cout << target_x << '\t' << target_y << endl;
          if (target_x < 0 || target_x >= IMAGE_WIDTH || target_y < 0 || target_y >= IMAGE_HEIGHT)
            continue;
          int target_pixel = target_y * IMAGE_WIDTH + target_x;
          if (target_mask.at(target_pixel) == true && distance_field[target_pixel] > DISTANCE_THRESHOLD)
            findBetterNearestNeighbor(source_image, target_image, source_mask, target_mask, nearest_neighbor_field, distance_field, target_pixel, direction, WINDOW_SIZE, source_distance_map);
        }
      }
    }
  }
  
  void updateColor(Mat &target_image, const int pixel, const vector<Vec3b> &color_values, const vector<double> &distances, const vector<double> &distances_2D, const double distance_var)
  {
    //assert(color_values.size() == distances.size());
    vector<vector<double> > histos(3, vector<double>(256, 0));
    double sum_weights = 0;
    for (int index = 0; index < color_values.size(); index++) {
      double distance = distances[index];
      //double weight = exp(-distance / (2 * distance_var)) * pow(1.3, -distances_2D[index]);
      double weight = (1 - distance) * pow(1.3, -distances_2D[index]);
      Vec3b color = color_values[index];
      for (int c = 0; c < 3; c++)
        histos[c][color[c]] += weight;
      sum_weights += weight;
    }
    
    Vec3b new_color;
    double CDF_lower_threshold = 0.3 * sum_weights;
    double CDF_higher_threshold = 0.7 * sum_weights;
    for (int c = 0; c < 3; c++) {
      double CDF = 0;
      double sum_effective_weighted_color_values = 0;
      double sum_effective_weights = 0;
      for (int color_value = 0; color_value < 256; color_value++) {
        CDF += histos[c][color_value];
        if (CDF < CDF_lower_threshold)
          continue;
        sum_effective_weighted_color_values += color_value * histos[c][color_value];
        sum_effective_weights += histos[c][color_value];
        if (CDF > CDF_higher_threshold)
          break;
      }
      new_color[c] = max(min(round(sum_effective_weighted_color_values / sum_effective_weights), 255.0), 0.0);
    }
    target_image.at<Vec3b>(pixel / target_image.cols, pixel % target_image.cols) = new_color;
    //target_image.at<Vec3b>(pixel / target_image.cols, pixel % target_image.cols) = color_values[color_values.size() / 2];
  }
  
  Mat calcTargetImage(const Mat &source_image, const Mat &target_image, const ImageMask &source_mask, const ImageMask &target_mask, const Mat &lower_level_source_image, const ImageMask &lower_level_source_mask, const ImageMask &lower_level_target_mask, const int WINDOW_SIZE)
  {
    const int IMAGE_WIDTH = source_image.cols;
    const int IMAGE_HEIGHT = source_image.rows;
    
    bool require_upscale = source_image.cols != lower_level_source_image.cols;
    
    vector<int> source_pixels = source_mask.getPixels();
    vector<int> target_pixels = target_mask.getPixels();
    
    vector<double> distance_map = source_mask.calcDistanceMapOutside();
    
    const int NUM_ITERATIONS = 15;

    const vector<double> source_distance_map = source_mask.calcDistanceMapOutside();
    
    Mat current_target_image = target_image.clone();
    for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
      //cout << iteration << endl;
      vector<int> nearest_neighbor_field(IMAGE_WIDTH * IMAGE_HEIGHT, -1);
      vector<double> distance_field(IMAGE_WIDTH * IMAGE_HEIGHT, 1);
      for (vector<int>::const_iterator pixel_it = target_pixels.begin(); pixel_it != target_pixels.end(); pixel_it++) {
	if (source_mask.at(*pixel_it) == true) {
	  nearest_neighbor_field[*pixel_it] = *pixel_it;
	  distance_field[*pixel_it] = 0; //calcPatchDistance(source_image, current_target_image, source_mask, target_mask, pixel, pixel);
        } else {
	  nearest_neighbor_field[*pixel_it] = source_pixels[rand() % source_pixels.size()];
	  distance_field[*pixel_it] = calcPatchDistance(source_image, current_target_image, source_mask, target_mask, nearest_neighbor_field[*pixel_it], *pixel_it, WINDOW_SIZE, source_distance_map);
	}
      }
      calcNearestNeighborField(source_image, current_target_image, source_mask, target_mask, nearest_neighbor_field, distance_field, WINDOW_SIZE);
      
      vector<double> distances_vec;
      for (vector<int>::const_iterator pixel_it = target_pixels.begin(); pixel_it != target_pixels.end(); pixel_it++) {
        if (source_mask.at(*pixel_it) == true)
          continue;
        distances_vec.push_back(distance_field[*pixel_it]);
      }
      double distance_var = max(pow(calcMeanAndSVar(distances_vec)[1], 2), 0.000001);
      // sort(sorted_distances.begin(), sorted_distances.end());
      // double distance_var = sorted_distances[round(0.75 * sorted_distances.size())];
      
      // for (vector<int>::const_iterator pixel_it = target_pixels.begin(); pixel_it != target_pixels.end(); pixel_it++) {
      //   if (nearest_neighbor_field[pixel] == -1 || source_mask[nearest_neighbor_field[pixel]] == false)
      //     cout << pixel << '\t' << nearest_neighbor_field[pixel] << endl;
      // }
      
      
      if (iteration < NUM_ITERATIONS || require_upscale == false) {
	for (vector<int>::const_iterator pixel_it = target_pixels.begin(); pixel_it != target_pixels.end(); pixel_it++) {
	  if (source_mask.at(*pixel_it) == true)
            continue;
          vector<int> window_pixels = target_mask.findMaskWindowPixels(*pixel_it, WINDOW_SIZE);
          // window_pixels.clear();
          // window_pixels.push_back(*pixel_it);
          vector<Vec3b> color_values;
          vector<double> distances;
	  vector<double> distances_2D;
          for (vector<int>::const_iterator window_pixel_it = window_pixels.begin(); window_pixel_it != window_pixels.end(); window_pixel_it++) {
            int nearest_neighbor = nearest_neighbor_field[*window_pixel_it];
            if (nearest_neighbor == -1 || source_mask.at(nearest_neighbor) == false)
              continue;
            int x_offset = *pixel_it % IMAGE_WIDTH - *window_pixel_it % IMAGE_WIDTH;
            int y_offset = *pixel_it / IMAGE_WIDTH - *window_pixel_it / IMAGE_WIDTH;
            if (nearest_neighbor % IMAGE_WIDTH + x_offset < 0 || nearest_neighbor % IMAGE_WIDTH + x_offset >= IMAGE_WIDTH || nearest_neighbor / IMAGE_WIDTH + y_offset < 0 || nearest_neighbor / IMAGE_WIDTH + y_offset >= IMAGE_HEIGHT || source_mask.at((nearest_neighbor / IMAGE_WIDTH + y_offset) * IMAGE_WIDTH + nearest_neighbor % IMAGE_WIDTH + x_offset) == false)
              continue;
            color_values.push_back(source_image.at<Vec3b>(nearest_neighbor / IMAGE_WIDTH + y_offset, nearest_neighbor % IMAGE_WIDTH + x_offset));
            distances.push_back(distance_field[*window_pixel_it]);
	    distances_2D.push_back(distance_map[*window_pixel_it]);
          }
	  updateColor(current_target_image, *pixel_it, color_values, distances, distances_2D, distance_var);
        }
	
	bool save_matching_images = false;
        if (save_matching_images) {
          static int index = 0;
          const int IMAGE_PADDING = 10;
          Mat nearest_neighbor_image(IMAGE_HEIGHT, IMAGE_WIDTH * 2 + IMAGE_PADDING, CV_8UC3);
          nearest_neighbor_image.setTo(Scalar(255, 255, 255));
          Mat target_region(nearest_neighbor_image, Rect(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
          current_target_image.copyTo(target_region);
          Mat source_region(nearest_neighbor_image, Rect(IMAGE_WIDTH + IMAGE_PADDING, 0, IMAGE_WIDTH, IMAGE_HEIGHT));
          source_image.copyTo(source_region);
          for (vector<int>::const_iterator pixel_it = target_pixels.begin(); pixel_it != target_pixels.end(); pixel_it++) {
            int nearest_neighbor = nearest_neighbor_field[*pixel_it];
            if (source_mask.at(*pixel_it) == true) {
              assert(*pixel_it == nearest_neighbor);
              continue;
            }
            if (rand() % (IMAGE_WIDTH / 5) == 0)
              line(nearest_neighbor_image, Point(*pixel_it % IMAGE_WIDTH, *pixel_it / IMAGE_WIDTH), Point(nearest_neighbor % IMAGE_WIDTH + IMAGE_WIDTH + IMAGE_PADDING, nearest_neighbor / IMAGE_WIDTH), Scalar(0, 0, 255));
          }
          stringstream nearest_neighbor_image_filename;
          nearest_neighbor_image_filename << "Test/nearest_neighbor_image_" << index << ".bmp";
          //imwrite(nearest_neighbor_image_filename.str(), nearest_neighbor_image);
          index++;
        }
      } else {
        const int LOWER_LEVEL_IMAGE_WIDTH = lower_level_source_image.cols;
        const int LOWER_LEVEL_IMAGE_HEIGHT = lower_level_source_image.rows;
        Mat lower_level_target_image = lower_level_source_image.clone();
        for (int lower_level_pixel = 0; lower_level_pixel < LOWER_LEVEL_IMAGE_WIDTH * LOWER_LEVEL_IMAGE_HEIGHT; lower_level_pixel++) {
          if (lower_level_target_mask.at(lower_level_pixel) == false || lower_level_source_mask.at(lower_level_pixel) == true)
            continue;
	  
          vector<int> window_pixels = target_mask.findMaskWindowPixels(convertPixel(lower_level_pixel, LOWER_LEVEL_IMAGE_WIDTH, LOWER_LEVEL_IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT), WINDOW_SIZE);
          //        window_pixels.clear();
          //        window_pixels.push_back(pixel);
          vector<Vec3b> color_values;
          vector<double> distances;
	  vector<double> distances_2D;
          for (vector<int>::const_iterator window_pixel_it = window_pixels.begin(); window_pixel_it != window_pixels.end(); window_pixel_it++) {
            if (target_mask.at(*window_pixel_it) == false)
              continue;
            int nearest_neighbor = nearest_neighbor_field[*window_pixel_it];
            if (nearest_neighbor == -1)
              continue;
	    
	    int lower_level_window_pixel = convertPixel(*window_pixel_it, IMAGE_WIDTH, IMAGE_HEIGHT, LOWER_LEVEL_IMAGE_WIDTH, LOWER_LEVEL_IMAGE_HEIGHT);
	    
            int x_offset = lower_level_pixel % LOWER_LEVEL_IMAGE_WIDTH - lower_level_window_pixel % LOWER_LEVEL_IMAGE_WIDTH;
            int y_offset = lower_level_pixel / LOWER_LEVEL_IMAGE_WIDTH - lower_level_window_pixel / LOWER_LEVEL_IMAGE_WIDTH;
	    
	    // if (x_offset % 2 == 1 || y_offset % 2 == 1)
	    //   continue;
	    int lower_level_nearest_neighbor = convertPixel(nearest_neighbor, IMAGE_WIDTH, IMAGE_HEIGHT, LOWER_LEVEL_IMAGE_WIDTH, LOWER_LEVEL_IMAGE_HEIGHT);
            int lower_level_nearest_neighbor_x = lower_level_nearest_neighbor % LOWER_LEVEL_IMAGE_WIDTH;
            int lower_level_nearest_neighbor_y = lower_level_nearest_neighbor / LOWER_LEVEL_IMAGE_WIDTH;
	    
            if (lower_level_nearest_neighbor_x + x_offset < 0 || lower_level_nearest_neighbor_x + x_offset >= LOWER_LEVEL_IMAGE_WIDTH || lower_level_nearest_neighbor_y + y_offset < 0 || lower_level_nearest_neighbor_y + y_offset >= LOWER_LEVEL_IMAGE_HEIGHT || lower_level_source_mask.at((lower_level_nearest_neighbor_y + y_offset) * LOWER_LEVEL_IMAGE_WIDTH + (lower_level_nearest_neighbor_x + x_offset)) == false)
              continue;
	    color_values.push_back(lower_level_source_image.at<Vec3b>(lower_level_nearest_neighbor_y + y_offset, lower_level_nearest_neighbor_x + x_offset));
            distances.push_back(distance_field[*window_pixel_it]);
	    distances_2D.push_back(distance_map[*window_pixel_it]);
          }
	  updateColor(lower_level_target_image, lower_level_pixel, color_values, distances, distances_2D, distance_var);
        }
	current_target_image = lower_level_target_image.clone();
	
      }
    }
    
    // if (source_image.cols == 125) {
    //   imwrite("Test/image_with_source_mask.bmp", source_mask.drawImageWithMask(target_image));
    //   imwrite("Test/target_image.bmp", current_target_image);
    //   //exit(1);
    // }
    
    resize(current_target_image, current_target_image, Size(lower_level_source_image.cols, lower_level_source_image.rows));
    
    // for (int pixel = 0; pixel < lower_level_source_mask.size(); pixel++)
    //   if (lower_level_source_mask[pixel] == true)
    //     current_target_image.at<Vec3b>(pixel / lower_level_source_image.cols, pixel % lower_level_source_image.cols) = lower_level_source_image.at<Vec3b>(pixel / lower_level_source_image.cols, pixel % lower_level_source_image.cols);
    return current_target_image;
  }
}
  
  
  cv::Mat completeImage(const cv::Mat &input_image, const vector<bool> &input_source_mask, const vector<bool> &input_target_mask, const int WINDOW_SIZE, const Matrix3d &unwarp_transform)
  {
    return completeImage(input_image, ImageMask(input_source_mask, input_image.cols, input_image.rows), ImageMask(input_target_mask, input_image.cols, input_image.rows), WINDOW_SIZE, unwarp_transform);
  }
  
  cv::Mat completeImage(const cv::Mat &input_image, const ImageMask &input_source_mask, const ImageMask &input_target_mask, const int WINDOW_SIZE, const Matrix3d &unwarp_transform)
  {
    int min_x = 1000000, max_x = -1000000, min_y = 1000000, max_y = -1000000;
    for (int pixel = 0; pixel < input_image.cols * input_image.rows; pixel++) {
      if (input_target_mask.at(pixel) == false)
	continue;
      Vector3d pixel_vec;
      pixel_vec << pixel % input_image.cols, pixel / input_image.cols, 1;
      Vector3d unwarped_pixel_vec = unwarp_transform * pixel_vec;
      if (unwarped_pixel_vec[2] == 0)
	continue;
      int x = round(unwarped_pixel_vec[0] / unwarped_pixel_vec[2]);
      int y = round(unwarped_pixel_vec[1] / unwarped_pixel_vec[2]);
      if (x < min_x)
	min_x = x;
      if (x > max_x)
        max_x = x;
      if (y < min_y)
        min_y = y;
      if (y > max_y)
        max_y = y;
    }
    
    Matrix3d warp_transfrom = unwarp_transform.inverse();
    Mat unwarped_image = Mat::zeros(max_y - min_y + 1, max_x - min_x + 1, CV_8UC3);
    Mat unwarped_source_mask_image = Mat::zeros(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1);
    Mat unwarped_target_mask_image = Mat::zeros(max_y - min_y + 1, max_x - min_x + 1, CV_8UC1);
    for (int unwarped_y = min_y; unwarped_y <= max_y; unwarped_y++) {
      for (int unwarped_x = min_x; unwarped_x <= max_x; unwarped_x++) {
	Vector3d unwarped_pixel_vec;
        unwarped_pixel_vec << unwarped_x, unwarped_y, 1;
        Vector3d pixel_vec = warp_transfrom * unwarped_pixel_vec;
	if (pixel_vec[2] == 0)
	  continue;
	int x = round(pixel_vec[0] / pixel_vec[2]);
	int y = round(pixel_vec[1] / pixel_vec[2]);
	if (x < 0 || x >= input_image.cols || y < 0 || y >= input_image.rows)
	  continue;
        unwarped_image.at<Vec3b>(unwarped_y - min_y, unwarped_x - min_x) = input_image.at<Vec3b>(y, x);
	if (input_source_mask.at(y * input_image.cols + x) == true)
	  unwarped_source_mask_image.at<uchar>(unwarped_y - min_y, unwarped_x - min_x) = 255;
	if (input_target_mask.at(y * input_image.cols + x) == true)
          unwarped_target_mask_image.at<uchar>(unwarped_y - min_y, unwarped_x - min_x) = 255;
      }
    }
    //imwrite("Test/unwarped_image.bmp", unwarped_image);
    //    exit(1);
    ImageMask unwarped_source_mask(unwarped_source_mask_image);
    ImageMask unwarped_target_mask(unwarped_target_mask_image);
    
    //imwrite("Test/unwarped_source_mask_image.bmp", unwarped_source_mask.drawMaskImage());
    //imwrite("Test/unwarped_target_mask_image.bmp", unwarped_target_mask.drawMaskImage());
    
    
    int num_source_pixels = unwarped_source_mask.getNumPixels();
    int num_target_pixels = unwarped_target_mask.getNumPixels();
    
    if (num_target_pixels == num_source_pixels || num_source_pixels == 0) {
      return input_target_mask.drawImageWithMask(input_image, false, Vec3b(0, 0, 0), true, Vec3b(255, 0, 0));
    }
    
    Mat image_for_completion = unwarped_image.clone();
    cvtColor(image_for_completion, image_for_completion, CV_BGR2Lab);
    for (int pixel = 0; pixel < unwarped_image.cols * unwarped_image.rows; pixel++) {
      Vec3b color = image_for_completion.at<Vec3b>(pixel / unwarped_image.cols, pixel % unwarped_image.cols);
      color[0] = round(color[0] * 1.0 / 3);
      image_for_completion.at<Vec3b>(pixel / unwarped_image.cols, pixel % unwarped_image.cols) = color;
    }

    
    
    vector<Mat> image_pyramid;
    image_pyramid.push_back(image_for_completion);
    vector<ImageMask> source_mask_pyramid;
    source_mask_pyramid.push_back(unwarped_source_mask);
    vector<ImageMask> target_mask_pyramid;
    target_mask_pyramid.push_back(unwarped_target_mask);
    
    while (true) {
      Mat previous_image = image_pyramid.back();
      
      const int NEW_IMAGE_WIDTH = round(previous_image.cols / 2);
      const int NEW_IMAGE_HEIGHT = round(previous_image.rows / 2);
      if (NEW_IMAGE_WIDTH <= 1 || NEW_IMAGE_HEIGHT <= 1)
        break;
      
      Mat new_image;
      //resize(previous_image, new_image, Size(NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT), 0, 0, INTER_NEAREST);
      resize(previous_image, new_image, Size(NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT));
      
      ImageMask new_source_mask = source_mask_pyramid.back();
      new_source_mask.resizeWithBias(NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT, false);
      ImageMask new_target_mask = target_mask_pyramid.back();
      new_target_mask.resize(NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT);
      
      int num_source_pixels = new_source_mask.getNumPixels();
      int num_target_pixels = new_target_mask.getNumPixels();
      
      if (num_source_pixels == 0)
        break;
      
      image_pyramid.push_back(new_image);
      source_mask_pyramid.push_back(new_source_mask);
      target_mask_pyramid.push_back(new_target_mask);
      
      if (num_target_pixels - num_source_pixels < WINDOW_SIZE * WINDOW_SIZE)
	break;
    }
    
    
    Mat target_image = image_pyramid.back().clone();
    for (int level = image_pyramid.size() - 1; level >= 0; level--) {
      //cout << level << endl;
      // if (level != 0)
      //   continue;
      
      Mat source_image = image_pyramid[level];
      const int IMAGE_WIDTH = source_image.cols;
      const int IMAGE_HEIGHT = source_image.rows;
      
      ImageMask source_mask = source_mask_pyramid[level];
      ImageMask target_mask = target_mask_pyramid[level];
      
      if (level == image_pyramid.size() - 1) {
	//cout << source_mask << endl;
	//cout << target_mask << endl;
	// Mat target_image_gray;
	// cvtColor(target_image, target_image_gray, CV_BGR2GRAY);
	// cout << target_image_gray << endl;
	// exit(1);
	
	vector<double> distance_map;
	vector<int> boundary_map;
	source_mask.calcBoundaryDistanceMap(boundary_map, distance_map);
	
	vector<int> source_pixels = source_mask.getPixels();
	vector<int> target_pixels = target_mask.getPixels();
	for (vector<int>::const_iterator target_pixel_it = target_pixels.begin(); target_pixel_it != target_pixels.end(); target_pixel_it++) {
	  if (source_mask.at(*target_pixel_it) == false) {
	    //int source_pixel = source_pixels[rand() % source_pixels.size()];
	    int source_pixel = boundary_map[*target_pixel_it];
	    target_image.at<Vec3b>(*target_pixel_it / IMAGE_WIDTH, *target_pixel_it % IMAGE_WIDTH) = source_image.at<Vec3b>(source_pixel / IMAGE_WIDTH, source_pixel % IMAGE_WIDTH);
	  }
	}
      }
      
      //      vector<int> source_pixels = source_mask.getPixels();
      vector<int> target_pixels = target_mask.getPixels();
      
      // vector<int> nearest_neighbor_field(IMAGE_WIDTH * IMAGE_HEIGHT, -1);
      // vector<double> distance_field(IMAGE_WIDTH * IMAGE_HEIGHT, 1);
      
      for (vector<int>::const_iterator target_pixel_it = target_pixels.begin(); target_pixel_it != target_pixels.end(); target_pixel_it++) {
	if (source_mask.at(*target_pixel_it)) {
          target_image.at<Vec3b>(*target_pixel_it / IMAGE_WIDTH, *target_pixel_it % IMAGE_WIDTH) = source_image.at<Vec3b>(*target_pixel_it / IMAGE_WIDTH, *target_pixel_it % IMAGE_WIDTH);
        }
      }
      
      Mat new_target_image = calcTargetImage(source_image, target_image, source_mask, target_mask, image_pyramid[max(level - 1, 0)], source_mask_pyramid[max(level - 1, 0)], target_mask_pyramid[max(level - 1, 0)], WINDOW_SIZE);
      
      stringstream target_image_filename;
      target_image_filename << "Test/target_image_" << level << ".bmp";
      //imwrite(target_image_filename.str(), new_target_image);
      
      target_image = new_target_image.clone();
    }
    
    //GaussianBlur(target_image, target_image, Size(5, 5), 0, 0);
    //GaussianBlur(target_image, target_image, Size(5, 5), 0, 0);
    
    Mat image_completed = target_image.clone();
    for (int pixel = 0; pixel < unwarped_image.cols * unwarped_image.rows; pixel++) {
      Vec3b color = image_completed.at<Vec3b>(pixel / unwarped_image.cols, pixel % unwarped_image.cols);
      color[0] = min(round(color[0] * 3), 255.0);
      image_completed.at<Vec3b>(pixel / unwarped_image.cols, pixel % unwarped_image.cols) = color;
    }
    cvtColor(image_completed, image_completed, CV_Lab2BGR);
    
    //imwrite("Test/image_completed.bmp", image_completed);
    
    Mat warped_image_completed = input_image.clone();
    for (int pixel = 0; pixel < input_image.cols * input_image.rows; pixel++) {
      if (input_source_mask.at(pixel) == true)
        warped_image_completed.at<Vec3b>(pixel / input_image.cols, pixel % input_image.cols) = input_image.at<Vec3b>(pixel / input_image.cols, pixel % input_image.cols);
      else if (input_target_mask.at(pixel) == false)
        warped_image_completed.at<Vec3b>(pixel / input_image.cols, pixel % input_image.cols) = Vec3b(255, 0, 0);
      else {
	Vector3d pixel_vec;
	pixel_vec << pixel % input_image.cols, pixel / input_image.cols, 1;
	Vector3d unwarped_pixel_vec = unwarp_transform * pixel_vec;
	if (unwarped_pixel_vec[2] == 0)
	  continue;
	int x = round(unwarped_pixel_vec[0] / unwarped_pixel_vec[2]) - min_x;
	int y = round(unwarped_pixel_vec[1] / unwarped_pixel_vec[2]) - min_y;
	if (unwarped_target_mask.at(y * unwarped_image.cols + x))
	  warped_image_completed.at<Vec3b>(pixel / input_image.cols, pixel % input_image.cols) = image_completed.at<Vec3b>(y, x);
        else {
	  vector<int> xs;
	  xs.push_back(floor(unwarped_pixel_vec[0] / unwarped_pixel_vec[2]) - min_x);
	  xs.push_back(ceil(unwarped_pixel_vec[0] / unwarped_pixel_vec[2]) - min_x);
          vector<int> ys;
          ys.push_back(floor(unwarped_pixel_vec[1] / unwarped_pixel_vec[2]) - min_y);
          ys.push_back(ceil(unwarped_pixel_vec[1] / unwarped_pixel_vec[2]) - min_y);
	  for (vector<int>::const_iterator x_it = xs.begin(); x_it != xs.end(); x_it++) {
	    for (vector<int>::const_iterator y_it = ys.begin(); y_it != ys.end(); y_it++) {
	      if (*x_it >= 0 && *x_it < unwarped_image.cols && *y_it >= 0 && *y_it < unwarped_image.rows && unwarped_target_mask.at(*y_it * unwarped_image.cols + *x_it))
		warped_image_completed.at<Vec3b>(pixel / input_image.cols, pixel % input_image.cols) = image_completed.at<Vec3b>(*y_it, *x_it);
	    }
	  }
	}
      }
    }
    //imwrite("Test/warped_image_completed.bmp", warped_image_completed);
    //exit(1);
    //image_completed = warped_image_completed.clone();
    return warped_image_completed;
  }
  
  
  cv::Mat completeImageUsingFusionSpace(const cv::Mat &image, const ImageMask &source_mask, const ImageMask &target_mask, const int WINDOW_SIZE)
  {
    // ImageCompletionCostFunctor cost_functor(image, source_mask, target_mask, WINDOW_SIZE, 100, 1, 200);
    // ImageCompletionProposalGenerator proposal_generator(image, source_mask, target_mask);
    // FusionSpaceSolver solver(image.cols * image.rows, findNeighborsForAllPixels(image.cols, image.rows), cost_functor, proposal_generator, 200);
    
    // vector<double> distance_map;
    // vector<int> boundary_map;
    // source_mask.calcBoundaryDistanceMap(boundary_map, distance_map);
    
    // vector<int> source_pixels = source_mask.getPixels();
    // vector<int> initial_solution(image.cols * image.rows);
    // for (int pixel = 0; pixel < image.cols * image.rows; pixel++)
    //   if (target_mask.at(pixel) == false || source_mask.at(pixel))
    // 	initial_solution[pixel] = pixel;
    //   else
    // 	initial_solution[pixel] = boundary_map[pixel];
    // 	//initial_solution[pixel] = source_pixels[rand() % source_pixels.size()];
    
    // vector<int> current_solution = initial_solution;
    // Mat target_image = image.clone();
    // for (int iteration = 0; iteration < 100; iteration++) {
    //   current_solution = solver.solve(1, current_solution);
      
    //   for (int pixel = 0; pixel < image.cols * image.rows; pixel++)
    // 	target_image.at<Vec3b>(pixel / image.cols, pixel % image.cols) = target_image.at<Vec3b>(current_solution[pixel] / image.cols, current_solution[pixel] % image.cols);
    //   stringstream target_image_filename;
    //   target_image_filename << "Test/target_image_" << iteration << ".bmp";
    //   imwrite(target_image_filename.str(), target_image);
    // }
    // return target_image;
  }
}
