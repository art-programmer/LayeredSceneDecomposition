#include "ConcaveHullFinder.h"

#include <iostream>
#include <cmath>


#include "utils.h"
#include <opencv2/core/core.hpp>

#include "cv_utils/cv_utils.h"


using namespace std;
using namespace cv;


ConcaveHullFinder::ConcaveHullFinder(const int image_width, const int image_height, const vector<double> &point_cloud, const vector<int> &segmentation, const std::map<int, Segment> &segments, const vector<bool> &ROI_mask, const RepresenterPenalties penalties, const DataStatistics statistics, const bool consider_background) : point_cloud_(point_cloud), segmentation_(segmentation), ROI_mask_(ROI_mask), IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(segmentation.size()), NUM_SURFACES_(segments.size()), penalties_(penalties), statistics_(statistics)
{
  for (map<int, Segment>::const_iterator segment_it = segments.begin(); segment_it != segments.end(); segment_it++)
    surface_depths_[segment_it->first] = segment_it->second.getDepthMap();

  for (map<int, Segment>::const_iterator segment_it = segments.begin(); segment_it != segments.end(); segment_it++)
    segment_type_map_[segment_it->first] = segment_it->second.getType();
  
  surface_normals_angles_ = vector<double>(NUM_SURFACES_ * NUM_SURFACES_);
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES_; segment_id_1++) {
    if (segment_type_map_[segment_id_1] != 0)
      continue;
    for (int segment_id_2 = segment_id_1; segment_id_2 < NUM_SURFACES_; segment_id_2++) {
      if (segment_type_map_[segment_id_2] != 0)
        continue;
      vector<double> depth_plane_1 = segments.at(segment_id_1).getDepthPlane();
      vector<double> depth_plane_2 = segments.at(segment_id_2).getDepthPlane();
      double cos_value = 0;
      for (int c = 0; c < 3; c++)
	cos_value += depth_plane_1[c] * depth_plane_2[c];
      double angle = acos(min(abs(cos_value), 1.0));
      surface_normals_angles_[segment_id_1 * NUM_SURFACES_ + segment_id_2] = angle;
      surface_normals_angles_[segment_id_2 * NUM_SURFACES_ + segment_id_1] = angle;
    }
  }

  for (int segment_id = 0; segment_id < NUM_SURFACES_; segment_id++) {
    if (segment_type_map_[segment_id] != 0)
      continue;
    vector<double> depth_plane = segments.at(segment_id).getDepthPlane();
    double cos_value = depth_plane[1];
    double angle = acos(min(abs(cos_value), 1.0));
    if (angle < statistics_.similar_angle_threshold)
      segment_direction_map_[segment_id] = 0;
    else if ((M_PI / 2 - angle) < statistics_.similar_angle_threshold)
      segment_direction_map_[segment_id] = 1;
    else
      segment_direction_map_[segment_id] = -1;
  }
  
  calcConcaveHull();
}

ConcaveHullFinder::~ConcaveHullFinder()
{
}

vector<int> ConcaveHullFinder::getConcaveHull()
{
  return concave_hull_labels_;
}

void ConcaveHullFinder::calcConcaveHull()
{
  vector<double> visible_depths(NUM_PIXELS_);
  vector<vector<int> > surface_occluding_relations(NUM_SURFACES_, vector<int>(NUM_SURFACES_, 0));
  set<int> ROI_segment_ids_set;
  int num_ROI_pixels = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (ROI_mask_[pixel] == false)
      continue;
    int segment_id = segmentation_[pixel];
    assert(segment_id >= 0 && segment_id < NUM_SURFACES_);
    ROI_segment_ids_set.insert(segment_id);
    double depth = surface_depths_[segment_id][pixel];
    visible_depths[pixel] = depth;

    for (int other_segment_id = 0; other_segment_id < NUM_SURFACES_; other_segment_id++) {
      if (other_segment_id == segment_id)
        continue;
      double other_depth = surface_depths_[other_segment_id][pixel];
      if (other_depth > depth || other_depth < 0)
        surface_occluding_relations[segment_id][other_segment_id]++;
      else if (other_depth < depth)
        surface_occluding_relations[segment_id][other_segment_id]--;
    }
    
    num_ROI_pixels++;
  }
  
  vector<int> horizontal_segment_ids_vec;
  vector<int> vertical_segment_ids_vec;
  for (set<int>::const_iterator segment_it = ROI_segment_ids_set.begin(); segment_it != ROI_segment_ids_set.end(); segment_it++) {
    if (segment_type_map_[*segment_it] != 0)
      continue;
    if (segment_direction_map_[*segment_it] == 0)
      horizontal_segment_ids_vec.push_back(*segment_it);
    else if (segment_direction_map_[*segment_it] == 1)
      vertical_segment_ids_vec.push_back(*segment_it);
  }
  
  //cout << horizontal_segment_ids_vec.size() << '\t' << vertical_segment_ids_vec.size() << endl;
  vector<int> best_score_concave_hull;
  double best_score = 0;
  for (int num_horizontal_surfaces = 0; num_horizontal_surfaces <= min(2, static_cast<int>(horizontal_segment_ids_vec.size())); num_horizontal_surfaces++) {
    vector<vector<int> > horizontal_surfaces_vec = cv_utils::findAllCombinations(horizontal_segment_ids_vec, num_horizontal_surfaces);
    for (int num_vertical_surfaces = 0; num_vertical_surfaces <= min(3, static_cast<int>(vertical_segment_ids_vec.size())); num_vertical_surfaces++) {
      if (num_vertical_surfaces == 0 && num_horizontal_surfaces == 0)
	continue;
      vector<vector<int> > vertical_surfaces_vec = cv_utils::findAllCombinations(vertical_segment_ids_vec, num_vertical_surfaces);
      for (vector<vector<int> >::const_iterator horizontal_surfaces_it = horizontal_surfaces_vec.begin(); horizontal_surfaces_it != horizontal_surfaces_vec.end(); horizontal_surfaces_it++) {
	for (vector<vector<int> >::const_iterator vertical_surfaces_it = vertical_surfaces_vec.begin(); vertical_surfaces_it != vertical_surfaces_vec.end(); vertical_surfaces_it++) {
	  vector<int> concave_hull_surfaces = *horizontal_surfaces_it;
	  concave_hull_surfaces.insert(concave_hull_surfaces.end(), vertical_surfaces_it->begin(), vertical_surfaces_it->end());

	  vector<int> concave_hull(NUM_PIXELS_);
	  vector<double> depths(NUM_PIXELS_);
	  int num_invalid_pixels = 0;
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    if (ROI_mask_[pixel] == false)
	      continue;
	    int selected_surface_id = concave_hull_surfaces[0];
	    double selected_depth = surface_depths_[concave_hull_surfaces[0]][pixel];
	    //cout << selected_surface_id << '\t' << selected_depth << endl;
	    for (vector<int>::const_iterator concave_hull_surface_it = concave_hull_surfaces.begin() + 1; concave_hull_surface_it != concave_hull_surfaces.end(); concave_hull_surface_it++) {
	      double depth = surface_depths_[*concave_hull_surface_it][pixel];
	      //cout << *concave_hull_surface_it << '\t' << depth << endl;
	      if (surface_occluding_relations[*concave_hull_surface_it][selected_surface_id] + surface_occluding_relations[selected_surface_id][*concave_hull_surface_it] >= 0) {
		if ((depth < selected_depth || selected_depth < 0) && depth > 0) {
		  selected_surface_id = *concave_hull_surface_it;
		  selected_depth = depth;
		}
	      } else {
		if (depth > selected_depth && selected_depth > 0) {
		  selected_surface_id = *concave_hull_surface_it;
		  selected_depth = depth;
		}
	      }
	    }
	    concave_hull[pixel] = selected_surface_id;
            depths[pixel] = selected_depth;
	  }

	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
            if (ROI_mask_[pixel] == false)
              continue;
	    double depth = depths[pixel];
            if (depth < visible_depths[pixel] - statistics_.depth_conflict_tolerance) {
	      num_invalid_pixels++;
	    }
	  }
	  
	  int num_ROI_pixels = 0;
	  int num_consistent_pixels = 0;
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    if (ROI_mask_[pixel] == false)
	      continue;
	    num_ROI_pixels++;
	    if (segmentation_[pixel] == concave_hull[pixel])
	      num_consistent_pixels++;  
	  }

	  double score = 1.0 * (num_consistent_pixels - num_invalid_pixels * 10) / num_ROI_pixels;

	  if (best_score_concave_hull.size() == 0 || score > best_score) {
	    best_score_concave_hull = concave_hull;
	    best_score = score;
	  }
	}
      }
    }
  }
  //  exit(1);

  concave_hull_labels_ = best_score_concave_hull;
  if (concave_hull_labels_.size() == 0) {
    cout << "Background concave hull not found." << endl;
    return;
  }
  
  concave_hull_surfaces_.clear();
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    if (ROI_mask_[pixel] == true)
      concave_hull_surfaces_.insert(concave_hull_labels_[pixel]);
  
  
  // map<int, Vec3b> color_table;
  // Mat ori_region_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  // Mat concave_hull_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   if (ROI_mask_[pixel] == false) {
  //     ori_region_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(255, 0, 0);
  //     concave_hull_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(255, 0, 0);
  //     continue;
  //   }
  //   int segment_id = segmentation_[pixel];
  //   if (color_table.count(segment_id) == 0) {
  //     int gray_value = rand() % 256;
  //     color_table[segment_id] = Vec3b(gray_value, gray_value, gray_value);
  //   }
  //   ori_region_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_id];
  //   int concave_hull_segment_id = concave_hull_labels_[pixel];
  //   if (color_table.count(concave_hull_segment_id) == 0) {
  //     int gray_value = rand() % 256;
  //     color_table[concave_hull_segment_id] = Vec3b(gray_value, gray_value, gray_value);
  //   }
  //   concave_hull_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[concave_hull_segment_id];
  // }
  
  // static int index = 0;
  // stringstream ori_region_image_filename;
  // ori_region_image_filename << "Test/background_ori_region_image_.bmp";
  // imwrite(ori_region_image_filename.str(), ori_region_image);
  // stringstream concave_hull_image_filename;
  // concave_hull_image_filename << "Test/background_concave_hull_image.bmp";
  // imwrite(concave_hull_image_filename.str(), concave_hull_image);
  // //  index++;
}

