#include "StructureFinder.h"

#include <iostream>
#include <cmath>

// #include "OpenGM/mplp.hxx"
// #include <opengm/inference/trws/trws_trws.hxx>
// #include <opengm/inference/alphaexpansion.hxx>
// #include <opengm/inference/graphcut.hxx>
// #include <opengm/inference/auxiliary/minstcutboost.hxx>
#include "utils.h"
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;


StructureFinder::StructureFinder(const int image_width, const int image_height, const map<int, Segment> &segments, const vector<bool> &candidate_segment_mask, const std::vector<int> visible_segmentation, const vector<double> &visible_depths, const vector<double> &background_depths, const vector<int> &segment_backmost_layer_index_map, const RepresenterPenalties penalties, const DataStatistics statistics) : IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(image_width * image_height), segments_(segments), NUM_SURFACES_(segments.size()), candidate_segment_mask_(candidate_segment_mask), visible_segmentation_(visible_segmentation), visible_depths_(visible_depths), background_depths_(background_depths), segment_backmost_layer_index_map_(segment_backmost_layer_index_map), penalties_(penalties), statistics_(statistics)
{
  findTwoOrthogonalSurfaceStructures();
  //optimizeConcaveHull();
}

vector<pair<double, vector<int> > > StructureFinder::getStructures() const
{
  return structure_score_surface_ids_pairs_;
}

void StructureFinder::findTwoOrthogonalSurfaceStructures()
{
  vector<vector<int> > surface_occluding_relations(NUM_SURFACES_, vector<int>(NUM_SURFACES_, 0));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int segment_id = visible_segmentation_[pixel];
    if (candidate_segment_mask_[segment_id] == false)
      continue;
    double depth = segments_.at(segment_id).getDepth(pixel);
    for (int other_segment_id = 0; other_segment_id < NUM_SURFACES_; other_segment_id++) {
      if (other_segment_id == segment_id)
        continue;
      if (candidate_segment_mask_[other_segment_id] == false)
        continue;
      double other_depth = segments_.at(other_segment_id).getDepth(pixel);
      if (other_depth > depth || other_depth < 0)
        surface_occluding_relations[segment_id][other_segment_id]++;
      else if (other_depth < depth)
        surface_occluding_relations[segment_id][other_segment_id]--;
    }
  }

  vector<double> surface_normals_angles(NUM_SURFACES_ * NUM_SURFACES_);
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES_; segment_id_1++) {
    if (candidate_segment_mask_[segment_id_1] == false)
      continue;
    for (int segment_id_2 = segment_id_1 + 1; segment_id_2 < NUM_SURFACES_; segment_id_2++) {
      if (candidate_segment_mask_[segment_id_2] == false)
        continue;
      vector<double> depth_plane_1 = segments_.at(segment_id_1).getDepthPlane();
      vector<double> depth_plane_2 = segments_.at(segment_id_2).getDepthPlane();
      double cos_value = 0;
      for (int c = 0; c < 3; c++)
        cos_value += depth_plane_1[c] * depth_plane_2[c];
      double angle = acos(min(abs(cos_value), 1.0));
      surface_normals_angles[segment_id_1 * NUM_SURFACES_ + segment_id_2] = angle;
      surface_normals_angles[segment_id_2 * NUM_SURFACES_ + segment_id_1] = angle;
    }
  }

  vector<vector<int> > segment_pixels_vec(NUM_SURFACES_);
  vector<vector<int> > segment_ranges(NUM_SURFACES_);
  for (int segment_id = 0; segment_id < NUM_SURFACES_; segment_id++) {
    if (candidate_segment_mask_[segment_id] == false)
      continue;
    vector<int> segment_pixels = segments_.at(segment_id).getSegmentPixels();
    segment_pixels_vec[segment_id] = segment_pixels;
    
    vector<int> range(4);
    for (int c = 0; c < 2; c++) {
      range[c * 2 + 0] = 1000000;
      range[c * 2 + 1] = -1000000;
    }
    for (vector<int>::const_iterator pixel_it = segment_pixels.begin(); pixel_it != segment_pixels.end(); pixel_it++) {
      int x = *pixel_it % IMAGE_WIDTH_;
      int y = *pixel_it / IMAGE_WIDTH_;
      if (x < range[0])
	range[0] = x;
      if (x > range[1])
        range[1] = x;
      if (y < range[2])
        range[2] = y;
      if (y < range[3])
        range[3] = y;
    }
    segment_ranges[segment_id] = range;
  }
  
  
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES_; segment_id_1++) {
    if (candidate_segment_mask_[segment_id_1] == false)
      continue;
    for (int segment_id_2 = segment_id_1 + 1; segment_id_2 < NUM_SURFACES_; segment_id_2++) {
      if (candidate_segment_mask_[segment_id_2] == false)
        continue;
      if (max(segment_backmost_layer_index_map_[segment_id_1], segment_backmost_layer_index_map_[segment_id_2]) == 0)
	continue;
      
      bool concavity_or_convexity = surface_occluding_relations[segment_id_1][segment_id_2] + surface_occluding_relations[segment_id_2][segment_id_1] >= 0;
      // if (segment_id_1 != 8 || segment_id_2 != 9)
      // 	continue;
      double angle = surface_normals_angles[segment_id_1 * NUM_SURFACES_ + segment_id_2];
      if (M_PI / 2 - angle >= statistics_.similar_angle_threshold)
	continue;

      vector<int> range_1 = segment_ranges[segment_id_1];
      vector<int> range_2 = segment_ranges[segment_id_2];
      if ((range_1[0] > range_2[1] || range_2[0] > range_1[1]) && (range_1[2] > range_2[3] || range_2[2] > range_1[3]))
	continue;

      vector<int> structure_surface_ids(NUM_PIXELS_);
      vector<double> structure_depths(NUM_PIXELS_);
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	double depth_1 = segments_.at(segment_id_1).getDepth(pixel);
	double depth_2 = segments_.at(segment_id_2).getDepth(pixel);
          //cout << *concave_hull_surface_it << '\t' << depth << endl;
	if (concavity_or_convexity) {
	  if (depth_1 < depth_2 && depth_1 > 0) {
            structure_surface_ids[pixel] = segment_id_1;
	    structure_depths[pixel] = depth_1;
	  }
          if (depth_2 < depth_1 && depth_2 > 0) {
            structure_surface_ids[pixel] = segment_id_2;
            structure_depths[pixel] = depth_2;
          }
        } else {
          if (depth_1 > depth_2 && depth_1 > 0) {
            structure_surface_ids[pixel] = segment_id_1;
            structure_depths[pixel] = depth_1;
          }
          if (depth_2 > depth_1 && depth_2 > 0) {
            structure_surface_ids[pixel] = segment_id_2;
            structure_depths[pixel] = depth_2;
          }
	}
      }

      int num_consistent_pixels = 0;
      int num_inpainting_pixels = 0;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	if (structure_surface_ids[pixel] == visible_segmentation_[pixel])
	  num_consistent_pixels++;
	else {
	  double depth = structure_depths[pixel];
	  if (depth >= visible_depths_[pixel] - statistics_.depth_conflict_tolerance && depth <= background_depths_[pixel] + statistics_.depth_conflict_tolerance) {
	    num_inpainting_pixels++;
	  }
	}
      }
      double score = 1.0 * (num_consistent_pixels + num_inpainting_pixels) / NUM_PIXELS_;
      structure_score_surface_ids_pairs_.push_back(make_pair(score, structure_surface_ids));
    }
  }
  sort(structure_score_surface_ids_pairs_.begin(), structure_score_surface_ids_pairs_.end());
  
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs_.size(); pair_index++) {
    double score = structure_score_surface_ids_pairs_[pair_index].first;
    vector<int> surface_ids = structure_score_surface_ids_pairs_[pair_index].second;
    
    Mat structure_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
    map<int, int> color_table;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int segment_id = surface_ids[pixel];
      if (segment_id == -1)
    	continue;
      if (color_table.count(segment_id) == 0)
    	color_table[segment_id] = rand() % 256;
      structure_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_id];
    }
    
    // stringstream structure_image_filename;
    // structure_image_filename << "Test/structure_image_ " << pair_index << ".bmp";
    // imwrite(structure_image_filename.str(), structure_image);

    cout << "structure: ";
    for (map<int, int>::const_iterator segment_it = color_table.begin(); segment_it != color_table.end(); segment_it++)
      cout << segment_it->first << '\t';
    cout << "score:" << score << endl;
  }
}
