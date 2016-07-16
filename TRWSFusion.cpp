//
//  TRWSFusion.cpp
//  SurfaceStereo
//
//  Created by Chen Liu on 11/7/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#include "TRWSFusion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"
#include "cv_utils/cv_utils.h"

using namespace cv;

TRWSFusion::TRWSFusion(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const RepresenterPenalties &penalties, const DataStatistics &statistics, const bool consider_surface_cost) : image_(image), point_cloud_(point_cloud), normals_(normals), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), penalties_(penalties), statistics_(statistics), consider_surface_cost_(consider_surface_cost)
{
  calcColorDiffVar();
}

TRWSFusion::~TRWSFusion()
{
}

double TRWSFusion::calcUnaryCost(const int pixel, const int label)
{
  double input_depth = point_cloud_[pixel * 3 + 2];
  bool inside_ROI = proposal_ROI_mask_[pixel];
  vector<int> layer_labels(proposal_num_layers_);
  int label_temp = label;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels[layer_index] = label_temp % (proposal_num_surfaces_ + 1);
    label_temp /= (proposal_num_surfaces_ + 1);
  }
  
  int foremost_non_empty_layer_index = proposal_num_layers_;
  double foremost_non_empty_layer_depth = 0;
  Segment foremost_non_empty_segment;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    if (layer_labels[layer_index] < proposal_num_surfaces_) {
      foremost_non_empty_layer_index = layer_index;
      foremost_non_empty_layer_depth = proposal_surface_depths_[layer_labels[layer_index]][pixel];
      foremost_non_empty_segment = proposal_segments_[layer_labels[layer_index]];
      
      break;
    }
  }
  
  assert(foremost_non_empty_layer_index < proposal_num_layers_);
  
  int unary_cost = 0;
  //background empty cost
  {
    if (layer_labels[proposal_num_layers_ - 1] == proposal_num_surfaces_)
      unary_cost += penalties_.huge_pen;
  }
  //depth cost
  {
    double depth_diff = foremost_non_empty_segment.calcDistance(point_cloud_, pixel);
    double depth_diff_threshold = foremost_non_empty_layer_index == proposal_num_layers_ - 1 ? statistics_.background_depth_diff_tolerance : 0;
    double depth_diff_cost = input_depth < 0 ? 0 : (1 - exp(-pow(max(depth_diff - depth_diff_threshold, 0.0), 2) / (2 * statistics_.depth_diff_var))) * penalties_.data_depth_pen;
    unary_cost += depth_diff_cost;
    
    
    int depth_conflict_cost = 0;
    double previous_depth = 0;
    for (int layer_index = foremost_non_empty_layer_index; layer_index < proposal_num_layers_; layer_index++) {
      if (layer_labels[layer_index] == proposal_num_surfaces_)
	continue;
      double depth = proposal_surface_depths_[layer_labels[layer_index]][pixel];
      if (depth < previous_depth - statistics_.depth_conflict_tolerance) {
	depth_conflict_cost += penalties_.huge_pen;
      } else
	previous_depth = depth;
    }
    unary_cost += depth_conflict_cost;
    if (depth_diff_cost < 0)
      cout << "depth " << depth_diff_cost << '\t' << depth_diff << '\t' << pixel << '\t' << input_depth << '\t' << foremost_non_empty_layer_depth << '\t' << layer_labels[foremost_non_empty_layer_index] << endl;
  }
  
  //angle cost
  { 
    double angle = foremost_non_empty_segment.calcAngle(normals_, pixel);
    double normal_diff_cost = angle * penalties_.data_normal_pen;
    unary_cost += normal_diff_cost;
    if (normal_diff_cost < 0)
      cout << "normal " << normal_diff_cost << '\t' << angle << '\t' << unary_cost << endl;
  }
  
  //color difference cost
  {
    Vec3f hsv_color = blurred_hsv_image_.at<Vec3f>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    double color_likelihood = foremost_non_empty_segment.predictColorLikelihood(pixel, hsv_color);
    
    double data_color_cost = min(max(-color_likelihood, 0.0), -statistics_.pixel_fitting_color_likelihood_threshold) * penalties_.data_color_pen;
    unary_cost += data_color_cost;
    if (data_color_cost < -100 || data_color_cost > 10000)
      cout << "color " << hsv_color << '\t' << color_likelihood << endl;
  }
  
  //same label cost
  {
    int same_label_cost = 0;
    set<int> used_surface_ids;
    for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
      if (layer_labels[layer_index] == proposal_num_surfaces_)
	continue;
      if (used_surface_ids.count(layer_labels[layer_index]) > 0)
	same_label_cost += penalties_.huge_pen;
      used_surface_ids.insert(layer_labels[layer_index]);
    }
    unary_cost += same_label_cost;
  }
  
  //non-plane segment cost
  {
    int segment_type = foremost_non_empty_segment.getSegmentType();
    int segment_type_cost_scale = segment_type == -1 ? 2 : (segment_type == 0 ? 0 : 1);
    int non_plane_segment_cost = segment_type_cost_scale * penalties_.data_non_plane_pen;
    unary_cost += non_plane_segment_cost;
  }
    
  return unary_cost;
}

double TRWSFusion::calcPairwiseCost(const int pixel_1, const int pixel_2, const int label_1, const int label_2)
{
  if (label_1 == label_2)
    return 0;
  vector<int> layer_labels_1(proposal_num_layers_);
  int label_temp_1 = label_1;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_1[layer_index] = label_temp_1 % (proposal_num_surfaces_ + 1);
    label_temp_1 /= (proposal_num_surfaces_ + 1);
  }
  vector<int> layer_labels_2(proposal_num_layers_);
  int label_temp_2 = label_2;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_2[layer_index] = label_temp_2 % (proposal_num_surfaces_ + 1);
    label_temp_2 /= (proposal_num_surfaces_ + 1);
  }
  
  double pairwise_cost = 0;
  bool surface_1_visible = true;
  bool surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 == surface_id_2) {
      if (surface_id_1 < proposal_num_surfaces_) {
	surface_1_visible = false;
	surface_2_visible = false;
	continue;
      }
    }
    if (surface_id_1 < proposal_num_surfaces_ && surface_id_2 < proposal_num_surfaces_) {
      double depth_1_1 = proposal_surface_depths_[surface_id_1][pixel_1];
      double depth_1_2 = proposal_surface_depths_[surface_id_1][pixel_2];
      double depth_2_1 = proposal_surface_depths_[surface_id_2][pixel_1];
      double depth_2_2 = proposal_surface_depths_[surface_id_2][pixel_2];
      
      if (depth_1_1 <= 0 || depth_1_2 <= 0 || depth_2_1 <= 0 || depth_2_2 <= 0)
	return penalties_.huge_pen / 100;
      
      double diff_1 = abs(depth_1_1 - depth_2_1);
      double diff_2 = abs(depth_1_2 - depth_2_2);
      double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
      double min_diff = min(min(diff_1, diff_2), diff_middle);
      
      pairwise_cost += min(min_diff / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
      
      surface_1_visible = false;
      surface_2_visible = false;
      
      
    } else if (surface_id_1 < proposal_num_surfaces_ || surface_id_2 < proposal_num_surfaces_) {
      if (surface_id_1 < proposal_num_surfaces_ && surface_1_visible) {
	surface_1_visible = false;
      }
      if (surface_id_2 < proposal_num_surfaces_ && surface_2_visible) {
	surface_2_visible = false;
      }
      pairwise_cost += penalties_.smoothness_empty_non_empty_ratio * penalties_.smoothness_pen;
    }
  }
  
  surface_1_visible = true;
  surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_surfaces_) {
      if (surface_1_visible == true) {
        if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_1].calcDistanceOffset(pixel_1, pixel_2) == 1)
	  pairwise_cost += penalties_.smoothness_concave_shape_pen;
	surface_1_visible = false;
      }
    }
    if (surface_id_2 < proposal_num_surfaces_) {
      if (surface_1_visible == true) {
	if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_2].calcDistanceOffset(pixel_2, pixel_1) == 1)
	  pairwise_cost += penalties_.smoothness_concave_shape_pen;
	surface_2_visible = false;
      }
    }
  }
  
  
  int visible_surface_1 = -1;
  int visible_surface_2 = -1;
  int visible_layer_index_1 = -1;
  int visible_layer_index_2 = -1;
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_surfaces_) {
      if (visible_surface_1 == -1) {
        visible_surface_1 = surface_id_1;
        visible_layer_index_1 = layer_index;
      }
    }
    if (surface_id_2 < proposal_num_surfaces_) {
      if (visible_surface_2 == -1) {
        visible_surface_2 = surface_id_2;
        visible_layer_index_2 = layer_index;
      }
    }
  }
  
  if (visible_surface_1 != visible_surface_2) {
    pairwise_cost += exp(-pow(calcColorDiff(pixel_1, pixel_2), 2) / (2 * color_diff_var_)) * penalties_.smoothness_anisotropic_diffusion_pen;
  }
  
  double distance_2D = sqrt(pow(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_, 2) + pow(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_, 2));
  return pairwise_cost / distance_2D;
}


vector<int> TRWSFusion::fuse(const vector<vector<int> > &proposal_labels, const int proposal_num_surfaces, const int proposal_num_layers, const map<int, Segment> &proposal_segments, const vector<int> &previous_solution_indices, const vector<bool> &proposal_ROI_mask)
{
  cout << "fuse" << endl;
  
  proposal_num_surfaces_ = proposal_num_surfaces;
  proposal_num_layers_ = proposal_num_layers;
  proposal_segments_ = proposal_segments;
  
  proposal_surface_depths_.clear();
  for (map<int, Segment>::const_iterator segment_it = proposal_segments.begin(); segment_it != proposal_segments.end(); segment_it++)
    proposal_surface_depths_[segment_it->first] = segment_it->second.getDepthMap();
  if (proposal_ROI_mask.size() == NUM_PIXELS_)
    proposal_ROI_mask_ = proposal_ROI_mask;
  else
    proposal_ROI_mask_ = vector<bool>(NUM_PIXELS_, true);
  
  const int NUM_NODES = consider_surface_cost_ ? NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ : NUM_PIXELS_;
  
  unique_ptr<MRFEnergy<TypeGeneral> > energy(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
  vector<MRFEnergy<TypeGeneral>::NodeId> nodes(NUM_NODES);
  
  
  int pixel_index_offset = 0;
  int indicator_index_offset = 0;
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    const int NUM_PROPOSALS = pixel_proposal.size();
    if (NUM_PROPOSALS == 0) {
      cout << "empty proposal error: " << pixel << endl;
      exit(1);
    }
    vector<double> cost(NUM_PROPOSALS);
    for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
      cost[proposal_index] = calcUnaryCost(pixel, pixel_proposal[proposal_index]);
    nodes[pixel + pixel_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&cost[0]));
  }
  
  if (consider_surface_cost_ == true) {
    for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i++) {
      vector<int> layer_surface_indicator_proposal = proposal_labels[i];
      const int NUM_PROPOSALS = layer_surface_indicator_proposal.size();
      vector<double> surface_cost(NUM_PROPOSALS);
      for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
        surface_cost[proposal_index] = layer_surface_indicator_proposal[proposal_index] == 1 ? penalties_.surface_pen : 0;
      nodes[i + indicator_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&surface_cost[0]));
    }
  }
  
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      vector<int> neighbor_pixel_proposal = proposal_labels[neighbor_pixel];
      vector<double> cost(pixel_proposal.size() * neighbor_pixel_proposal.size(), 0);
      for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++)
	for (int proposal_index_2 = 0; proposal_index_2 < neighbor_pixel_proposal.size(); proposal_index_2++)
          cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = calcPairwiseCost(pixel, neighbor_pixel, pixel_proposal[proposal_index_1], neighbor_pixel_proposal[proposal_index_2]);
      bool has_non_zero_cost = false;
      for (int i = 0; i < cost.size(); i++)
	if (cost[i] > 0)
	  has_non_zero_cost = true;
      if (has_non_zero_cost == true)
	energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[neighbor_pixel + pixel_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
    }
  }
  
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {
    map<int, map<int, vector<double> > > pairwise_costs;
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(proposal_labels);
    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
      vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_1(NUM_PIXELS_);
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        vector<int> pixel_proposal = proposal_labels[pixel];
        for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
          int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index_1)) % (proposal_num_surfaces_ + 1);
          if (surface_id < proposal_num_surfaces_)
            pixel_surface_proposals_map_vec_1[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
        }
      }
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
        vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_2(NUM_PIXELS_);
	if (layer_index_2 == layer_index_1)
	  pixel_surface_proposals_map_vec_2 = pixel_surface_proposals_map_vec_1;
	else {
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    vector<int> pixel_proposal = proposal_labels[pixel];
	    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
	      int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index_2)) % (proposal_num_surfaces_ + 1);
	      if (surface_id < proposal_num_surfaces_)
		pixel_surface_proposals_map_vec_2[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
	    }
	  }
	}
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
	for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
	  set<int> surface_pixel_pairs_1 = *pixel_it;
	  set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
	  for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
	    for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;
              
	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
		continue;
	      double cost = 0;
	      if (layer_index_2 == layer_index_1) {
		if (surface_id_2 >= surface_id_1)
		  continue;
		if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
		  continue;
		double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
		cost = min(depth_diff / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_smoothness_pen + penalties_.smoothness_small_constant_pen;
	      } else {
		if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_tolerance)
		  cost = penalties_.other_viewpoint_depth_conflict_pen;
	      }
	      if (cost < 0.000001)
		continue;
	      
	      if (pixel_1 < pixel_2) {
                if (pairwise_costs.count(pixel_1) == 0 || pairwise_costs[pixel_1].count(pixel_2) == 0)
		  pairwise_costs[pixel_1][pixel_2] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      } else {
		if (pairwise_costs.count(pixel_2) == 0 || pairwise_costs[pixel_2].count(pixel_1) == 0)
                  pairwise_costs[pixel_2][pixel_1] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      }
              vector<int> surface_proposals_1 = pixel_surface_proposals_map_vec_1[pixel_1][surface_id_1];
	      vector<int> surface_proposals_2 = pixel_surface_proposals_map_vec_2[pixel_2][surface_id_2];
	      for (vector<int>::const_iterator proposal_it_1 = surface_proposals_1.begin(); proposal_it_1 != surface_proposals_1.end(); proposal_it_1++)
		for (vector<int>::const_iterator proposal_it_2 = surface_proposals_2.begin(); proposal_it_2 != surface_proposals_2.end(); proposal_it_2++)
		  if (pixel_1 < pixel_2)
		    pairwise_costs[pixel_1][pixel_2][*proposal_it_1 + *proposal_it_2 * proposal_labels[pixel_1].size()] += cost;
                  else
		    pairwise_costs[pixel_2][pixel_1][*proposal_it_2 + *proposal_it_1 * proposal_labels[pixel_2].size()] += cost;
	    }
	  }
	}
      }
    }
    
    for (map<int, map<int, vector<double> > >::iterator pixel_it_1 = pairwise_costs.begin(); pixel_it_1 != pairwise_costs.end(); pixel_it_1++)
      for (map<int, vector<double> >::iterator pixel_it_2 = pixel_it_1->second.begin(); pixel_it_2 != pixel_it_1->second.end(); pixel_it_2++)
	energy->AddEdge(nodes[pixel_it_1->first + pixel_index_offset], nodes[pixel_it_2->first + pixel_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pixel_it_2->second[0]));
  }
  
  
  if (consider_surface_cost_ == true) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
          int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_surfaces_ + surface_id;
	  
          vector<int> layer_surface_indicator_proposal = proposal_labels[layer_surface_indicator_index];
	  vector<double> cost(pixel_proposal.size() * layer_surface_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
          for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
            for (int proposal_index_2 = 0; proposal_index_2 < layer_surface_indicator_proposal.size(); proposal_index_2++) {
	      int label = pixel_proposal[proposal_index_1];
	      int label_surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
              double layer_surface_indicator_conflict_cost = (label_surface_id == surface_id && layer_surface_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_surface_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = layer_surface_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
	  
	  if (has_non_zero_cost == true)
            energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[layer_surface_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
        }
      }
    }
  }
  
  
  const int NUM_INDICATORS = proposal_num_layers_ * proposal_num_surfaces_;
  vector<int> fixed_indicator_mask(NUM_INDICATORS, -1);
  int num_fixed_indicators = 0;
  map<int, set<int> > surface_layers;
  if (consider_surface_cost_) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++) {
	int label = pixel_proposal[proposal_index];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  if (surface_id < proposal_num_surfaces_) {
	    surface_layers[surface_id].insert(layer_index);
	  }
	}
      }
    }
  
    for (map<int, set<int> >::const_iterator surface_it = surface_layers.begin(); surface_it != surface_layers.end(); surface_it++) {
      set<int> layers = surface_it->second;
      if (layers.size() == proposal_num_surfaces_)
	continue;
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	if (layers.count(layer_index) > 0)
	  continue;
	int indicator_index = layer_index * proposal_num_surfaces_ + surface_it->first;
	vector<double> fixed_indicator_cost_diff(2, 0);
	fixed_indicator_cost_diff[1] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	fixed_indicator_mask[indicator_index] = 0;
	num_fixed_indicators++;
      }
    }
  }

  
  static double previous_energy = -1;
  bool check_previous_energy = true;
  if (check_previous_energy) {
    vector<int> previous_solution_labels(NUM_PIXELS_);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      previous_solution_labels[pixel] = proposal_labels[pixel][previous_solution_indices[pixel]];
    vector<int> indicators(proposal_num_surfaces * proposal_num_layers_, 0);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int label = previous_solution_labels[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = label / static_cast<int>(pow(proposal_num_surfaces + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces + 1);
	if (surface_id < proposal_num_surfaces) {
	  indicators[proposal_num_surfaces * layer_index + surface_id] = 1;
	}
      }
    }
    previous_solution_labels.insert(previous_solution_labels.end(), indicators.begin(), indicators.end());
    double previous_solution_energy = checkSolutionEnergy(previous_solution_labels);
    assert(previous_energy < 0 || abs(previous_solution_energy - previous_energy) < 1);

    bool test_possible_solution = false;
    if (test_possible_solution) {
      vector<int> possible_solution = previous_solution_labels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int ori_label = previous_solution_labels[pixel];
	int new_label = 0;
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int surface_id = ori_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  if (surface_id != 11)
	    new_label += surface_id * pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index);
          else
	    new_label += 1 * pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index);
        }
	possible_solution[pixel] = new_label;
      }
      energy_ = checkSolutionEnergy(possible_solution);
      return possible_solution;
    }
  }
  
  MRFEnergy<TypeGeneral>::Options options;
  options.m_iterMax = 2000;
  options.m_printIter = 200;
  options.m_printMinIter = 100;
  options.m_eps = 0.1;

  //energy->SetAutomaticOrdering();
  //energy->ZeroMessages();
  //energy->AddRandomMessages(0, 0, 0.001);
  
  energy->Minimize_TRW_S(options, lower_bound_, energy_);
  solution_.assign(NUM_NODES, 0);

  vector<int> fused_labels(NUM_NODES);
  vector<double> confidences(NUM_NODES);
  for (int i = 0; i < NUM_NODES; i++) {
    
    int label = i < NUM_PIXELS_ ? energy->GetSolution(nodes[i + pixel_index_offset]) : energy->GetSolution(nodes[i + indicator_index_offset]);
    solution_[i] = label;
    fused_labels[i] = proposal_labels[i][label];
  }
  
  
  checkSolutionEnergy(fused_labels);
  
  const double OPTIMAL_THRESHOLD_SCALE = 1.1;
  const double LOWER_BOUND_DIFF_THRESHOLD = 0.01;
  
  if (energy_ <= lower_bound_ * OPTIMAL_THRESHOLD_SCALE) {
    //delete energy;
    if (energy_ < previous_energy)
      previous_energy = energy_;
    return fused_labels;
  } else {
    energy_ = 100000000;
    lower_bound_ = 100000000;
    return fused_labels;
  }

  bool optimal_solution_found = false;

  
  int NUM_INCONFIDENT_INDICATORS = NUM_INDICATORS * 0;
  vector<pair<double, int> > confidence_index_pairs;
  for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + NUM_INDICATORS; i++)
    confidence_index_pairs.push_back(make_pair(confidences[i], i - NUM_PIXELS_));
  sort(confidence_index_pairs.begin(), confidence_index_pairs.end());
    
  bool new_indicator_fixed = false;
  for (int i = NUM_INCONFIDENT_INDICATORS; i < NUM_INDICATORS; i++) {
    if (abs(confidence_index_pairs[i].first - penalties_.surface_pen) < 0.0001) {
      new_indicator_fixed = true;
	
      int indicator_index = confidence_index_pairs[i].second;
      vector<double> fixed_indicator_cost_diff(2, 0);
      if (fused_labels[NUM_PIXELS_ + indicator_index] == 0)
	fixed_indicator_cost_diff[1] = 1000000;
      else
	fixed_indicator_cost_diff[0] = 1000000;
      energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
      fixed_indicator_mask[indicator_index] = fused_labels[NUM_PIXELS_ + indicator_index];
      num_fixed_indicators++;
    }
  }
  if (new_indicator_fixed == true) {
    for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
      int not_fixed_layer_index = -1;
      bool has_non_empty_layer = false;
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == -1) {
	  if (not_fixed_layer_index == -1)
	    not_fixed_layer_index = layer_index;
	  else {
	    not_fixed_layer_index = -1;
	    break;
	  }
	} else if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == 1) {
	  has_non_empty_layer = true;
	  break;
	}
      }
      if (not_fixed_layer_index != -1 && has_non_empty_layer == false) {
	int indicator_index = not_fixed_layer_index * proposal_num_surfaces_ + surface_id;
	vector<double> fixed_indicator_cost_diff(2, 0);
	fixed_indicator_cost_diff[0] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	  
	fixed_indicator_mask[indicator_index] = 1;
	num_fixed_indicators++;
      }
    }
      
      
    double lower_bound;
    //energy->ZeroMessages();
    energy->Minimize_TRW_S(options, lower_bound, energy_);
    if (energy_ <= lower_bound * OPTIMAL_THRESHOLD_SCALE)
      optimal_solution_found = true;
  }
    

  while (num_fixed_indicators < NUM_INDICATORS && optimal_solution_found == false) {
    double lowest_energy = -1;
    int lowest_energy_indicator_index = -1;
    int lowest_energy_indicator_value = -1;
    for (int indicator_index = 0; indicator_index < NUM_INDICATORS; indicator_index++) {
      if (fixed_indicator_mask[indicator_index] != -1)
	continue;
      vector<double> cost_diff(2, 0);
      cost_diff[1] = 1000000;
      energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
      double lower_bound_0, energy_0;
      //energy->ZeroMessages();
      cout << "try to fix indicator " << indicator_index << " as 0" << endl;
      energy->Minimize_TRW_S(options, lower_bound_0, energy_0);
      
      if (lowest_energy < 0 || lower_bound_0 < lowest_energy) {
	lowest_energy = lower_bound_0;
	lowest_energy_indicator_index = indicator_index;
	lowest_energy_indicator_value = 0;
      }
      
      cost_diff[0] = 1000000;
      cost_diff[1] = -1000000;
      energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
      double lower_bound_1, energy_1;
      //energy->ZeroMessages();
      cout << "try to fix indicator " << indicator_index << " as 1" << endl;
      energy->Minimize_TRW_S(options, lower_bound_1, energy_1);
      
      if (lowest_energy < 0 || lower_bound_1 < lowest_energy) {
	lowest_energy = lower_bound_1;
	lowest_energy_indicator_index = indicator_index;
	lowest_energy_indicator_value = 1;
      }
      
      cost_diff[0] = -1000000;
      cost_diff[1] = 0;
      energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
      
      if (energy_0 <= lower_bound_0 * OPTIMAL_THRESHOLD_SCALE && lower_bound_0 <= lower_bound_1) {
	cost_diff[0] = 0;
	cost_diff[1] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	//energy->ZeroMessages();
	energy->Minimize_TRW_S(options, lower_bound_0, energy_0);
	energy_ = energy_0;
	optimal_solution_found = true;
	break;
      }
      if (energy_1 <= lower_bound_1 * OPTIMAL_THRESHOLD_SCALE && lower_bound_1 <= lower_bound_0) {
	cost_diff[0] = 1000000;
	cost_diff[1] = 0;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	//energy->ZeroMessages();
	energy->Minimize_TRW_S(options, lower_bound_1, energy_1);
	energy_ = energy_1;
	optimal_solution_found = true;
	break;
      }
      // if (abs(lower_bound_0 - lower_bound_1) < min(lower_bound_0, lower_bound_1) * LOWER_BOUND_DIFF_THRESHOLD) {
      //   if (lower_bound_0 < lower_bound_1) {
      //     cout << "fix indicator " << indicator_index << " as 0" << endl;
      //     cost_diff[0] = 0;
      //     cost_diff[1] = 1000000;
      //     energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
      //   } else {
      //     cout << "fix indicator " << indicator_index << " as 1" << endl;
      //     cost_diff[0] = 1000000;
      //     cost_diff[1] = 0;
      //     energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
      //   }	  
      // }
    }
    if (optimal_solution_found == true)
      break;
    
    vector<double> fixed_indicator_cost_diff(2, 0);
    if (lowest_energy_indicator_value == 0)
      fixed_indicator_cost_diff[1] = 1000000;
    else
      fixed_indicator_cost_diff[0] = 1000000;
    energy->AddNodeData(nodes[NUM_PIXELS_ + lowest_energy_indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
    fixed_indicator_mask[lowest_energy_indicator_index] = lowest_energy_indicator_value;
    num_fixed_indicators++;
    cout << "fix indicator " << lowest_energy_indicator_index << " as " << lowest_energy_indicator_value << endl;
    
    for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
      int not_fixed_layer_index = -1;
      bool has_non_empty_layer = false;
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == -1) {
	  if (not_fixed_layer_index == -1)
	    not_fixed_layer_index = layer_index;
	  else {
	    not_fixed_layer_index = -1;
	    break;
	  }
	} else if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == 1) {
	  has_non_empty_layer = true;
	  break;
	}
      }
      if (not_fixed_layer_index != -1 && has_non_empty_layer == false) {
	int indicator_index = not_fixed_layer_index * proposal_num_surfaces_ + surface_id;
	vector<double> fixed_indicator_cost_diff(2, 0);
	fixed_indicator_cost_diff[0] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	
	fixed_indicator_mask[indicator_index] = 1;
	num_fixed_indicators++;
      }
    }
  }
  
  fused_labels.assign(NUM_NODES, 0);
  for (int node_index = 0; node_index < NUM_NODES; node_index++) {
    int label = node_index < NUM_PIXELS_ ? energy->GetSolution(nodes[node_index + pixel_index_offset]) : energy->GetSolution(nodes[node_index + indicator_index_offset]);
    fused_labels[node_index] = proposal_labels[node_index][label];
  }
  
  
  energy_ = checkSolutionEnergy(fused_labels);
  
  if (energy_ < previous_energy)
    previous_energy = energy_;
  return fused_labels;
}



vector<double> TRWSFusion::getEnergyInfo()
{
  vector<double> energy_info(2);
  energy_info[0] = energy_;
  energy_info[1] = lower_bound_;
  return energy_info;
}

double TRWSFusion::checkSolutionEnergy(const vector<int> &solution_for_check)
{
  vector<int> solution = solution_for_check;
  
  if (consider_surface_cost_) {
    vector<int> correct_indicators(proposal_num_surfaces_ * proposal_num_layers_, 0);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int label = solution[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	if (surface_id < proposal_num_surfaces_) {
	  correct_indicators[proposal_num_surfaces_ * layer_index + surface_id] = 1;
	}
      }
    }
    bool has_indicator_conflict = false;
    for (int indicator_index = 0; indicator_index < proposal_num_surfaces_ * proposal_num_layers_; indicator_index++) {
      if (solution[indicator_index + NUM_PIXELS_] != correct_indicators[indicator_index]) {
	has_indicator_conflict = true;
	//cout << "correct indicator: " << indicator_index << '\t' << proposal_num_surfaces_ << '\t' << solution[indicator_index + NUM_PIXELS_] << endl;
	solution[indicator_index + NUM_PIXELS_] = correct_indicators[indicator_index];
      }
    }
  }
  
  
  double unary_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    unary_cost += calcUnaryCost(pixel, solution[pixel]);
  
  double pairwise_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      pairwise_cost += calcPairwiseCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]);
      
    }
  }
  
  double other_viewpoint_depth_change_cost = 0;
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {
    
    vector<vector<int> > solution_labels(solution.size());
    for (int i = 0; i < solution.size(); i++)
      solution_labels[i].push_back(solution[i]);
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(solution_labels);
    
    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
        for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
          set<int> surface_pixel_pairs_1 = *pixel_it;
          set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
          for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
            for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;
	      
	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
                continue;
              double cost = 0;
              if (layer_index_2 == layer_index_1) {
                if (surface_id_1 >= surface_id_2)
                  continue;
                if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
                  continue;
                double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
                cost = min(depth_diff / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_smoothness_pen + penalties_.smoothness_small_constant_pen;
              } else {
                if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_tolerance) {
                  cost = penalties_.other_viewpoint_depth_conflict_pen;
		  cout << "other viewpoint cost: " << pixel_1 << '\t' << pixel_2 << '\t' << proposal_segments_.at(surface_id_1).getDepth(pixel_1) << '\t' << proposal_segments_.at(surface_id_2).getDepth(pixel_2) << endl;
		}
              }
	      other_viewpoint_depth_change_cost += cost;
	    }
	  }
        }
      }
    }
  }
  
  double surface_cost = 0;
  double layer_cost = 0;
  if (consider_surface_cost_) {
    for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i++) {
      int layer_surface_indicator = solution[i];
      surface_cost += layer_surface_indicator == 1 ? penalties_.surface_pen : 0;
    }
    
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int pixel_label = solution[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_surfaces_ + surface_id;
	  
	  int layer_surface_indicator = solution[layer_surface_indicator_index];
	  int label_surface_id = pixel_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  surface_cost += (label_surface_id == surface_id && layer_surface_indicator == 0) ? penalties_.huge_pen : 0;
	}
      }
    }
  }
  
  double total_cost = unary_cost + pairwise_cost + other_viewpoint_depth_change_cost + surface_cost;
  cout << "cost: " << total_cost << " = " << unary_cost << " + " << pairwise_cost << " + " << other_viewpoint_depth_change_cost << " + " << surface_cost << endl;
  return total_cost;
}



void TRWSFusion::calcColorDiffVar()
{
  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  blurred_image.convertTo(blurred_hsv_image_, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image_, blurred_hsv_image_, CV_BGR2HSV);
  
  double color_diff_sum2 = 0;
  double depth_diff_sum2 = 0;
  int num_pairs = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double depth = point_cloud_[pixel * 3 + 2];
    if (depth < 0)
      continue;
    vector<int> neighbor_pixels;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;  
      double neighbor_depth = point_cloud_[neighbor_pixel * 3 + 2];
      if (neighbor_depth < 0)
	continue;
      color_diff_sum2 += pow(calcColorDiff(pixel, neighbor_pixel), 2);
      depth_diff_sum2 += pow(neighbor_depth - depth, 2);
      num_pairs++;
    }
  }
  color_diff_var_ = color_diff_sum2 / num_pairs;
  cout << "color diff var: " << color_diff_var_ << endl;
  cout << "depth diff var: " << depth_diff_sum2 / num_pairs << endl;
}

double TRWSFusion::calcColorDiff(const int pixel_1, const int pixel_2)
{
  Vec3f color_1 = blurred_hsv_image_.at<Vec3f>(pixel_1 / IMAGE_WIDTH_, pixel_1 % IMAGE_WIDTH_);
  Vec3f color_2 = blurred_hsv_image_.at<Vec3f>(pixel_2 / IMAGE_WIDTH_, pixel_2 % IMAGE_WIDTH_);
  
  double color_diff = sqrt(pow(color_1[1] * cos(color_1[0] * M_PI / 180) - color_2[1] * cos(color_2[0] * M_PI / 180), 2) + pow(color_1[1] * sin(color_1[0] / 180 * M_PI) - color_2[1] * sin(color_2[0] / 180 * M_PI), 2));
  
  return color_diff;
}

vector<vector<set<int> > > TRWSFusion::calcOverlapPixels(const vector<vector<int> > &proposal_labels)
{
  vector<vector<set<int> > > layer_pixel_surface_pixel_pairs(proposal_num_layers_, vector<set<int> >(NUM_PIXELS_ * 4));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
        if (surface_id == proposal_num_surfaces_)
	  continue;
	
        vector<int> projected_pixels = proposal_segments_.at(surface_id).projectToOtherViewpoints(pixel, statistics_.viewpoint_movement);
        for (vector<int>::const_iterator projected_pixel_it = projected_pixels.begin(); projected_pixel_it != projected_pixels.end(); projected_pixel_it++) {
	  layer_pixel_surface_pixel_pairs[layer_index][*projected_pixel_it].insert(surface_id * NUM_PIXELS_ + pixel);
        }
      }
    }
  }
  
  return layer_pixel_surface_pixel_pairs;
}
