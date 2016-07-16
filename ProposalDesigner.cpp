#include "ProposalDesigner.h"

#include <iostream>

#include "utils.h"
#include "ConcaveHullFinder.h"
#include "StructureFinder.h"

#include "cv_utils/cv_utils.h"


using namespace std;
using namespace cv;


ProposalDesigner::ProposalDesigner(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const vector<double> &camera_parameters, const int num_layers, const RepresenterPenalties penalties, const DataStatistics statistics) : image_(image), point_cloud_(point_cloud), normals_(normals), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), statistics_(statistics), NUM_PIXELS_(image.cols * image.rows), NUM_LAYERS_(num_layers), NUM_ALL_PROPOSAL_ITERATIONS_(3)
{
  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  blurred_image.convertTo(blurred_hsv_image_, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image_, blurred_hsv_image_, CV_BGR2HSV);
  
  initializeCurrentSolution();
  
  proposal_type_indices_ = vector<int>(7);
  for (int c = 0; c < 7; c++)
    proposal_type_indices_[c] = c;
  proposal_type_index_ptr_ = -1;
  all_proposal_iteration_ = 0;
}

ProposalDesigner::~ProposalDesigner()
{
}

void ProposalDesigner::setCurrentSolution(const vector<int> &current_solution_labels, const int current_solution_num_surfaces, const std::map<int, Segment> &current_solution_segments)
{
  map<int, int> surface_id_map;
  int new_surface_id = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id == current_solution_num_surfaces)
        continue;
      if (surface_id_map.count(surface_id) == 0) {
        surface_id_map[surface_id] = new_surface_id;
        new_surface_id++;
      }
    }
    if (surface_id_map.size() == current_solution_num_surfaces)
      break;
  }
  surface_id_map[current_solution_num_surfaces] = new_surface_id;


  current_solution_segments_.clear();
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments.begin(); segment_it != current_solution_segments.end(); segment_it++) {
    if (surface_id_map.count(segment_it->first) > 0) {
      current_solution_segments_[surface_id_map[segment_it->first]] = segment_it->second;
    }
  }

  
  vector<int> new_current_solution_labels(NUM_PIXELS_);
  int new_current_solution_num_surfaces = new_surface_id;
  
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    if (checkLabelValidity(pixel, current_solution_label, current_solution_num_surfaces, current_solution_segments) == false) {
      cout << "invalid current label at pixel: " << pixel << endl;
      exit(1);
    }
    int new_label = 0;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      new_label += surface_id_map[surface_id] * pow(new_current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index);
    }
    new_current_solution_labels[pixel] = new_label;
  }

  current_solution_labels_ = new_current_solution_labels;
  current_solution_num_surfaces_ = new_current_solution_num_surfaces;
  
}

bool ProposalDesigner::getProposal(int &iteration, vector<vector<int> > &proposal_labels, int &proposal_num_surfaces, map<int, Segment> &proposal_segments, string &proposal_type)
{
  if (proposal_type_index_ptr_ < 0 || proposal_type_index_ptr_ >= proposal_type_indices_.size()) {
    random_shuffle(proposal_type_indices_.begin(), proposal_type_indices_.end());
    proposal_type_index_ptr_ = 0;
    all_proposal_iteration_++;
    if (all_proposal_iteration_ > NUM_ALL_PROPOSAL_ITERATIONS_)
      return false;
  }
  
  const int NUM_PROPOSAL_TYPES = 7;
  bool first_attempt = true;
  if (iteration == 0) {
    bool generate_success = generateSegmentAddingProposal(0);
    assert(generate_success);
  } else if (iteration == 1) {
    bool generate_success = generateConcaveHullProposal(true);
    assert(generate_success);
  } else if (iteration == 2) {
    bool generate_success = generateSegmentRefittingProposal();
    assert(generate_success);
  } else {
    while (true) {
      bool generate_success = false;
      if (single_surface_candidate_pixels_.size() > 0) {
	generate_success = generateSingleSurfaceExpansionProposal();
      } else {
	int proposal_type_index = proposal_type_indices_[proposal_type_index_ptr_];
	proposal_type_index_ptr_++;
	
	switch (proposal_type_index) {
	case 0:
	  generate_success = generateSegmentRefittingProposal();
	  break;
	case 1:
	  generate_success = generateConcaveHullProposal(true);
	  break;
	case 2:
	  generate_success = generateSingleSurfaceExpansionProposal();
	  break;
	case 3:
	  generate_success = generateLayerSwapProposal();
	  break;
	case 4:
	  generate_success = generateSegmentAddingProposal();
	  break;
	case 5:
	  generate_success = generateBackwardMergingProposal();
	  break;
	case 6:
	  generate_success = generateStructureExpansionProposal();
	  break;
	default:
	  return false;
	}
      }      
      if (generate_success == true)
	break;
      first_attempt = false;
    }
  }
  
  proposal_labels = proposal_labels_;
  proposal_num_surfaces = proposal_num_surfaces_;
  proposal_segments = proposal_segments_;
  proposal_type = proposal_type_;
  return true;
}


void ProposalDesigner::addIndicatorVariables(const int num_indicator_variables)
{
  int num = num_indicator_variables == -1 ? NUM_LAYERS_ * proposal_num_surfaces_ : num_indicator_variables;
  vector<int> indicator_labels(2);
  indicator_labels[0] = 0;
  indicator_labels[1] = 1;
  for (int i = 0; i < num; i++)
    proposal_labels_.push_back(indicator_labels);
}


bool ProposalDesigner::checkLabelValidity(const int pixel, const int label, const int num_surfaces, const map<int, Segment> &segments)
{
  double previous_depth = 0;
  
  bool has_depth_conflict = false;
  bool has_same_label = false;
  bool empty_background = false;
  bool segmentation_inconsistency = false;
  bool background_inconsistency = false;
  bool has_layer_estimation_conflict = false;
  bool sub_region_extended = false;
      
  int foremost_non_empty_surface_id = -1;
  vector<bool> used_surface_id_mask(num_surfaces, false);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = label / static_cast<int>(pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (num_surfaces + 1);
    if (surface_id == num_surfaces) {
      if (layer_index == NUM_LAYERS_ - 1)
        empty_background = true;
      continue;
    }
    double depth = segments.at(surface_id).getDepth(pixel);
    
    if (used_surface_id_mask[surface_id] == true) {
      has_same_label = true;
      break;
    }
    used_surface_id_mask[surface_id] = true;
    if (foremost_non_empty_surface_id == -1) {
      foremost_non_empty_surface_id = surface_id;
    }
    if (depth < previous_depth - statistics_.depth_conflict_tolerance) {
      has_depth_conflict = true;
      break;
    }
    previous_depth = depth;
  }
  if (has_depth_conflict == false && has_same_label == false && empty_background == false)
    return true;
  else
    return false;
}


bool ProposalDesigner::generateSegmentRefittingProposal()
{
  cout << "generate segment refitting proposal" << endl;
  proposal_type_ = "segment_refitting_proposal";
  
  const int SMALL_SEGMENT_NUM_PIXELS_THRESHOLD = 10;  
  
  vector<set<int> > layer_surface_ids_vec(NUM_LAYERS_);
  map<int, map<int, vector<int> > > segment_layer_visible_pixels;
  map<int, map<int, vector<int> > > segment_layer_pixels;
  vector<bool> occluded_segment_mask(NUM_LAYERS_, false);
  vector<bool> background_segment_mask(NUM_LAYERS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	segment_layer_pixels[surface_id][layer_index].push_back(pixel);
	if (is_visible == true) {
	  segment_layer_visible_pixels[surface_id][layer_index].push_back(pixel);
          is_visible = false;
	} else
	  occluded_segment_mask[surface_id] = true;
        layer_surface_ids_vec[layer_index].insert(surface_id);
	if (layer_index == NUM_LAYERS_ - 1)
	  background_segment_mask[surface_id] = true;
      }
    }
  }
  
  
  proposal_segments_ = current_solution_segments_;

  int new_proposal_segment_index = current_solution_num_surfaces_;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (map<int, map<int, vector<int> > >::const_iterator segment_it = segment_layer_pixels.begin(); segment_it != segment_layer_pixels.end(); segment_it++) {
    for (map<int, vector<int> >::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(segment_it->first);
  }

  for (map<int, map<int, vector<int> > >::const_iterator segment_it = segment_layer_visible_pixels.begin(); segment_it != segment_layer_visible_pixels.end(); segment_it++) {
    vector<int> visible_pixels;
    
    for (map<int, vector<int> >::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      visible_pixels.insert(visible_pixels.end(), layer_it->second.begin(), layer_it->second.end());

    if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
      continue;

    vector<bool> fitting_pixel_mask(NUM_PIXELS_, false);
    for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++)
      fitting_pixel_mask[*pixel_it] = true;
    
    vector<double> depth_plane_1;
    {
      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  fitting_pixel_mask[*pixel_it] = false;

	for (map<int, vector<int> >::const_iterator layer_it = segment_layer_pixels[segment_it->first].begin(); layer_it != segment_layer_pixels[segment_it->first].end(); layer_it++)
	  for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	    layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
	
	proposal_segments_[new_proposal_segment_index] = segment;
	new_proposal_segment_index++;

	depth_plane_1 = segment.getDepthPlane();
      }
    }
    
    {
      vector<int> fitting_pixels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
        if (fitting_pixel_mask[pixel] == true)
          fitting_pixels.push_back(pixel);

      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_pixels, penalties_, statistics_);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	vector<double> depth_plane_2 = segment.getDepthPlane();

	double cos_value = 0;
	for (int c = 0; c < 3; c++)
	  cos_value += depth_plane_1[c] * depth_plane_2[c];
	double angle = acos(min(abs(cos_value), 1.0));
	if (angle > statistics_.similar_angle_threshold) {
	  for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	    layer_pixel_segment_indices_map[segment_it->second.begin()->first][*pixel_it].insert(new_proposal_segment_index);
	  
	  proposal_segments_[new_proposal_segment_index] = segment;
	  new_proposal_segment_index++;
	}
      }
    }

    if (background_segment_mask[segment_it->first] == false) {
      if (visible_pixels.size() <= statistics_.bspline_surface_num_pixels_threshold) {
        Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_, 2);
	vector<int> fitted_pixels = segment.getSegmentPixels();
        if (fitted_pixels.size() >= SMALL_SEGMENT_NUM_PIXELS_THRESHOLD && segment.getType() >= 0) {
	  for (map<int, vector<int> >::const_iterator layer_it = segment_layer_pixels[segment_it->first].begin(); layer_it != segment_layer_pixels[segment_it->first].end(); layer_it++)
	    for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
	      layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
	  
	  proposal_segments_[new_proposal_segment_index] = segment;
	  new_proposal_segment_index++;
	}
      }
    }
  }
  
  const int NUM_DILATION_ITERATIONS = 2;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    while (true) {
      bool has_change = false;
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (proposal_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) && dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
              has_change = true;
            }
          }
        }
      }
      if (has_change == false)
        break;
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }

    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (proposal_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }

    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }

  // {
  //   Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   map<int, Vec3b> color_table;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     int layer_index = 0;
  //     int segment_index = 1;
  //     for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++)
  // 	if (*segment_it >= current_solution_num_surfaces_)
  // 	  segment_index *= (*segment_it + 1);
  //     segment_index = layer_pixel_segment_indices_map[1][pixel].count(8) > 0 ? 1 : 0;
  //     if (color_table.count(segment_index) == 0) {
  // 	Vec3b color;
  // 	for (int c = 0; c < 3; c++)
  // 	  color[c] = rand() % 256;
  // 	color_table[segment_index] = color;
  //     }
  //     new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
  
  //   }
  //   imwrite("Test/refitted_segment_image.bmp", new_segment_image);
  // }
  
  proposal_num_surfaces_ = proposal_segments_.size();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	pixel_layer_surfaces_map[layer_index].insert(surface_id);
      } else {
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
      }
    }
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;


    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
    
  }
  
  addIndicatorVariables();
  return true;
}



bool ProposalDesigner::generateSingleSurfaceExpansionProposal(const int denoted_expansion_segment_id)
{
  cout << "generate single surface expansion proposal" << endl;
  proposal_type_ = "single_surface_expansion_proposal";

  if (single_surface_candidate_pixels_.size() == 0) {
    single_surface_candidate_pixels_.assign(NUM_PIXELS_ * 2, -1);
    for (int pixel = 0; pixel < NUM_PIXELS_ * 2; pixel++)
      single_surface_candidate_pixels_[pixel] = pixel;
  }
  
  int expansion_segment_id = denoted_expansion_segment_id;
  int expansion_type = rand() % 2;
  if (current_solution_segments_.count(expansion_segment_id) == 0) {
    int random_pixel = single_surface_candidate_pixels_[rand() % single_surface_candidate_pixels_.size()];
    int current_solution_label = current_solution_labels_[random_pixel % NUM_PIXELS_];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	expansion_segment_id = surface_id;
	break;
      }
    }
    expansion_type = random_pixel / NUM_PIXELS_;
  }
  
  
  map<int, int> expansion_segment_layer_counter;
  bool is_occluded = false;
  vector<bool> expansion_segment_visible_pixel_mask(NUM_PIXELS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id == expansion_segment_id) {
        expansion_segment_layer_counter[layer_index]++;
	if (is_visible == false) {
	  is_occluded = true;
	  break;
	} else
	  expansion_segment_visible_pixel_mask[pixel] = true;
      }
      if (surface_id < current_solution_num_surfaces_)
	is_visible = false;
    }
  }
  vector<int> new_single_surface_candidate_pixels;
  for (vector<int>::const_iterator pixel_it = single_surface_candidate_pixels_.begin(); pixel_it != single_surface_candidate_pixels_.end(); pixel_it++)
    if (*pixel_it / NUM_PIXELS_ != expansion_type || expansion_segment_visible_pixel_mask[*pixel_it % NUM_PIXELS_] == false)
      new_single_surface_candidate_pixels.push_back(*pixel_it);
  single_surface_candidate_pixels_ = new_single_surface_candidate_pixels;
  

  if (expansion_segment_layer_counter.size() > 1 && expansion_type == 1)
    return false;
  if (is_occluded && expansion_type == 0)
    return false;

  
  int expansion_segment_layer_index = -1;
  int max_layer_count = 0;
  for (map<int, int>::const_iterator layer_it = expansion_segment_layer_counter.begin(); layer_it != expansion_segment_layer_counter.end(); layer_it++) {
    if (layer_it->second > max_layer_count) {
      expansion_segment_layer_index = layer_it->first;
      max_layer_count = layer_it->second;
    }
  }
  if (expansion_segment_layer_index == -1)
    return false;

  
  cout << "segment: " << expansion_segment_id << "\texpansion type: " << expansion_type << endl;
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    if (expansion_type == 0) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(expansion_segment_id);
      for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    } else {
      pixel_layer_surfaces_map[expansion_segment_layer_index].insert(expansion_segment_id);
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - expansion_segment_layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_ && surface_id != expansion_segment_id)
	for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
      for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    }
        
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
  }
  
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateLayerSwapProposal()
{
  cout << "generate layer swap proposal" << endl;
  proposal_type_ = "layer_swap_proposal";


  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
        layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_id);
    }
  }

  const int NUM_DILATION_ITERATIONS = 2;
  for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    
    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
    }
    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }

  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;

  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];    

    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      if (layer_index < NUM_LAYERS_ - 1)
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
    for (int layer_index = 0; layer_index < NUM_LAYERS_ - 1; layer_index++) {
      set<int> segments = layer_pixel_segment_indices_map[layer_index][pixel];
      for (int target_layer_index = max(0, layer_index - 1); target_layer_index <= min(NUM_LAYERS_ - 1, layer_index + 1); target_layer_index++)
	pixel_layer_surfaces_map[target_layer_index].insert(segments.begin(), segments.end());
    }
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);    

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateConcaveHullProposal(const bool consider_background)
{
  cout << "generate concave hull proposal" << endl;
  proposal_type_ = "concave_hull_proposal";
  

  vector<vector<int>> layer_pixel_inpainting_surface_ids(NUM_LAYERS_);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<int> pixel_inpainting_surface_ids(NUM_PIXELS_, -1);
    if (layer_index == 0) {
      layer_pixel_inpainting_surface_ids[layer_index] = pixel_inpainting_surface_ids;
      continue;
    }

    vector<int> layer_surface_ids(NUM_PIXELS_);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      layer_surface_ids[pixel] = surface_id;
    }

    vector<bool> visited_pixel_mask(NUM_PIXELS_, false);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      if (visited_pixel_mask[pixel] == true || layer_surface_ids[pixel] == current_solution_num_surfaces_)
        continue;
      vector<bool> region_mask(NUM_PIXELS_, false);
      vector<int> border_pixels;
      border_pixels.push_back(pixel);
      visited_pixel_mask[pixel] = true;
      while (true) {
        vector<int> new_border_pixels;
        for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
          region_mask[*border_pixel_it] = true;
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
            if (layer_surface_ids[*neighbor_pixel_it] != current_solution_num_surfaces_ && visited_pixel_mask[*neighbor_pixel_it] == false) {
	      new_border_pixels.push_back(*neighbor_pixel_it);
	      visited_pixel_mask[*neighbor_pixel_it] = true;	
  	    }
          }
	}
        if (new_border_pixels.size() == 0)
          break;
        border_pixels = new_border_pixels;
      }

      unique_ptr<ConcaveHullFinder> concave_hull_finder(new ConcaveHullFinder(IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, layer_surface_ids, current_solution_segments_, region_mask, penalties_, statistics_, consider_background));
      vector<int> concave_hull = concave_hull_finder->getConcaveHull();
      if (concave_hull.size() == 0)
	continue;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	if (region_mask[pixel] == true)
	  pixel_inpainting_surface_ids[pixel] = concave_hull[pixel];
    }
    layer_pixel_inpainting_surface_ids[layer_index] = pixel_inpainting_surface_ids;
  }

  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;

  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  int max_num_proposals = 0;  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      if (layer_pixel_inpainting_surface_ids[layer_index][pixel] != -1 && layer_pixel_inpainting_surface_ids[layer_index][pixel] != surface_id) {
      	pixel_layer_surfaces_map[layer_index].insert(layer_pixel_inpainting_surface_ids[layer_index][pixel]);
	for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
      }
    }
    for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_ - 1; target_layer_index++)
      pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  addIndicatorVariables();
  return true;
}


bool ProposalDesigner::generateSegmentAddingProposal(const int denoted_segment_adding_type)
{
  cout << "generate segment adding proposal" << endl;
  proposal_type_ = "segment_adding_proposal";

  int segment_adding_type = denoted_segment_adding_type;
  if (segment_adding_type == -1)
    segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : rand() % 3 != 0 ? 1 : 1;
  segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : 1;
  
  
  vector<bool> bad_fitting_pixel_mask(NUM_PIXELS_, true);
  if (segment_adding_type != 0) {
    Mat bad_fitting_pixel_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (surface_id < current_solution_num_surfaces_) {
	  if (current_solution_segments_.at(surface_id).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) == false && (point_cloud_[pixel * 3 + 2] < current_solution_segments_.at(surface_id).getDepth(pixel) + statistics_.depth_conflict_tolerance || layer_index == 0))
            bad_fitting_pixel_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 255;
	  break;
	}
      }
    }
    Mat closed_bad_fitting_pixel_image = bad_fitting_pixel_image.clone();
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    for (int iteration = 0; iteration < 1; iteration++) {
      erode(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      dilate(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      erode(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
      dilate(closed_bad_fitting_pixel_image, closed_bad_fitting_pixel_image, element);
    }
    
    //imwrite("Test/bad_fitting_pixel_mask_image.bmp", closed_bad_fitting_pixel_image);
    bad_fitting_pixel_mask = vector<bool>(NUM_PIXELS_, false);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (closed_bad_fitting_pixel_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) > 128)
    	bad_fitting_pixel_mask[pixel] = true;
    
  }
  
  vector<double> visible_depths(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
          visible_depths[pixel] = depth;
          is_visible = false;
        }
      }
    }
  }
  
  
  int foremost_empty_layer_index = NUM_LAYERS_ - 1;
  if (segment_adding_type == 1) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      for (int layer_index = 0; layer_index < foremost_empty_layer_index + 1; layer_index++) {
	int current_solution_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (current_solution_surface_id < current_solution_num_surfaces_)
	  foremost_empty_layer_index = layer_index - 1;
      }
      if (foremost_empty_layer_index == -1)
	break;
    }
  }
  
  proposal_segments_ = current_solution_segments_;
  vector<set<int> > pixel_segment_indices_map(NUM_PIXELS_);

  const int SMALL_SEGMENT_NUM_PIXELS_THRESHOLD = 10;
  int num_bad_fitting_pixels = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    if (bad_fitting_pixel_mask[pixel])
      num_bad_fitting_pixels++;
  const int NUM_FITTED_PIXELS_THRESHOLD = num_bad_fitting_pixels * 0.8;
  
  {
    vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
    int num_fitted_pixels = 0;
    int proposal_segment_index = current_solution_num_surfaces_;
    while (true) {
      vector<int> visible_pixels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	if (unfitted_pixel_mask[pixel] == true)
	  visible_pixels.push_back(pixel);
      if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
        break;
      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_);
      vector<int> fitted_pixels = segment.getSegmentPixels();
      if (fitted_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
	break;
      if (segment.getType() < 0)
	continue;
      num_fitted_pixels += fitted_pixels.size();
      proposal_segments_[proposal_segment_index] = segment;
      for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++) {
	unfitted_pixel_mask[*pixel_it] = false;
	pixel_segment_indices_map[*pixel_it].insert(proposal_segment_index);
      }
      if (num_fitted_pixels > NUM_FITTED_PIXELS_THRESHOLD)
	break;
      proposal_segment_index++;
    }
  }

  {
    if (segment_adding_type != 0) {
      vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
      int proposal_segment_index = proposal_segments_.size();
      int num_new_planes = proposal_segments_.size() - current_solution_num_surfaces_;
      for (int i = 0; i < num_new_planes; i++) {
	vector<int> visible_pixels;
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	  if (unfitted_pixel_mask[pixel] == true)
	    visible_pixels.push_back(pixel);
	if (visible_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
          break;

        Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, visible_pixels, penalties_, statistics_, 2);
      
	vector<int> fitted_pixels = segment.getSegmentPixels();
	if (fitted_pixels.size() < SMALL_SEGMENT_NUM_PIXELS_THRESHOLD)
	  break;

	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  unfitted_pixel_mask[*pixel_it] = false;

	if (fitted_pixels.size() > statistics_.bspline_surface_num_pixels_threshold || segment.getType() < 0)
	  continue;
      
	proposal_segments_[proposal_segment_index] = segment;
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  pixel_segment_indices_map[*pixel_it].insert(proposal_segment_index);
	proposal_segment_index++;
      }
    }
  }

  const int NUM_DILATION_ITERATIONS = 2;
  {
    while (true) {
      bool has_change = false;
      
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (proposal_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) && dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
	      dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      has_change = true;
	    }
	  }
	}
      }
      if (has_change == false)
	break;
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }
    

    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (proposal_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
          }
        }
      }
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
    }
  }


  if (segment_adding_type == 0) {
    const int NUM_PIXEL_SEGMENTS_THRESHOLD = 2;
    vector<bool> unfitted_pixel_mask = bad_fitting_pixel_mask;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (pixel_segment_indices_map[pixel].size() > 0)
        unfitted_pixel_mask[pixel] = true;
    while (true) {
      bool has_change = false;
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        if (unfitted_pixel_mask[pixel] == false)
          continue;
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
          for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); neighbor_segment_it++) {
            if (proposal_segments_[*neighbor_segment_it].getDepth(pixel) > 0)
              new_pixel_segment_indices_map[pixel].insert(*neighbor_segment_it);
          }
        }
        
        if (new_pixel_segment_indices_map[pixel].size() >= NUM_PIXEL_SEGMENTS_THRESHOLD)
          unfitted_pixel_mask[pixel] = false;
        if (new_pixel_segment_indices_map[pixel].size() != pixel_segment_indices_map[pixel].size())
          has_change = true;
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
      if (has_change == false)
        break;
    }
  }

  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, pixel_segment_indices_map);
  
  // {
  //   Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   map<int, Vec3b> color_table;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (layer_pixel_segment_indices_map[NUM_LAYERS_ - 1][pixel].size() == 0)
  //       continue;
  //     int segment_index = 1;
  //     for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
  // 	if (*segment_it >= current_solution_num_surfaces_)
  // 	  segment_index *= (*segment_it + 1);
  //     if (color_table.count(segment_index) == 0) {
  //       Vec3b color;
  //       for (int c = 0; c < 3; c++)
  //         color[c] = rand() % 256;
  //       color_table[segment_index] = color;
  //     }
  //     new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_index];
  //   }
  //   imwrite("Test/new_segment_image.bmp", new_segment_image);
  // }
  
  cout << "number of new segments: " << proposal_segments_.size() - current_solution_num_surfaces_ << endl;
  if (proposal_segments_.size() - current_solution_num_surfaces_ == 0)
    return false;

  proposal_num_surfaces_ = proposal_segments_.size();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];

    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
	pixel_layer_surfaces_map[layer_index].insert(surface_id);
      else
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }

    pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());
    if (segment_adding_type == 0 && pixel_segment_indices_map[pixel].size() == 0)
      for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
	pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(new_segment_index);

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
  }
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateStructureExpansionProposal(const int denoted_expansion_layer_index, const int denoted_expansion_pixel)
{
  cout << "generate structure expansion proposal" << endl;
  proposal_type_ = "structure_expansion_proposal";

  vector<bool> candidate_segment_mask(current_solution_num_surfaces_, true);
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<double> background_depths(NUM_PIXELS_, -1);
  vector<int> segment_backmost_layer_index_map(current_solution_num_surfaces_, 0);
  vector<int> visible_segmentation(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    bool is_background = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
	  visible_depths[pixel] = depth;
	  visible_segmentation[pixel] = surface_id;
          is_visible = false;
        }
	if (layer_index == NUM_LAYERS_ - 1) {
	  background_depths[pixel] = depth;
          candidate_segment_mask[surface_id] = false;
	}
	segment_backmost_layer_index_map[surface_id] = max(segment_backmost_layer_index_map[surface_id], layer_index);
      }
    }
  }
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
    if (segment_it->second.getType() != 0)
      candidate_segment_mask[segment_it->first] = false;
  }
  
  unique_ptr<StructureFinder> structure_finder(new StructureFinder(IMAGE_WIDTH_, IMAGE_HEIGHT_, current_solution_segments_, candidate_segment_mask, visible_segmentation, visible_depths, background_depths, segment_backmost_layer_index_map, penalties_, statistics_));
  vector<pair<double, vector<int> > > structure_score_surface_ids_pairs = structure_finder->getStructures();
  if (structure_score_surface_ids_pairs.size() == 0)
    return false;
  
  double score_sum = 0;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++)
    score_sum += structure_score_surface_ids_pairs[pair_index].first;

  vector<int> structure_surface_ids;
  double selected_score = cv_utils::randomProbability() * score_sum;
  score_sum = 0;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++) {
    score_sum += structure_score_surface_ids_pairs[pair_index].first;
    if (score_sum >= selected_score) {
      structure_surface_ids = structure_score_surface_ids_pairs[pair_index].second;
     break;
    }
  }

  int backmost_layer_index = 0;
  set<int> structure_surfaces;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int surface_id = structure_surface_ids[pixel];
    if (surface_id == -1)
      continue;
    backmost_layer_index = max(backmost_layer_index, segment_backmost_layer_index_map[surface_id]);
  }

  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);

      if (surface_id < proposal_num_surfaces_ && layer_index <= backmost_layer_index)
	for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
    }
    if (structure_surface_ids[pixel] != -1)
      pixel_layer_surfaces_map[backmost_layer_index].insert(structure_surface_ids[pixel]);

    for (int target_layer_index = 0; target_layer_index < backmost_layer_index; target_layer_index++)
      pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateBackwardMergingProposal(const int denoted_target_layer_index)
{
  cout << "generate backward merging proposal" << endl;
  proposal_type_ = "backward_merging_proposal";
  
  int target_layer_index = denoted_target_layer_index;
  if (target_layer_index == -1) {
    int random_pixel = rand() % NUM_PIXELS_;
    int current_solution_label = current_solution_labels_[random_pixel];
    for (int layer_index = 1; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        target_layer_index = layer_index;
        break;
      }
    }
  }
  target_layer_index = NUM_LAYERS_ -1;

  vector<double> background_depths(NUM_PIXELS_, -1);
  vector<double> visible_depths(NUM_PIXELS_, -1);

  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    bool is_background = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
	if (is_visible) {
	  visible_depths[pixel] = depth;
	  is_visible = false;
	}
	if (layer_index >= target_layer_index && is_background) {
	  background_depths[pixel] = depth;
	  is_background = false;
	}
      }
    }
  }
  
  vector<set<int> > pixel_segment_indices_map(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < target_layer_index; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_ && current_solution_segments_[surface_id].getType() == 0)
	pixel_segment_indices_map[pixel].insert(surface_id);
    }
  }
  
  while (true) {
    bool has_change = false;
      
    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
	for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	  if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	    continue;
	  double segment_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
	  if (segment_depth < 0)
	    continue;
	  
	  if ((segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_tolerance && segment_depth < background_depths[*neighbor_pixel_it] + statistics_.depth_conflict_tolerance) || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	    has_change = true;
	  }
	}
      }
    }
    if (has_change == false)
      break;
    pixel_segment_indices_map = new_pixel_segment_indices_map;
  }
  
  const int NUM_DILATION_ITERATIONS = 2;
  for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
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
	for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
          if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
            continue;
          if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
	    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
        }
      }
    }
    pixel_segment_indices_map = new_pixel_segment_indices_map;
  }
  

  // {
  //   Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   map<int, Vec3b> color_table;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (pixel_segment_indices_map[pixel].count(11) == 0)
  // 	continue;
  //     int segment_index = 1;
  //     for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
  // 	if (*segment_it >= current_solution_num_surfaces_)
  // 	  segment_index *= (*segment_it + 1);
  //     if (color_table.count(segment_index) == 0) {
  // 	Vec3b color;
  // 	for (int c = 0; c < 3; c++)
  // 	  color[c] = rand() % 256;
  // 	color_table[segment_index] = color;
  //     }
  //     new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(255, 255, 255);
  //   }
  //   imwrite("Test/backward_merging_image.bmp", new_segment_image);
  // }
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      
      if (layer_index < target_layer_index)
      	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
    
    pixel_layer_surfaces_map[target_layer_index].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());

    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);

    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);

    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      

    proposal_labels_[pixel] = valid_pixel_proposals;

    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }

  addIndicatorVariables();

  return true;
}


bool ProposalDesigner::generateDesiredProposal()
{
  cout << "generate desired proposal" << endl;
  proposal_type_ = "desired_proposal";
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  vector<int> visible_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    int layer_0_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0)) % (current_solution_num_surfaces_ + 1);
    int layer_1_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1)) % (current_solution_num_surfaces_ + 1);
    int layer_2_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2)) % (current_solution_num_surfaces_ + 1);

    int proposal_label = current_solution_label;
    
    if (layer_1_surface_id == 5 && layer_0_surface_id == 6)
      proposal_label += (current_solution_num_surfaces_ - layer_1_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1);
    
    proposal_labels_[pixel].push_back(proposal_label);
  }
  
  addIndicatorVariables();

  return true;
}

bool ProposalDesigner::generateSingleProposal()
{
  cout << "generate single proposal" << endl;
  proposal_type_ = "single_proposal";
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    proposal_labels_[pixel].push_back(current_solution_label);
  }
  addIndicatorVariables();
  
  return true;
}

void ProposalDesigner::initializeCurrentSolution()
{
  current_solution_labels_ = vector<int>(NUM_PIXELS_, 0);
  current_solution_num_surfaces_ = 0;
  current_solution_segments_.clear();
}

vector<int> ProposalDesigner::calcPixelProposals(const int num_surfaces, const map<int, set<int> > &pixel_layer_surfaces_map)
{
  vector<int> pixel_proposals(1, 0);
  for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++) {
    vector<int> new_pixel_proposals;
    for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
      for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
	new_pixel_proposals.push_back(*label_it + *segment_it * pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_it->first));
    pixel_proposals = new_pixel_proposals;
  }
  return pixel_proposals;
}

int ProposalDesigner::convertToProposalLabel(const int current_solution_label)
{
  int proposal_label = 0;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
    if (surface_id == current_solution_num_surfaces_)
      proposal_label += proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
    else
      proposal_label += (surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
  }
  return proposal_label;
}

vector<int> ProposalDesigner::getCurrentSolutionIndices()
{
  return current_solution_indices_;
}
