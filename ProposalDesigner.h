#ifndef __LayerDepthMap__ProposalDesigner__
#define __LayerDepthMap__ProposalDesigner__

#include <vector>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <memory>

//#include "LayerInpainter.h"
//#include "GraphRepresenter.h"
//#include "LayerEstimator.h"
#include "Segment.h"

//using namespace cv;
//using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;


struct MeanshiftParams {
  double spatial_bandwidth;
  double range_bandwidth;
  int minimum_regions_area;
};

const std::string EDISON_PATH = "edison";
const std::string EDISON_EXE = "edison/edison edison/config.txt";

class ProposalDesigner{
  
 public:
  ProposalDesigner(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const int num_layers, const RepresenterPenalties penalties, const DataStatistics statistics);
  
  ~ProposalDesigner();
  
  //generate a proposal and return
  bool getProposal(int &iteration, std::vector<std::vector<int> > &proposal_labels, int &proposal_num_surfaces, std::map<int, Segment> &proposal_segments, std::string &proposal_type);
  
  //set current solution which might be used to generate new proposals
  void setCurrentSolution(const std::vector<int> &current_solution_labels, const int current_solution_num_surfaces, const std::map<int, Segment> &current_solution_segments);
  
  //initialize current solution which might be used to generate new proposals
  void initializeCurrentSolution();
  
  //get the indices (pixel-wise) of the current solution inside all proposals
  std::vector<int> getCurrentSolutionIndices();
  
  
 private:
  const cv::Mat image_;
  const std::vector<double> point_cloud_;
  const std::vector<double> normals_;
  const Eigen::MatrixXd projection_matrix_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const std::vector<double> CAMERA_PARAMETERS_;
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  
  cv::Mat blurred_hsv_image_;
  
  std::vector<bool> ROI_mask_;
  int NUM_LAYERS_;
  const int NUM_PIXELS_;
  
  std::vector<int> current_solution_labels_;
  int current_solution_num_surfaces_;
  std::map<int, Segment> current_solution_segments_;
  
  std::vector<std::vector<int> > proposal_labels_;
  int proposal_num_surfaces_;
  std::map<int, Segment> proposal_segments_;
  std::string proposal_type_;
  
  int num_confident_segments_threshold_;
  double segment_confidence_threshold_;
  
  
  std::vector<std::vector<int> > segmentations_;
  
  std::set<std::map<int, int> > used_confident_segment_layer_maps_;
  
  std::vector<int> current_solution_indices_;
  
  std::vector<int> single_surface_candidate_pixels_;
  
  std::vector<int> proposal_type_indices_;
  int proposal_type_index_ptr_;
  int all_proposal_iteration_;
  const int NUM_ALL_PROPOSAL_ITERATIONS_;
  
  //generate segment refitting proposal
  bool generateSegmentRefittingProposal();

  //generate single surface expansion proposal (provide segment_id to expand specific segment)
  bool generateSingleSurfaceExpansionProposal(const int segment_id = -1);
  
  //generate single surface expansion proposal
  bool generateLayerSwapProposal();

  //generate concave hull proposal
  bool generateConcaveHullProposal(const bool consider_background = true);
  
  //generate segment adding proposal
  bool generateSegmentAddingProposal(const int denoted_segment_adding_type = -1);

  //generate structure expansion proposal
  bool generateStructureExpansionProposal(const int layer_index = -1, const int pixel = -1);

  //generate backward merging proposal
  bool generateBackwardMergingProposal(const int denoted_target_layer_index = -1);
  
  //generate desired proposal (for debug)
  bool generateDesiredProposal();
  
  //generate a proposal identical with current solution (for debug)
  bool generateSingleProposal(); 

  //calculate possible proposals for a pixel given which surfaces will appear in which layers
  std::vector<int> calcPixelProposals(const int num_surfaces, const std::map<int, std::set<int> > &pixel_layer_surfaces_map);

  //add surface indicator variables (for formulating label cost)
  void addIndicatorVariables(const int num_indicator_variables = -1);

  //check the validity of a label
  bool checkLabelValidity(const int pixel, const int label, const int num_surfaces, const std::map<int, Segment> &segments);

  //convert a current solution label to a proposal label
  int convertToProposalLabel(const int current_solution_label);
};


#endif /* defined(__LayerDepthMap__ProposalDesigner__) */
