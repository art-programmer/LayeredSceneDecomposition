#ifndef __LayerDepthMap__TRWSFusion__
#define __LayerDepthMap__TRWSFusion__

#include <stdio.h>
#include <vector>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataStructure.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

using namespace std;

class TRWSFusion
{
 public:
    
  TRWSFusion(const cv::Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const RepresenterPenalties &penalties, const DataStatistics &statistics, const bool consider_surface_cost = true);
  
  // Destructor
  ~TRWSFusion();
  
  
  //find the best configuration in all proposals and use it to update the current solution
  vector<int> fuse(const vector<vector<int> > &proposal_labels, const int proposal_num_surfaces, const int proposal_num_layers, const map<int, Segment> &proposal_segments, const vector<int> &previous_solution_indices, const vector<bool> &proposal_ROI_mask = vector<bool>());

  //get information about optimization
  std::vector<double> getEnergyInfo();
  
  
 private:
  const int IMAGE_WIDTH_, IMAGE_HEIGHT_, NUM_PIXELS_;
  const cv::Mat image_;
  cv::Mat blurred_hsv_image_;
  const vector<double> point_cloud_;
  const vector<double> normals_;
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  const bool consider_surface_cost_;
  
  int proposal_num_surfaces_;
  int proposal_num_layers_;
  map<int, Segment> proposal_segments_;
  map<int, vector<double> > proposal_surface_depths_;
  vector<bool> proposal_ROI_mask_;
  vector<int> proposal_distance_to_boundaries_;
  
  
  double energy_;
  double lower_bound_;
  
  vector<int> solution_;
  vector<int> ori_labels_;
  
  
  double color_diff_var_;
  
  
  //calculate unary cost of a specific label a specific pixel
  double calcUnaryCost(const int pixel, const int label);
  
  //calculate pairwise cost
  double calcPairwiseCost(const int pixel_1, const int pixel_2, const int label_1, const int label_2);
  
  //check solution energy (mainly for debug purpose)
  double checkSolutionEnergy(const vector<int> &solution_for_check);
  
  //calculate the variance of color difference
  void calcColorDiffVar();
  
  //calculate color difference
  double calcColorDiff(const int pixel_1, const int pixel_2);
  
  //calculate overlap region after the viewpoint is moved to either left, right, up or down
  std::vector<std::vector<std::set<int> > > calcOverlapPixels(const vector<vector<int> > &proposal_labels);
};

#endif /* defined(__LayerDepthMap__TRWSFusion__) */
