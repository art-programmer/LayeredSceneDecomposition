#ifndef __LayerDepthMap__ConcaveHullFinder__
#define __LayerDepthMap__ConcaveHullFinder__

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "DataStructure.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

class ConcaveHullFinder{
  
 public:
  ConcaveHullFinder(const int image_width, const int image_height, const std::vector<double> &point_cloud, const std::vector<int> &segmentation, const std::map<int, Segment> &segments, const std::vector<bool> &ROI_mask, const RepresenterPenalties penalties, const DataStatistics statistics, const bool consider_background);
  
  ~ConcaveHullFinder();

  //get concave hull
  std::vector<int> getConcaveHull();

  
 private:
  const std::vector<int> segmentation_;
  const std::vector<double> point_cloud_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  
  
  std::map<int, std::vector<double> > surface_point_clouds_;
  std::map<int, std::vector<double> > surface_depths_;
  std::map<int, std::map<int, int> > surface_relations_;
  std::map<int, int> segment_type_map_;
  std::map<int, int> segment_direction_map_;
  std::vector<double> surface_normals_angles_;
  
  const std::vector<bool> ROI_mask_;
  const int NUM_SURFACES_;
  const int NUM_PIXELS_;
  
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  
  std::vector<int> concave_hull_labels_;
  std::set<int> concave_hull_surfaces_;
  

  //calculate concave hull
  void calcConcaveHull();
};

#endif /* defined(__LayerDepthMap__ConcaveHullFinder__) */
