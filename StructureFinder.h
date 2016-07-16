#ifndef __LayerDepthMap__StructureFinder__
#define __LayerDepthMap__StructureFinder__

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "DataStructure.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

class StructureFinder{

 public:
  StructureFinder(const int image_width, const int image_height, const std::map<int, Segment> &segments, const std::vector<bool> &candidate_segment_mask, const std::vector<int> visible_segmentation, const std::vector<double> &visible_depths, const std::vector<double> &background_depths, const std::vector<int> &segment_backmost_layer_index_map, const RepresenterPenalties penalties, const DataStatistics statistics);

  std::vector<std::pair<double, std::vector<int> > > getStructures() const;
  
 private:
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;

  const int NUM_SURFACES_;
  const int NUM_PIXELS_;

  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;

  const std::map<int, Segment> segments_;
  const std::vector<int> visible_segmentation_;
  const std::vector<bool> candidate_segment_mask_;
  const std::vector<double> visible_depths_;
  const std::vector<double> background_depths_;
  const std::vector<int> &segment_backmost_layer_index_map_;
  
  std::vector<std::pair<double, std::vector<int> > > structure_score_surface_ids_pairs_;

    
  void findTwoOrthogonalSurfaceStructures();
};

#endif /* defined(__LayerDepthMap__StructureFinder__) */
