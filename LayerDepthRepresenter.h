#ifndef __LayerDepthMap__LayerDepthRepresenter__
#define __LayerDepthMap__LayerDepthRepresenter__

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <memory>
//#include "ProposalGenerator.h"
#include "Segment.h"

#include <Eigen/Dense>


class LayerDepthRepresenter {
  
 public:
  LayerDepthRepresenter(const cv::Mat &image, const std::vector<double> &point_cloud, const RepresenterPenalties &penalties, const DataStatistics &statistics, const cv::Mat &ori_image, const std::vector<double> &ori_point_cloud, const int num_layers);
  
  ~LayerDepthRepresenter();
  
  
 private:
  const cv::Mat image_;
  const cv::Mat ori_image_;
  const std::vector<double> point_cloud_;
  const std::vector<double> ori_point_cloud_;
  std::vector<double> normals_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const int NUM_PIXELS_;
  
  const RepresenterPenalties PENALTIES_;
  const DataStatistics STATISTICS_;
  
  
  std::map<int, std::vector<double> > surface_depths_;
  std::map<int, int> surface_colors_;
  double max_depth_;
  
  //  vector<int> labels_;
  
  std::vector<bool> ROI_mask_;
  int num_surfaces_;
  int num_layers_;
  
//unique_ptr<ProposalGenerator> proposal_generator_;
  
  std::vector<std::vector<int> > layers_;
  
  std::map<int, std::set<int> > layer_surfaces_;
  std::map<int, std::set<int> > layer_front_surfaces_;
  std::map<int, std::set<int> > layer_back_surfaces_;
  
  std::vector<double> camera_parameters_;
  
  double disp_image_numerator_;
  

  //optimize layer representation
  void optimizeLayerRepresentation();
  
  //write rendering information
  void writeRenderingInfo(const std::vector<int> &solution, const int solution_num_surfaces, const std::map<int, Segment> &solution_segments);

  //generate a HTML page containing result images
  void generateLayerImageHTML(const std::map<int, std::vector<double> > &iteration_statistics_map, const std::map<int, std::string> &iteration_proposal_type_map);
  
  //upsample results to original resolution
  void upsampleSolution(const std::vector<int> &solution_labels, const int solution_num_surfaces, const std::map<int, Segment> &solution_segments, std::vector<int> &upsampled_solution_labels, int &upsampled_solution_num_surfaces, std::map<int, Segment> &upsampled_solution_segments);
};

//write intermediate results to cache
void writeLayers(const cv::Mat &image, const int image_width, const int image_height, const std::vector<double> &point_cloud, const std::vector<double> &camera_parameters, const int num_layers, const std::vector<int> &solution, const int solution_num_surfaces, const std::map<int, Segment> &solution_segments, const int result_index, const cv::Mat &ori_image, const std::vector<double> &ori_point_cloud);

//read intermediate results from cache
bool readLayers(const int image_width, const int image_height, const std::vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int num_layers, std::vector<int> &solution, int &solution_num_surfaces, std::map<int, Segment> &solution_segments, const int result_index);


#endif /* defined(__LayerDepthMap__LayerDepthRepresenter__) */
