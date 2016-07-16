#ifndef __LayerDepthMap__Segment__
#define __LayerDepthMap__Segment__

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataStructure.h"
//#include "BSpline.h"


class Segment{
  
 public:
  Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const std::vector<int> &pixels, const RepresenterPenalties &penalties, const DataStatistics &input_statistics = DataStatistics(), const int segment_type = 0);
  Segment(const int image_width, const int image_height, const std::vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics);
  //Segment(const Segment &segment);
  Segment();
  
  //~Segment();
  
  //write segment to file
  friend std::ostream & operator <<(std::ostream &out_str, const Segment &segment);
  
  //read segment from file
  friend std::istream & operator >>(std::istream &in_str, Segment &segment);
  
  //segment assignment
  Segment &operator = (const Segment &segment);
  
  
  //predict color likelihood based on the GMM model
  double predictColorLikelihood(const int pixel, const cv::Vec3f hsv_color) const;
  
  //set GMM model based on saved file.
  void setGMM(const cv::FileNode GMM_file_node);
  
  //get GMM model in order to save to file.
  cv::Ptr<cv::ml::EM> getGMM() const;
  
  //get depth map
  std::vector<double> getDepthMap() const;
  
  //get depth at specific pixel
  double getDepth(const int pixel) const;
  
  //get depth at pixel specified by ratios
  double getDepth(const double x_ratio, const double y_ratio) const;
  
  //get plane parameters
  std::vector<double> getDepthPlane() const;
  
  //get segment pixels
  std::vector<int> getSegmentPixels() const;

  //get segment type
  int getType() const;
  
  //check whether the segment fits the specific pixel or not
  bool checkPixelFitting(const cv::Mat &hsv_image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const int pixel) const;

  //calculate the angle between normals of the input visible surface and the segment
  double calcAngle(const std::vector<double> &normals, const int pixel);

  //calculate the difference of the distance between two pixels and the segment.
  int calcDistanceOffset(const int pixel_1, const int pixel_2);
  
  //bool buildSubSegment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &visible_pixels);
  
  //project the segmnet to different viewpoints
  std::vector<int> projectToOtherViewpoints(const int pixel, const double viewpoint_movement);

  //get the segment type
  int getSegmentType() const;
  
  //calculate 3D distance between the 3D point at specific pixel and the segment
  double calcDistance(const std::vector<double> &point_cloud, const int pixel);
  
  
 private:
  int IMAGE_WIDTH_;
  int IMAGE_HEIGHT_;
  
  int NUM_PIXELS_;
  std::vector<double> CAMERA_PARAMETERS_;
  
  
  RepresenterPenalties penalties_;
  DataStatistics input_statistics_;
  
  std::vector<int> segment_pixels_;
  std::vector<double> disp_plane_;
  std::vector<double> depth_plane_;
  std::vector<double> depth_map_;
  std::vector<double> normals_;
  
  int segment_type_;
  
  cv::Ptr<cv::ml::EM> GMM_;
  
  double segment_confidence_;
  
  std::vector<bool> segment_mask_;
  double segment_radius_;
  double segment_center_x_;
  double segment_center_y_;
  
  std::vector<int> distance_map_;
  
  
  //fit a plane segment
  void fitDepthPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  
  //fit a b-spline segment
  void fitBSplineSurface(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  
  //fit a plane segment which is parallel to the image plane
  void fitParallelSurface(const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<int> &pixels);
  
  //calculate depth map for this segment
  void calcDepthMap(const std::vector<double> &point_cloud = std::vector<double>(), const std::vector<int> &fitted_pixels = std::vector<int>());
  
  //calculate color statistics for this segment
  void calcColorStatistics(const cv::Mat &image, const std::vector<int> &pixels);
  
  //calculate mask info for this segment
  void calcSegmentMaskInfo();
  
  //the distance map stores the distance between any pixel and the segment on image domain
  void calcDistanceMap();
  
  //find largest connected component
  std::vector<int> findLargestConnectedComponent(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
  
  //calculate point cloud map for this segment.
  std::vector<double> calcPointCloud();
  
  //delete invalid pixels which have no depth values
  std::vector<int> deleteInvalidPixels(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
};

#endif /* defined(__LayerDepthMap__Segment__) */
