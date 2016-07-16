#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/photo/photo.hpp>
#include <gflags/gflags.h>


#include "LayerDepthRepresenter.h"
#include "utils.h"
#include "TRW_S/MRFEnergy.h"


using namespace std;
using namespace cv;

DEFINE_string(image_path, "", "Image path.");
DEFINE_string(point_cloud_path, "", "Point cloud path.");
DEFINE_int32(num_layers, 4, "The number of layers.");


int main(int argc, char* argv[]) {
  
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  srand(time(0));
  
  Mat ori_image = imread(FLAGS_image_path, 1);
  assert(ori_image);
  vector<double> ori_point_cloud = loadPointCloud(FLAGS_point_cloud_path);
  assert(ori_point_cloud.size() == ori_image.cols * ori_image.rows * 3);
  
  double zoom_scale = min(200.0 / max(ori_image.cols, ori_image.rows), 1.0);
  Mat image = ori_image.clone();
  vector<double> point_cloud = ori_point_cloud;
  zoomScene(image, point_cloud, zoom_scale, zoom_scale);
  
  
  RepresenterPenalties penalties;
  
  penalties.data_depth_pen = 2000; //data cost for depth difference
  penalties.data_normal_pen = 200; //data cost for color difference
  penalties.data_color_pen = 10; //data cost for normal difference
  penalties.data_non_plane_pen = 100; //parameter in data cost  
  
  
  penalties.smoothness_pen = 10000; //smoothness cost for depth change
  penalties.smoothness_small_constant_pen = 1; //small constant smoothness cost for smooth boundary (with label changes)
  penalties.smoothness_concave_shape_pen = 5000; //smoothness cost for concave shape
  penalties.smoothness_anisotropic_diffusion_pen = 500; //smoothness cost based on color difference (anisotropic diffusion)
  
  penalties.other_viewpoint_smoothness_pen = 2000; //smoothness cost for neighboring pixels from other viewpoint
  penalties.other_viewpoint_depth_conflict_pen = 200000; //depth conflict penalty for different layers at the same pixel from other viewpoint
  
  penalties.surface_pen = 20000; //label cost for the occurrence of a surface
  
  penalties.smoothness_empty_non_empty_ratio = 0.05; //the ratio of smoothness cost between a empty pixel and a non-empty pixel over the smoothness cost for a depth change of statistics.depth_change_smoothness_threshold
  
  penalties.huge_pen = 1000000; //a huge penalty for cases with conflicts
  
  
  DataStatistics statistics;
  
  statistics.pixel_fitting_distance_threshold = 0.03; //the 3D distance threshold when fitting a surface model
  statistics.pixel_fitting_angle_threshold = 30 * M_PI / 180; //the angle threshold when fitting a surface model
  statistics.pixel_fitting_color_likelihood_threshold = -20; //the color likelihood threshold when fitting a surface model
  statistics.depth_diff_var = 0.01; //the variance of depth difference for unary cost calculation
  statistics.similar_angle_threshold = 20 * M_PI / 180; //the angle threshold for two vectors to be regarded as parallel or vertical
  
  statistics.viewpoint_movement = 0.1; //the amount of viewpoint movement for parallex term calculation
  
  statistics.depth_conflict_tolerance = 0.03; //the tolerance for depth conflict between two layers at the same pixel 
  statistics.depth_change_smoothness_threshold = 0.02; //depth change threshold for two surfaces to be regarded as smooth at the intersection
  statistics.bspline_surface_num_pixels_threshold = image.cols * image.rows / 50; //the number pixels allowed to appear in a b-spline surface
  statistics.background_depth_diff_tolerance = 0.05; //small amount of depth difference allowed without penalty in the background layer
  
  LayerDepthRepresenter representer(image, point_cloud, penalties, statistics, ori_image, ori_point_cloud, FLAGS_num_layers);
}
