#ifndef __LayerDepthMap__DataStructure__
#define __LayerDepthMap__DataStructure__

#include <fstream>
#include <cmath>

struct RepresenterPenalties {
  double data_depth_pen;
  double data_color_pen;
  double data_normal_pen;
  double data_non_plane_pen;
  
  double surface_pen;
  
  double smoothness_pen;
  double smoothness_small_constant_pen;
  double smoothness_concave_shape_pen;
  double smoothness_anisotropic_diffusion_pen;
  
  double other_viewpoint_smoothness_pen;
  double other_viewpoint_depth_conflict_pen;
  
  double smoothness_empty_non_empty_ratio;
  
  double huge_pen;
};

struct DataStatistics {
  double pixel_fitting_distance_threshold;
  double pixel_fitting_angle_threshold;
  double pixel_fitting_color_likelihood_threshold;
  double depth_diff_var;
  double similar_angle_threshold;
  double depth_conflict_tolerance;
  double depth_change_smoothness_threshold;
  double viewpoint_movement;
  double bspline_surface_num_pixels_threshold;
  double background_depth_diff_tolerance;
};

#endif /* defined(__LayerDepthMap__DataStructure__) */
