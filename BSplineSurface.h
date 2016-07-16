#ifndef __LayerDepthMap__BSplineSurface__
#define __LayerDepthMap__BSplineSurface__

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataStructure.h"
//#include "BSpline.h"


class BSplineSurface{
  
 public:
  BSplineSurface(const std::vector<double> &point_cloud, const std::vector<int> &pixels, const int image_width, const int image_heigth, const double stride_x, const double stride_y, const int bspline_order);
  
  //get depth map
  std::vector<double> getDepthMap() const;
  
 private:
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const int NUM_PIXELS_;
  const double STRIDE_X_;
  const double STRIDE_Y_;
  const int BSPLINE_ORDER_;
  
  std::vector<int> segment_pixels_;
  std::vector<double> control_point_xs_;
  std::vector<double> control_point_ys_;
  
  std::vector<double> depth_map_;
  

  //initialize control points
  void initControlPoints();

  //fit b-spline surface
  void fitBSplineSurface(const std::vector<double> &point_cloud, const std::vector<int> &pixels);

  //calculate 2D basis function value
  double calcBasisFunctionValue2D(const double x, const double y, const double control_point_x, const double control_point_y, const double stride_x, const double stride_y, const int order);
  
  //calculate 1D basis function value
  double calcBasisFunctionValue1D(const double &x, const double &control_point_x, const double &stride_x, const int &order);
};

#endif /* defined(__LayerDepthMap__BSplineSurface__) */
