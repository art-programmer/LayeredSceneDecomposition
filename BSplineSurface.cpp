#include "BSplineSurface.h"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>


using namespace std;
using namespace cv;
using namespace Eigen;


BSplineSurface::BSplineSurface(const vector<double> &point_cloud, const vector<int> &pixels, const int image_width, const int image_height, const double stride_x, const double stride_y, const int bspline_order) : IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(image_width * image_height), STRIDE_X_(stride_x), STRIDE_Y_(stride_y), BSPLINE_ORDER_(bspline_order)
{
  initControlPoints();
  fitBSplineSurface(point_cloud, pixels);
}

double BSplineSurface::calcBasisFunctionValue2D(const double x, const double y, const double control_point_x, const double control_point_y, const double stride_x, const double stride_y, const int order)
{
  if (x < control_point_x || x >= control_point_x + stride_x * (order + 1) || y < control_point_y || y >= control_point_y + stride_y * (order + 1))
    return 0;
  return calcBasisFunctionValue1D(x, control_point_x, stride_x, order) * calcBasisFunctionValue1D(y, control_point_y, stride_y, order);
}

double BSplineSurface::calcBasisFunctionValue1D(const double &x, const double &control_point_x, const double &stride_x, const int &order)
{
  if (order == 0) {
    if (x >= control_point_x && x < control_point_x + stride_x)
      return 1;
    else
      return 0;
  }
  double weight_1 = (x - control_point_x) / (stride_x * order);
  double weight_2 = (control_point_x + stride_x * (order + 1) - x) / (stride_x * order);
  return weight_1 * calcBasisFunctionValue1D(x, control_point_x, stride_x, order - 1) + weight_2 * calcBasisFunctionValue1D(x, control_point_x + stride_x, stride_x, order - 1);
}

void BSplineSurface::initControlPoints()
{
  {
    int num_controls_points_x = IMAGE_WIDTH_ / STRIDE_X_ + BSPLINE_ORDER_ + 1;
    control_point_xs_.assign(num_controls_points_x, 0);
    double control_point_start_x = -STRIDE_X_ * BSPLINE_ORDER_;
    for (int i = 0; i < num_controls_points_x; i++)
      control_point_xs_[i] = control_point_start_x + STRIDE_X_ * i;
  }
  {
    int num_controls_points_y = IMAGE_HEIGHT_ / STRIDE_Y_ + BSPLINE_ORDER_ + 1;
    control_point_ys_.assign(num_controls_points_y, 0);
    double control_point_start_y = -STRIDE_Y_ * BSPLINE_ORDER_;
    for (int i = 0; i < num_controls_points_y; i++)
      control_point_ys_[i] = control_point_start_y + STRIDE_Y_ * i;
  }
}

void BSplineSurface::fitBSplineSurface(const vector<double> &point_cloud, const vector<int> &pixels)
{
  const double DATA_WEIGHT = 3;
  const double SMOOTHNESS_SECOND_DERIVATIVE_WEIGHT = 1;
  const double SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT = 1;

  const int NUM_CONTROL_POINTS_X = control_point_xs_.size();
  const int NUM_CONTROL_POINTS_Y = control_point_ys_.size();

  int num_valid_pixels = 0;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
    if (point_cloud[*pixel_it * 3 + 2] > 0)
      num_valid_pixels++;
  
  const int NUM_DATA_TERMS = num_valid_pixels;
  const int NUM_SMOOTHNESS_SECOND_DERIVATIVE_TERMS = NUM_CONTROL_POINTS_X * NUM_CONTROL_POINTS_Y;
  const int NUM_SMOOTHNESS_FIRST_DERIVATIVE_TERMS = (NUM_CONTROL_POINTS_X - 2) * NUM_CONTROL_POINTS_Y + NUM_CONTROL_POINTS_X * (NUM_CONTROL_POINTS_Y - 2);

  SparseMatrix<double> A(NUM_DATA_TERMS + NUM_SMOOTHNESS_SECOND_DERIVATIVE_TERMS + NUM_SMOOTHNESS_FIRST_DERIVATIVE_TERMS, NUM_CONTROL_POINTS_X * NUM_CONTROL_POINTS_Y);
  VectorXd b(NUM_DATA_TERMS + NUM_SMOOTHNESS_SECOND_DERIVATIVE_TERMS + NUM_SMOOTHNESS_FIRST_DERIVATIVE_TERMS);
  vector<Eigen::Triplet<double> > triplets;

  int valid_pixel_index = 0;
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    if (point_cloud[*pixel_it * 3 + 2] < 0)
      continue;
    int x = *pixel_it % IMAGE_WIDTH_;
    int y = *pixel_it / IMAGE_WIDTH_;
    //double sum_basis_function_values = 0;
    for (vector<double>::const_iterator control_point_x_it = control_point_xs_.begin(); control_point_x_it != control_point_xs_.end(); control_point_x_it++) {
      for (vector<double>::const_iterator control_point_y_it = control_point_ys_.begin(); control_point_y_it != control_point_ys_.end(); control_point_y_it++) {
	double basis_function_value = calcBasisFunctionValue2D(x, y, *control_point_x_it, *control_point_y_it, STRIDE_X_, STRIDE_Y_, BSPLINE_ORDER_);
	if (basis_function_value < 0.0001)
	  continue;
	triplets.push_back(Eigen::Triplet<double>(valid_pixel_index, (control_point_y_it - control_point_ys_.begin()) * NUM_CONTROL_POINTS_X + (control_point_x_it - control_point_xs_.begin()), basis_function_value * DATA_WEIGHT));
	//sum_basis_function_values += basis_function_value;
      }
    }
    b[valid_pixel_index] = point_cloud[*pixel_it * 3 + 2] * DATA_WEIGHT;
    valid_pixel_index++;
  }
  for (vector<double>::const_iterator control_point_y_it = control_point_ys_.begin(); control_point_y_it != control_point_ys_.end(); control_point_y_it++) {
    for (vector<double>::const_iterator control_point_x_it = control_point_xs_.begin(); control_point_x_it != control_point_xs_.end(); control_point_x_it++) {
      int index_x = control_point_x_it - control_point_xs_.begin();
      int index_y = control_point_y_it - control_point_ys_.begin();
      int index = index_y * NUM_CONTROL_POINTS_X + index_x;
      vector<int> neighbor_indices;
      if (index_x > 0)
        neighbor_indices.push_back(index - 1);
      if (index_x < NUM_CONTROL_POINTS_X - 1)
        neighbor_indices.push_back(index + 1);
      if (index_y > 0)
        neighbor_indices.push_back(index - NUM_CONTROL_POINTS_X);
      if (index_y < NUM_CONTROL_POINTS_Y - 1)
        neighbor_indices.push_back(index + NUM_CONTROL_POINTS_X);
      for (vector<int>::const_iterator neighbor_index_it = neighbor_indices.begin(); neighbor_index_it != neighbor_indices.end(); neighbor_index_it++)
	triplets.push_back(Eigen::Triplet<double>(NUM_DATA_TERMS + index, *neighbor_index_it, -1.0 * SMOOTHNESS_SECOND_DERIVATIVE_WEIGHT / neighbor_indices.size()));
      triplets.push_back(Eigen::Triplet<double>(NUM_DATA_TERMS + index, index, SMOOTHNESS_SECOND_DERIVATIVE_WEIGHT));
      b[NUM_DATA_TERMS + index] = 0;

      if (index_x > 0 && index_x < NUM_CONTROL_POINTS_X - 1) {
	int term_index = NUM_DATA_TERMS + NUM_SMOOTHNESS_SECOND_DERIVATIVE_TERMS + index_y * (NUM_CONTROL_POINTS_X - 2) + (index_x - 1);
	triplets.push_back(Eigen::Triplet<double>(term_index, index - 1, 1 * SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
	triplets.push_back(Eigen::Triplet<double>(term_index, index + 1, -1 * SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
	//        triplets.push_back(Eigen::Triplet<double>(term_index, index, SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
	b[term_index] = 0;
      }
      if (index_y > 0 && index_y < NUM_CONTROL_POINTS_Y - 1) {
	int term_index = NUM_DATA_TERMS + NUM_SMOOTHNESS_SECOND_DERIVATIVE_TERMS + (NUM_CONTROL_POINTS_X - 2) * NUM_CONTROL_POINTS_Y + (index_y - 1) * NUM_CONTROL_POINTS_X + index_x;
        triplets.push_back(Eigen::Triplet<double>(term_index, index - NUM_CONTROL_POINTS_X, 1 * SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
        triplets.push_back(Eigen::Triplet<double>(term_index, index + NUM_CONTROL_POINTS_X, -1 * SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
	//        triplets.push_back(Eigen::Triplet<double>(term_index, index, SMOOTHNESS_FIRST_DERIVATIVE_WEIGHT));
	b[term_index] = 0;
      }
    }
  }
  A.setFromTriplets(triplets.begin(), triplets.end());
  
  VectorXd control_point_depths = SparseQR<SparseMatrix<double>, NaturalOrdering<int> >(A).solve(b);
  
  depth_map_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double depth = 0;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    for (vector<double>::const_iterator control_point_x_it = control_point_xs_.begin(); control_point_x_it != control_point_xs_.end(); control_point_x_it++) {
      for (vector<double>::const_iterator control_point_y_it = control_point_ys_.begin(); control_point_y_it != control_point_ys_.end(); control_point_y_it++) {
	double basis_function_value = calcBasisFunctionValue2D(x, y, *control_point_x_it, *control_point_y_it, STRIDE_X_, STRIDE_Y_, BSPLINE_ORDER_);
	depth += control_point_depths[(control_point_y_it - control_point_ys_.begin()) * NUM_CONTROL_POINTS_X + (control_point_x_it - control_point_xs_.begin())] * basis_function_value;
      }
    }
    depth_map_[pixel] = depth;
  }

  
  // double fitting_error = 0;
  // for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++)
  //   fitting_error += pow(depth_map_[*pixel_it] - point_cloud[*pixel_it * 3 + 2], 2);
  // //    cout << depth_map_[*pixel_it] << '\t' << point_cloud[*pixel_it * 3 + 2] << endl;
  // cout << sqrt(fitting_error / pixels.size()) << endl;  
  
  // Mat disp_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // for (int y = 0; y < IMAGE_HEIGHT_; y++) {
  //   for (int x = 0; x < IMAGE_WIDTH_; x++) {
  //     int pixel = y * IMAGE_WIDTH_ + x;
  //     double depth = depth_map_[pixel];
  //     disp_image.at<uchar>(y, x) = 100 / depth;
  //   }
  // }
  // imwrite("Test/bspline_image.bmp", disp_image);
  // exit(1);
}

vector<double> BSplineSurface::getDepthMap() const
{
  return depth_map_;
}
