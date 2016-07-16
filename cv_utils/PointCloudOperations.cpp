#include "cv_utils.h"

#include <fstream>

using namespace std;

namespace cv_utils
{
  // void segmentPointCloudRansac(const vector<double> &point_cloud, const vector<vector<int> > &neighbors, vector<vector<double> > &planes, vector<int> &assignment, const double DENOTED_FITTING_ERROR_THRESHOLD, const int DENOTED_NUM_PLANES_THRESHOLD, const double DENOTED_FITTING_RATIO_THRESHOLD)
  // {
  //   const double FITTING_ERROR_THRESHOLD = DENOTED_FITTING_ERROR_THRESHOLD;
  //   const int NUM_PLANES_THRESHOLD = DENOTED_NUM_PLANES_THRESHOLD > 0 ? DENOTED_NUM_PLANES_THRESHOLD : point_cloud.size() / 3;
  //   for (int plane_index = 0; plane_index < NUM_PLANES_THRESHOLD; plane_index++) {
  //   }
  // }
  
  
  bool writePointCloud(const string &filename, const vector<double> &point_cloud, const int IMAGE_WIDTH, const int IMAGE_HEIGHT)
  {
  ofstream out_str(filename);
  if (!out_str)
    return false;
  out_str << IMAGE_WIDTH << '\t' << IMAGE_HEIGHT << endl;
  for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
    out_str << point_cloud[pixel * 3 + 0] << '\t' << point_cloud[pixel * 3 + 1] << '\t' << point_cloud[pixel * 3 + 2] << endl;
  out_str.close();
  return true;
}
  
  bool readPointCloud(const string &filename, vector<double> &point_cloud)
  {
  ifstream in_str(filename);
  if (!in_str)
    return false;
  int image_width, image_height;
  in_str >> image_width >> image_height;
  point_cloud.assign(image_width * image_height * 3, 0);
  for (int pixel = 0; pixel < image_width * image_height; pixel++)
    in_str >> point_cloud[pixel * 3 + 0] >> point_cloud[pixel * 3 + 1] >> point_cloud[pixel * 3 + 2];
  in_str.close();
  return true;
}
}
