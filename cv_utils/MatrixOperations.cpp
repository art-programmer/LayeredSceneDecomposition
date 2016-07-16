#include "cv_utils.h"

using namespace std;
using namespace Eigen;

namespace cv_utils
{
  vector<vector<double> > calcInverse(const vector<vector<double> > &matrix)
  {
    assert(matrix.size() > 0 && matrix.size() == matrix.begin()->size());
    const int NUM_DIMENSIONS = matrix.size();
    MatrixXd matrix_eigen(NUM_DIMENSIONS, NUM_DIMENSIONS);
    for (int c = 0; c < NUM_DIMENSIONS; c++)
      for (int d = 0; d < NUM_DIMENSIONS; d++)
	matrix_eigen(c, d) = matrix[c][d];
    MatrixXd inverse_matrix_eigen = matrix_eigen.inverse();
    vector<vector<double> > inverse_matrix(NUM_DIMENSIONS, vector<double>(NUM_DIMENSIONS));
    for (int c = 0; c < NUM_DIMENSIONS; c++)
      for (int d = 0; d < NUM_DIMENSIONS; d++)
	inverse_matrix[c][d] = inverse_matrix_eigen(c, d);
    return inverse_matrix;
  }
}
