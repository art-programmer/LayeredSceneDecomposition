#include "cv_utils.h"

#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace cv_utils
{
  
  template<typename Func> Mat drawValuesOnImage(const vector<double> &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const Func &func)
  {
    Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
      for (int x = 0; x < IMAGE_WIDTH; x++) {
	int color = round(func(values[y * IMAGE_WIDTH + x]));
	image.at<uchar>(y, x) = max(min(color, 255), 0);
      }
    }
    return image;
  }
  
  vector<double> calcBoxIntegrationMask(const vector<double> &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT)
  {
    vector<double> mask = values;
    for (int y = 0; y < IMAGE_HEIGHT; y++)
      for (int x = 1; x < IMAGE_WIDTH; x++)
	mask[y * IMAGE_WIDTH + x] += mask[y * IMAGE_WIDTH + (x - 1)];
    for (int x = 0; x < IMAGE_WIDTH; x++)
      for (int y = 1; y < IMAGE_HEIGHT; y++)
	mask[y * IMAGE_WIDTH + x] += mask[(y - 1) * IMAGE_WIDTH + x];
    return mask;
  }
  
  double calcBoxIntegration(const vector<double> &mask, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int x_1, const int y_1, const int x_2, const int y_2)
  {
    int min_x = min(x_1, x_2) - 1;
    int min_y = min(y_1, y_2) - 1;
    int max_x = min(max(x_1, x_2), IMAGE_WIDTH - 1);
    int max_y = min(max(y_1, y_2), IMAGE_HEIGHT - 1);
    double value_1 = (min_x >= 0 && min_y >= 0) ? mask[min_y * IMAGE_WIDTH + min_x] : 0;
    double value_2 = min_x >= 0 ? mask[max_y * IMAGE_WIDTH + min_x] : 0;
    double value_3 = min_y >= 0 ? mask[min_y * IMAGE_WIDTH + max_x] : 0;
    double value_4 = mask[max_y * IMAGE_WIDTH + max_x];
    return (value_1 + value_4) - (value_2 + value_3);
  }
  
  void calcWindowMeansAndVars(const std::vector<double> &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int WINDOW_SIZE, vector<double> &means, vector<double> &vars)
  {
    vector<double> sum_mask = calcBoxIntegrationMask(values, IMAGE_WIDTH, IMAGE_HEIGHT);
    vector<double> values2(IMAGE_WIDTH * IMAGE_HEIGHT);
    transform(values.begin(), values.end(), values2.begin(), [](const double &x) { return pow(x, 2); });
    vector<double> sum2_mask = calcBoxIntegrationMask(values2, IMAGE_WIDTH, IMAGE_HEIGHT);
    means.assign(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    vars.assign(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
      int x_1 = pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
      int y_1 = pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
      int x_2 = pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
      int y_2 = pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
      int area = (min(pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_WIDTH - 1) - max(pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1) * (min(pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_HEIGHT - 1) - max(pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1);
      double mean = calcBoxIntegration(sum_mask, IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area;
      double var = calcBoxIntegration(sum2_mask, IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area - pow(mean, 2);
      means[pixel] = mean;
      vars[pixel] = var;
    }
  }

  void calcWindowMeansAndVars(const std::vector<std::vector<double> > &values, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const int WINDOW_SIZE, vector<vector<double> > &means, vector<vector<double> > &vars)
  {
    const int NUM_CHANNELS = values.size();
    vector<vector<double> > sum_masks(NUM_CHANNELS);
    for (int c = 0; c < NUM_CHANNELS; c++)
      sum_masks[c] = calcBoxIntegrationMask(values[c], IMAGE_WIDTH, IMAGE_HEIGHT);
    vector<vector<double> > values2(NUM_CHANNELS * NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
    for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
      for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
	transform(values[c_1].begin(), values[c_1].end(), values[c_2].begin(), values2[c_1 * NUM_CHANNELS + c_2].begin(), [](const double &x, const double &y) { return x * y; });
    vector<vector<double> > sum2_masks(NUM_CHANNELS * NUM_CHANNELS);
    for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
      for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
	sum2_masks[c_1 * NUM_CHANNELS + c_2] = calcBoxIntegrationMask(values2[c_1 * NUM_CHANNELS + c_2], IMAGE_WIDTH, IMAGE_HEIGHT);
    
    means.assign(NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
    vars.assign(NUM_CHANNELS * NUM_CHANNELS, vector<double>(IMAGE_WIDTH * IMAGE_HEIGHT));
    for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
      int x_1 = pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
      int y_1 = pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
      int x_2 = pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
      int y_2 = pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
      int area = (min(pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_WIDTH - 1) - max(pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1) * (min(pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_HEIGHT - 1) - max(pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1);
      vector<double> mean(NUM_CHANNELS);
      for (int c = 0; c < NUM_CHANNELS; c++)
	mean[c] = calcBoxIntegration(sum_masks[c], IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area;
      vector<double> var(NUM_CHANNELS * NUM_CHANNELS);
      for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
        for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
	  var[c_1 * NUM_CHANNELS + c_2] = calcBoxIntegration(sum2_masks[c_1 * NUM_CHANNELS + c_2], IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area - mean[c_1] * mean[c_2];
      for (int c = 0; c < NUM_CHANNELS; c++)
	means[c][pixel] = mean[c];
      for (int c_1 = 0; c_1 < NUM_CHANNELS; c_1++)
        for (int c_2 = 0; c_2 < NUM_CHANNELS; c_2++)
	  vars[c_1 * NUM_CHANNELS + c_2][pixel] = var[c_1 * NUM_CHANNELS + c_2];
    }
  }

  void guidedFilter(const cv::Mat &guidance_image, const cv::Mat &input_image, cv::Mat &output_image, const double radius, const double epsilon)
  {
    const int IMAGE_WIDTH = guidance_image.cols;
    const int IMAGE_HEIGHT = guidance_image.rows;
    const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
    if (guidance_image.channels() == 1 && input_image.channels() == 1) {
      vector<double> guidance_image_values(NUM_PIXELS);
      vector<double> input_image_values(NUM_PIXELS);
      vector<double> guidance_image_input_image_values(NUM_PIXELS);
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
	for (int x = 0; x < IMAGE_WIDTH; x++) {
	  int pixel = y * IMAGE_WIDTH + x;
	  guidance_image_values[pixel] = 1.0 * guidance_image.at<uchar>(y, x) / 256;
	  input_image_values[pixel] = 1.0 * input_image.at<uchar>(y, x) / 256;
	  guidance_image_input_image_values[pixel] = (1.0 * guidance_image.at<uchar>(y, x) / 256) * (1.0 * input_image.at<uchar>(y, x) / 256);
	}
      }
      vector<double> guidance_image_means;
      vector<double> guidance_image_vars;
      calcWindowMeansAndVars(guidance_image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_means, guidance_image_vars);
      vector<double> input_image_means;
      vector<double> dummy_vars;
      calcWindowMeansAndVars(input_image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, input_image_means, dummy_vars);
      vector<double> guidance_image_input_image_means;
      calcWindowMeansAndVars(guidance_image_input_image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_input_image_means, dummy_vars);
      
      vector<double> a_values(NUM_PIXELS);
      vector<double> b_values(NUM_PIXELS);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
	a_values[pixel] = (guidance_image_input_image_means[pixel] - guidance_image_means[pixel] * input_image_means[pixel]) / (guidance_image_vars[pixel] + epsilon);
	b_values[pixel] = input_image_means[pixel] - a_values[pixel] * guidance_image_means[pixel];
      }
      
      vector<double> a_means;
      vector<double> b_means;
      calcWindowMeansAndVars(a_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, a_means, dummy_vars);
      calcWindowMeansAndVars(b_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, b_means, dummy_vars);
      
      output_image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
          int pixel = y * IMAGE_WIDTH + x;
	  output_image.at<uchar>(y, x) = max(min((a_means[pixel] * input_image_values[pixel] + b_means[pixel]) * 256, 255.0), 0.0);
	}
      }
    } else if (guidance_image.channels() == 3 && input_image.channels() == 3) {
      vector<vector<double> > guidance_image_values(3, vector<double>(NUM_PIXELS));
      vector<vector<double> > input_image_values(3, vector<double>(NUM_PIXELS));
      vector<vector<double> > guidance_image_input_image_values(3, vector<double>(NUM_PIXELS));
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
          int pixel = y * IMAGE_WIDTH + x;
	  Vec3b guidance_image_color = guidance_image.at<Vec3b>(y, x);
	  Vec3b input_image_color = input_image.at<Vec3b>(y, x);
	  for (int c = 0; c < 3; c++) {
            guidance_image_values[c][pixel] = 1.0 * guidance_image_color[c] / 256;
	    input_image_values[c][pixel] = 1.0 * input_image_color[c] / 256;
	    guidance_image_input_image_values[c][pixel] = (1.0 * guidance_image_color[c] / 256) * (1.0 * input_image_color[c] / 256);
	  }
        }
      }
      vector<vector<double> > guidance_image_means(3);
      vector<vector<double> > guidance_image_vars(3);
      vector<vector<double> > input_image_means(3);
      vector<vector<double> > guidance_image_input_image_means(3);
      for (int c = 0; c < 3; c++) {
	vector<double> dummy_vars;
	calcWindowMeansAndVars(guidance_image_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_means[c], guidance_image_vars[c]);
	calcWindowMeansAndVars(input_image_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, input_image_means[c], dummy_vars);
	calcWindowMeansAndVars(guidance_image_input_image_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_input_image_means[c], dummy_vars);
      }
      
      vector<double> a_values(NUM_PIXELS);
      vector<vector<double> > b_values(3, vector<double>(NUM_PIXELS));
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
	double guidance_image_input_image_covariance = 0;
	double guidance_image_var = 0;
	for (int c = 0; c < 3; c++) {
	  guidance_image_input_image_covariance += guidance_image_input_image_means[c][pixel] - guidance_image_means[c][pixel] * input_image_means[c][pixel];
	  guidance_image_var += guidance_image_vars[c][pixel];
	}
	a_values[pixel] = guidance_image_input_image_covariance / (guidance_image_var + epsilon);
	for (int c = 0; c < 3; c++)
	  b_values[c][pixel] = input_image_means[c][pixel] - a_values[pixel] * guidance_image_means[c][pixel];
      }
      
      vector<double> a_means;
      vector<double> dummy_vars;
      calcWindowMeansAndVars(a_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, a_means, dummy_vars);
      vector<vector<double> > b_means(3);
      for (int c = 0; c < 3; c++) {
	calcWindowMeansAndVars(b_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, b_means[c], dummy_vars);
      }
      
      output_image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
          int pixel = y * IMAGE_WIDTH + x;
	  Vec3b color;
	  for (int c = 0; c < 3; c++)
	    color[c] = max(min((a_means[pixel] * input_image_values[c][pixel] + b_means[c][pixel]) * 256, 255.0), 0.0);
	  output_image.at<Vec3b>(y, x) = color;
        }
      }
    } else if (guidance_image.channels() == 3 && input_image.channels() == 1) {
      vector<vector<double> > guidance_image_values(3, vector<double>(NUM_PIXELS));
      vector<double> input_image_values(NUM_PIXELS);
      vector<vector<double> > guidance_image_input_image_values(3, vector<double>(NUM_PIXELS));
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
	for (int x = 0; x < IMAGE_WIDTH; x++) {
	  int pixel = y * IMAGE_WIDTH + x;
	  Vec3b guidance_image_color = guidance_image.at<Vec3b>(y, x);
	  uchar input_image_color = input_image.at<uchar>(y, x);
	  input_image_values[pixel] = 1.0 * input_image_color / 256;
          for (int c = 0; c < 3; c++) {
	    guidance_image_values[c][pixel] = 1.0 * guidance_image_color[c] / 256;
	    guidance_image_input_image_values[c][pixel] = (1.0 * guidance_image_color[c] / 256) * (1.0 * input_image_color / 256);
	  }
	}
      }
      vector<vector<double> > guidance_image_means;
      vector<vector<double> > guidance_image_vars;
      calcWindowMeansAndVars(guidance_image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_means, guidance_image_vars);
      vector<double> dummy_vars;
      vector<double> input_image_means;
      calcWindowMeansAndVars(input_image_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, input_image_means, dummy_vars);
      vector<vector<double> > guidance_image_input_image_means(3);
      for (int c = 0; c < 3; c++) {
	calcWindowMeansAndVars(guidance_image_input_image_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, guidance_image_input_image_means[c], dummy_vars);
      }
      
      vector<vector<double> > a_values(3, vector<double>(NUM_PIXELS, 0));
      vector<double> b_values(NUM_PIXELS, 0);
      for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
	vector<double> guidance_image_input_image_covariance(3);
	vector<vector<double> > guidance_image_var(3, vector<double>(3));
	for (int c = 0; c < 3; c++)
	  guidance_image_input_image_covariance[c] = guidance_image_input_image_means[c][pixel] - guidance_image_means[c][pixel] * input_image_means[pixel];
	for (int c_1 = 0; c_1 < 3; c_1++)
	  for (int c_2 = 0; c_2 < 3; c_2++)
            guidance_image_var[c_1][c_2] = guidance_image_vars[c_1 * 3 + c_2][pixel] + epsilon * (c_1 == c_2);
	
	vector<vector<double> > guidance_image_var_inverse = calcInverse(guidance_image_var);
	vector<double> a_value(3, 0);
	for (int c_1 = 0; c_1 < 3; c_1++)
	  for (int c_2 = 0; c_2 < 3; c_2++)
	    a_value[c_1] += guidance_image_var_inverse[c_1][c_2] * guidance_image_input_image_covariance[c_2];
	for (int c = 0; c < 3; c++)
	  a_values[c][pixel] = a_value[c];

	double b = input_image_means[pixel];
	for (int c = 0; c < 3; c++)
	  b -= a_value[c] * guidance_image_means[c][pixel];
	b_values[pixel] = b;
      }
      
      vector<vector<double> > a_means(3);
      for (int c = 0; c < 3; c++)
	calcWindowMeansAndVars(a_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, a_means[c], dummy_vars);
      vector<double> b_means;
      calcWindowMeansAndVars(b_values, IMAGE_WIDTH, IMAGE_HEIGHT, radius * 2 + 1, b_means, dummy_vars);
      
      output_image = Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
	for (int x = 0; x < IMAGE_WIDTH; x++) {
	  int pixel = y * IMAGE_WIDTH + x;
	  double color = b_means[pixel];
	  for (int c = 0; c < 3; c++)
	    color += a_means[c][pixel] * guidance_image_values[c][pixel];
	  output_image.at<uchar>(y, x) = max(min(color * 256, 255.0), 0.0);
	}
      }
    }
  }
}


