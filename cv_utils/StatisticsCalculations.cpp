#include "cv_utils.h"

using namespace std;


namespace cv_utils
{
  vector<double> calcMeanAndSVar(const vector<double> &values)
  {
    double sum = 0, sum2 = 0;
    for (std::vector<double>::const_iterator value_it = values.begin(); value_it != values.end(); value_it++) {
      sum += *value_it;
      sum2 += pow(*value_it, 2);
    }
    double mean = sum / values.size();
    double svar = sqrt(sum2 / values.size() - pow(mean, 2));
    vector<double> mean_and_svar;
    mean_and_svar.push_back(mean);
    mean_and_svar.push_back(svar);
    return mean_and_svar;
  }
  
  void calcMeanAndSVar(const vector<vector<double> > &values, vector<double> &mean, vector<vector<double> > &var)
  {
    assert(values.size() > 0);
    const int NUM_DIMENSIONS = values.begin()->size();
    vector<double> sums(NUM_DIMENSIONS, 0);
    vector<vector<double> > sum2s(NUM_DIMENSIONS, vector<double>(NUM_DIMENSIONS, 0));
    for (std::vector<std::vector<double> >::const_iterator value_it = values.begin(); value_it != values.end(); value_it++) {
      for (std::vector<double>::const_iterator c_it = value_it->begin(); c_it != value_it->end(); c_it++) {
        sums[c_it - value_it->begin()] += *c_it;
	for (std::vector<double>::const_iterator d_it = value_it->begin(); d_it != value_it->end(); d_it++)
	  sum2s[c_it - value_it->begin()][d_it - value_it->begin()] += *c_it * *d_it;
      }
    }
    mean.assign(NUM_DIMENSIONS, 0);
    for (int c = 0; c < NUM_DIMENSIONS; c++)
      mean[c] = sums[c] / values.size();
    var.assign(NUM_DIMENSIONS, vector<double>(NUM_DIMENSIONS));
    for (int c = 0; c < NUM_DIMENSIONS; c++)
      for (int d = 0; d < NUM_DIMENSIONS; d++)
	var[c][d] = sum2s[c][d] / values.size() - mean[c] * mean[d];
  }
  
  vector<vector<int> > findAllCombinations(const vector<int> &candidates, const int num_elements)
  {
    if (num_elements == 0)
      return vector<vector<int> >(1, vector<int>());
    vector<vector<int> > combinations;
    int num_candidates = candidates.size();
    if (num_candidates < num_elements)
      return combinations;
    
    vector<bool> selected_element_mask(num_candidates, false);
    for (int index = num_candidates - num_elements; index < num_candidates; index++)
      selected_element_mask[index] = true;
    while (true) {
      vector<int> combination;
      for (int index = 0; index < num_candidates; index++) {
        if (selected_element_mask[index] == true) {
          combination.push_back(candidates[index]);
        }
      }
      combinations.push_back(combination);
      if (next_permutation(selected_element_mask.begin(), selected_element_mask.end()) == false)
        break;
    }
    return combinations;
    
    for (int configuration = 0; configuration < pow(2, num_candidates); configuration++) {
      vector<bool> selected_element_mask(num_candidates, false);
      int num_selected_elements = 0;
      int configuration_temp = configuration;
      for (int j = 0; j < num_candidates; j++) {
        if (configuration_temp % 2 == 1) {
          selected_element_mask[j] = true;
          num_selected_elements++;
          if (num_selected_elements > num_elements)
            break;
        }
        configuration_temp /= 2;
      }
      if (num_selected_elements != num_elements)
        continue;
      vector<int> combination;
      for (int j = 0; j < num_candidates; j++)
        if (selected_element_mask[j] == true)
          combination.push_back(candidates[j]);
      combinations.push_back(combination);
    }
    return combinations;
  }

  int calcNumDistinctValues(const vector<int> &values)
  {
    vector<int> sorted_values = values;
    sort(sorted_values.begin(), sorted_values.end());
    vector<int>::iterator unique_end = unique(sorted_values.begin(), sorted_values.end());
    return distance(sorted_values.begin(), unique_end);
  }
}
