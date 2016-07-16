#include <vector>
#include <iostream>

namespace cv_utils
{
  template<typename T> class Histogram
    {
    public:
    Histogram(const int NUM_GRAMS, const T MIN_VALUE, const T MAX_VALUE, const std::vector<T> &values = std::vector<T>()) : NUM_GRAMS_(NUM_GRAMS), MIN_VALUE_(MIN_VALUE), MAX_VALUE_(MAX_VALUE), num_values_(values.size()), histos_(std::vector<int>(NUM_GRAMS, 0))
	{
	  //std::cout << MIN_VALUE_ << '\t' << MAX_VALUE_ << std::endl;
	  assert(MIN_VALUE_ < MAX_VALUE_);
	  //std::cout << NUM_GRAMS_ << std::endl;
	  for (typename std::vector<T>::const_iterator value_it = values.begin(); value_it != values.end(); value_it++) {
            if (*value_it >= MIN_VALUE_ && *value_it <= MAX_VALUE) {
	      histos_[calcHistoIndex(*value_it)]++;
	      //std::cout << calcHistoIndex(*value_it) << std::endl;
	    }
	  }
        };
      double getEntropy()
      {
	double entropy = 0;
        for (std::vector<int>::const_iterator histo_it = histos_.begin(); histo_it != histos_.end(); histo_it++)
          entropy += -1.0 * *histo_it / num_values_ * log(1.0 * *histo_it / num_values_);
        return entropy;
      };
      double getProbability(const T &value)
      {
	if (value < MIN_VALUE_ || value > MAX_VALUE_)
          return 0;
        return histos_[calcHistoIndex(value)];
      };
      
    private:
      const int NUM_GRAMS_;
      const T MIN_VALUE_, MAX_VALUE_;
      int num_values_;
      std::vector<int> histos_;
      
      
      double calcHistoIndex(const T &value)
      {
	return std::min(std::max(static_cast<int>(1.0 * (value - MIN_VALUE_) / (MAX_VALUE_ - MIN_VALUE_) * NUM_GRAMS_), 0), NUM_GRAMS_ - 1);
      };
    };
}
