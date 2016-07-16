#ifndef IMAGE_MASK_H__
#define IMAGE_MASK_H__

#include <vector>
#include <opencv2/core/core.hpp>

namespace cv_utils
{
  class ImageMask
  {
  public:
    ImageMask();
    ImageMask(const std::vector<bool> &mask, const int width, const int height);
    ImageMask(const bool value, const int width, const int height);
    ImageMask(const std::vector<int> &pixels, const int width, const int height);
    ImageMask(const cv::Mat &image);
    
    ImageMask &operator = (const ImageMask &image_mask);
    
    //    ImageMask clone();
    
    void setMask(const std::vector<bool> &mask, const int width, const int height);
    
    void resize(const int new_width, const int new_height);
    void resizeByRatio(const double x_ratio, const double y_ratio);
    void resizeWithBias(const int new_width, const int new_height, const bool desired_value);
    
    void dilate(const int num_iterations = 1, const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8);
    void erode(const int num_iterations = 1, const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8);
    
    void smooth(const std::string type, const int window_size, const double sigma = 0);
    
    bool at(const int &pixel) const;
    void set(const int &pixel, const bool value);
    std::vector<int> getPixels() const;
    int getNumPixels() const;
    std::vector<double> getCenter() const;
    
    cv::Mat drawMaskImage(const int num_channels = 1) const;
    cv::Mat drawImageWithMask(const cv::Mat &image, const bool use_mask_color = true, const cv::Vec3b mask_color = cv::Vec3b(255, 255, 255), const bool use_outside_color = false, const cv::Vec3b outside_color = cv::Vec3b(0, 0, 0)) const;
    void readMaskImage(const cv::Mat &mask_image);
    
    std::vector<double> calcDistanceMapOutside(const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8) const;
    std::vector<double> calcDistanceMapInside(const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8) const;
    
    void calcBoundaryDistanceMap(std::vector<int> &boundary_map, std::vector<double> &distance_map, const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8) const;
    
    std::vector<std::vector<int> > findConnectedComponents(const bool USE_PANORAMA = false, const int NEIGHBOR_SYSTEM = 8);
    
    void printMask();
    
    void addPixels(const std::vector<int> &pixels);
    void subtractPixels(const std::vector<int> &pixels);
    
    ImageMask &operator +=(const ImageMask &image_mask);
    ImageMask &operator -=(const ImageMask &image_mask);
    friend ImageMask operator +(const ImageMask &image_mask_1, const ImageMask &image_mask_2);
    friend ImageMask operator -(const ImageMask &image_mask_1, const ImageMask &image_mask_2);
    friend std::ostream & operator <<(std::ostream &out_str, const ImageMask &image_mask);
    friend std::istream & operator >>(std::istream &in_str, ImageMask &image_mask);

    std::vector<int> findMaskWindowPixels(const int pixel, const int WINDOW_SIZE, const int USE_PANORAMA = false) const;
    
  private:
    std::vector<bool> mask_;
    int width_;
    int height_;
  };
}

#endif
