#include "LayerDepthRepresenter.h"
#include <iostream>
#include <fstream>
#include <ctime>

#include <Eigen/Sparse>

#include "utils.h"
#include "TRWSFusion.h"
#include "ProposalDesigner.h"
#include "cv_utils/cv_utils.h"
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <sys/types.h>


using namespace std;
using namespace cv;
using namespace Eigen;

DEFINE_string(result_folder, "", "The folder to save results.");
DEFINE_string(cache_folder, "", "The folder to save intermediate results.");

DEFINE_int32(num_iterations, 18, "The number of iterations.");


LayerDepthRepresenter::LayerDepthRepresenter(const Mat &image, const vector<double> &point_cloud, const RepresenterPenalties &penalties, const DataStatistics &statistics, const Mat &ori_image, const vector<double> &ori_point_cloud, const int num_layers) : image_(image), point_cloud_(point_cloud), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(IMAGE_WIDTH_ * IMAGE_HEIGHT_), PENALTIES_(penalties), STATISTICS_(statistics), ori_image_(ori_image), ori_point_cloud_(ori_point_cloud), num_layers_(num_layers)
{
  
  struct stat sb;
  if (stat(FLAGS_result_folder.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(FLAGS_result_folder.c_str(), 0777);
  
  if (stat(FLAGS_cache_folder.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(FLAGS_cache_folder.c_str(), 0777);
  
  
  ROI_mask_ = vector<bool>(NUM_PIXELS_, true);
  
  camera_parameters_.assign(3, 0);
  cv_utils::estimateCameraParameters(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_);
  cout << "camera parameters: " << camera_parameters_[0] << '\t' << camera_parameters_[1] << '\t' << camera_parameters_[2] << endl;
  
  normals_ = cv_utils::calcNormals(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);  
  
  optimizeLayerRepresentation();
}

LayerDepthRepresenter::~LayerDepthRepresenter()
{
}

void LayerDepthRepresenter::optimizeLayerRepresentation()
{
  srand(time(NULL));
  
  map<int, vector<double> > iteration_statistics_map;
  map<int, string> iteration_proposal_type_map;
  int iteration_start_index = 0;  
  double previous_energy = -1;  
  int previous_energy_iteration = -1;  
  int num_single_surface_expansion_proposals = 0;
  stringstream iteration_info_filename;
  
  iteration_info_filename << FLAGS_cache_folder << "/iteration_info.txt";
  ifstream iteration_info_in_str(iteration_info_filename.str());
  double total_running_time = 0;
  if (iteration_info_in_str) {
    while (true) {
      int iteration;
      string proposal_type;
      vector<double> statistics(3);
      iteration_info_in_str >> iteration >> proposal_type >> statistics[0] >> statistics[1] >> statistics[2];
      if (iteration_info_in_str.eof() == true)
	break;
      iteration_statistics_map[iteration] = statistics;
      iteration_proposal_type_map[iteration] = proposal_type;
      if (proposal_type == "single_surface_expansion_proposal")
	num_single_surface_expansion_proposals++;
      cout << iteration << '\t' << proposal_type << '\t' << statistics[0] << '\t' << statistics[1] << '\t' << statistics[2] << endl;
      total_running_time += statistics[2];
    }
    iteration_info_in_str.close();
  }
  cout << "total running time: " << total_running_time << endl;
    
  for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
    iteration_start_index = iteration_it->first + 1;
    if (previous_energy < 0 || iteration_it->second[0] < previous_energy) {
      previous_energy_iteration = iteration_it->first;
      previous_energy = iteration_it->second[0];
    }
  }
    

  unique_ptr<ProposalDesigner> proposal_designer(new ProposalDesigner(image_, point_cloud_, normals_, camera_parameters_, num_layers_, PENALTIES_, STATISTICS_));
  unique_ptr<TRWSFusion> TRW_solver(new TRWSFusion(image_, point_cloud_, normals_, PENALTIES_, STATISTICS_));
  
  
  if (previous_energy_iteration >= 0) {
    for (int iteration = 0; iteration <= previous_energy_iteration; iteration++) {
      vector<int> solution_labels;
      int solution_num_surfaces;
      map<int, Segment> solution_segments;
    }
    
    vector<int> previous_solution_labels;
    int previous_solution_num_surfaces;
    map<int, Segment> previous_solution_segments;
    
    bool read_success = readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments, previous_energy_iteration) == true;
    
    assert(read_success);
    
    proposal_designer->setCurrentSolution(previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments);
  }
  
  int best_solution_iteration = previous_energy_iteration;
  for (int iteration = iteration_start_index; iteration < FLAGS_num_iterations + num_single_surface_expansion_proposals; iteration++) {
    cout << "proposal: " << iteration << endl;
    
    const clock_t begin_time = clock();
    vector<vector<int> > proposal_labels;
    vector<int> proposal_segmentation;
    int proposal_num_surfaces;
    map<int, Segment> proposal_segments;
    vector<int> proposal_distance_to_boundaries;
    string proposal_type;
    if (proposal_designer->getProposal(iteration, proposal_labels, proposal_num_surfaces, proposal_segments, proposal_type) == false)
      break;

    if (proposal_type == "single_surface_expansion_proposal")
      num_single_surface_expansion_proposals++;
    
    vector<int> previous_solution_indices = proposal_designer->getCurrentSolutionIndices();
    vector<int> current_solution_labels = TRW_solver->fuse(proposal_labels, proposal_num_surfaces, num_layers_, proposal_segments, previous_solution_indices);
    vector<double> energy_info = TRW_solver->getEnergyInfo();
    vector<double> statistics = energy_info;
    statistics.push_back(static_cast<double>(clock() - begin_time) / CLOCKS_PER_SEC);
    iteration_statistics_map[iteration] = statistics;
    iteration_proposal_type_map[iteration] = proposal_type;

    writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, current_solution_labels, proposal_num_surfaces, proposal_segments, iteration, ori_image_, ori_point_cloud_);
    if (energy_info[0] >= previous_energy && previous_energy >= 0) {
      cout << "energy increases" << endl;
      continue;
    }
    previous_energy = energy_info[0];
    
    int current_solution_num_surfaces = proposal_num_surfaces;
    map<int, Segment> current_solution_segments = proposal_segments;
    
    proposal_designer->setCurrentSolution(current_solution_labels, current_solution_num_surfaces, current_solution_segments);
    
    
    ofstream iteration_info_out_str(iteration_info_filename.str());
    for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
      iteration_info_out_str << iteration_it->first << '\t' << iteration_proposal_type_map[iteration_it->first] << '\t' << iteration_it->second[0] << '\t' << iteration_it->second[1] << '\t' << iteration_it->second[2] << endl;
    }
    iteration_info_out_str.close();
    
    best_solution_iteration = iteration;
  }
  
  
  ofstream iteration_info_out_str(iteration_info_filename.str());
  for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
    iteration_info_out_str << iteration_it->first << '\t' << iteration_proposal_type_map[iteration_it->first] << '\t' << iteration_it->second[0] << '\t' << iteration_it->second[1] << '\t' << iteration_it->second[2] << endl;
  }
  iteration_info_out_str.close();
  
  generateLayerImageHTML(iteration_statistics_map, iteration_proposal_type_map);
  
  vector<int> solution_labels;
  int solution_num_surfaces;
  map<int, Segment> solution_segments;
  
  bool read_success = readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, solution_labels, solution_num_surfaces, solution_segments, best_solution_iteration);
  
  assert(read_success);
  
  writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, solution_labels, solution_num_surfaces, solution_segments, 10000, ori_image_, ori_point_cloud_);
  
  writeRenderingInfo(solution_labels, solution_num_surfaces, solution_segments);
  

  return;
}

void writeLayers(const Mat &image, const int image_width, const int image_height, const vector<double> &point_cloud, const vector<double> &camera_parameters, const int num_layers, const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments, const int result_index, const Mat &ori_image, const vector<double> &ori_point_cloud)
{
  const int NUM_PIXELS = image_width * image_height;
  vector<map<int, int> > layer_surface_x_sum(num_layers);
  vector<map<int, int> > layer_surface_y_sum(num_layers);
  vector<map<int, int> > layer_surface_counter(num_layers);
  vector<vector<int> > layer_surface_ids(num_layers, vector<int>(NUM_PIXELS, 0));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    vector<int> layer_labels(num_layers);
    int label_temp = solution[pixel];
    for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--) {
      layer_labels[layer_index] = label_temp % (solution_num_surfaces + 1);
      label_temp /= (solution_num_surfaces + 1);
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      int surface_id = layer_labels[layer_index];
      layer_surface_ids[layer_index][pixel] = surface_id;
      if (surface_id < solution_num_surfaces) {
	layer_surface_x_sum[layer_index][surface_id] += pixel % image_width;
	layer_surface_y_sum[layer_index][surface_id] += pixel / image_width;
	layer_surface_counter[layer_index][surface_id] += 1;
      }
    }
  }
  vector<map<int, int> > layer_surface_centers(num_layers);
  for (int layer_index = 0; layer_index < num_layers; layer_index++)
    for (map<int, int>::const_iterator counter_it = layer_surface_counter[layer_index].begin(); counter_it != layer_surface_counter[layer_index].end(); counter_it++)
      layer_surface_centers[layer_index][counter_it->first] = (layer_surface_y_sum[layer_index][counter_it->first] / counter_it->second) * image_width + (layer_surface_x_sum[layer_index][counter_it->first] / counter_it->second);
  
  vector<Mat> layer_images;
  vector<Mat> layer_mask_images;

  map<int, Vec3b> color_table;
  map<int, Vec3b> segment_color_table;
  vector<bool> visible_mask(ori_image.cols * ori_image.rows, true);
    
  map<int, Vec3b> layer_color_table;
  layer_color_table[2] = Vec3b(0, 0, 255);
  layer_color_table[1] = Vec3b(0, 255, 0);
  layer_color_table[0] = Vec3b(255, 0, 0);
  layer_color_table[3] = Vec3b(255, 0, 255);
  const double BLENDING_ALPHA = 0.5;
    
    
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    
    Mat layer_image = Mat::zeros(ori_image.rows, ori_image.cols, CV_8UC3);
    vector<int> surface_ids = layer_surface_ids[layer_index];
    for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
      int ori_x = ori_pixel % ori_image.cols;
      int ori_y = ori_pixel / ori_image.cols;
      int x = min(static_cast<int>(round(1.0 * ori_x / ori_image.cols * image_width)), image_width - 1);
      int y = min(static_cast<int>(round(1.0 * ori_y / ori_image.rows * image_height)), image_height - 1);
      int pixel = y * image_width + x;
      int surface_id = surface_ids[pixel];
      if (surface_id != solution_num_surfaces) {
	if (visible_mask[ori_pixel] == true) {
	  layer_image.at<Vec3b>(ori_y, ori_x) = ori_image.at<Vec3b>(ori_y, ori_x);
	  visible_mask[ori_pixel] = false;
	} else
	  layer_image.at<Vec3b>(ori_y, ori_x) = Vec3b(0, 0, 0);;
      } else {
	layer_image.at<Vec3b>(ori_y, ori_x) = Vec3b(255, 255, 255);;
      }          
    }
      
    vector<int> line_mask(ori_image.cols * ori_image.rows, -1);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int surface_id = surface_ids[pixel];
      if (surface_id == solution_num_surfaces)
	continue;
	
      int x = pixel % image_width;
      int y = pixel / image_width;
      vector<int> neighbor_pixels;
      if (x > 0)
	neighbor_pixels.push_back(pixel - 1);
      if (x < image_width - 1)
	neighbor_pixels.push_back(pixel + 1);
      if (y > 0)
	neighbor_pixels.push_back(pixel - image_width);
      if (y < image_height - 1)
	neighbor_pixels.push_back(pixel + image_width);
      if (x > 0 && y > 0)
	neighbor_pixels.push_back(pixel - 1 - image_width);
      if (x > 0 && y < image_height - 1)
	neighbor_pixels.push_back(pixel - 1 + image_width);
      if (x < image_width - 1 && y > 0)
	neighbor_pixels.push_back(pixel + 1 - image_width);
      if (x < image_width - 1 && y < image_height - 1)
	neighbor_pixels.push_back(pixel + 1 + image_width);
      bool on_boundary = false;
      int neighbor_segment_id = -1;
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	if (surface_ids[*neighbor_pixel_it] != surface_id) {
	  on_boundary = true;
	  neighbor_segment_id = surface_ids[*neighbor_pixel_it];
	  break;
	}
      }
	
      if (on_boundary == false)
	continue;
	
      if (color_table.count(surface_id) == 0)
	color_table[surface_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	
      int ori_x = min(static_cast<int>(round(1.0 * x / image_width * ori_image.cols)), ori_image.cols - 1);
      int ori_y = min(static_cast<int>(round(1.0 * y / image_height * ori_image.rows)), ori_image.rows - 1);
      line_mask[ori_y * ori_image.cols + ori_x] = surface_id;
    }
    const int LINE_WIDTH = 4;
    for (int iteration = 0; iteration < LINE_WIDTH; iteration++) {
      vector<int> new_line_mask = line_mask;
      for (int pixel = 0; pixel < ori_image.cols * ori_image.rows; pixel++) {
	if (line_mask[pixel] == -1)
	  continue;
	int x = pixel % ori_image.cols;
	int y = pixel / ori_image.cols;
	vector<int> neighbor_pixels;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < ori_image.cols - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - ori_image.cols);
	if (y < ori_image.rows - 1)
	  neighbor_pixels.push_back(pixel + ori_image.cols);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	  if (new_line_mask[*neighbor_pixel_it] == -1)
	    new_line_mask[*neighbor_pixel_it] = line_mask[pixel];
      }
      line_mask = new_line_mask;
    }
      
    for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
      if (line_mask[ori_pixel] == -1)
	continue;
      int ori_x = ori_pixel % ori_image.cols;
      int ori_y = ori_pixel / ori_image.cols;
      layer_image.at<Vec3b>(ori_y, ori_x) = color_table[line_mask[ori_pixel]];
    }
      
      
    Mat layer_mask_image = ori_image.clone();
    for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
      int ori_x = ori_pixel % ori_image.cols;
      int ori_y = ori_pixel / ori_image.cols;
      int x = min(static_cast<int>(round(1.0 * ori_x / ori_image.cols * image_width)), image_width - 1);
      int y = min(static_cast<int>(round(1.0 * ori_y / ori_image.rows * image_height)), image_height - 1);
      int pixel = y * image_width + x;
      int surface_id = surface_ids[pixel];
      if (surface_id != solution_num_surfaces) {
	Vec3b image_color = layer_mask_image.at<Vec3b>(ori_y, ori_x);
	Vec3b layer_color = layer_color_table[layer_index];
	if (segment_color_table.count(surface_id) == 0)
	  segment_color_table[surface_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	Vec3b segment_color = segment_color_table[surface_id];
	Vec3b blended_color;
	for (int c = 0; c < 3; c++)
	  blended_color[c] = min(image_color[c] * BLENDING_ALPHA + segment_color[c] * (1 - BLENDING_ALPHA), 255.0);
	layer_mask_image.at<Vec3b>(ori_y, ori_x) = blended_color;
      }
      if (line_mask[ori_pixel] != -1)
	layer_mask_image.at<Vec3b>(ori_y, ori_x) = Vec3b(0, 0, 0);
    }
      
    stringstream layer_mask_image_filename;
    layer_mask_image_filename << FLAGS_result_folder << "/layer_mask_image_" << layer_index << ".bmp";
    imwrite(layer_mask_image_filename.str().c_str(), layer_mask_image);
      
    layer_images.push_back(layer_image);
    layer_mask_images.push_back(layer_mask_image);
  }
    

  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    vector<int> surface_ids = layer_surface_ids[layer_index]; 
    Mat layer_image_raw = Mat::zeros(image_height, image_width, CV_8UC1);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;
      int surface_id = surface_ids[pixel];
      layer_image_raw.at<uchar>(y, x) = surface_id;
    }

    stringstream layer_image_raw_filename;
    
    layer_image_raw_filename << FLAGS_cache_folder << "/layer_image_raw_" << result_index << "_" << layer_index << ".bmp";
    imwrite(layer_image_raw_filename.str().c_str(), layer_image_raw);
  }


  const int IMAGE_PADDING = 0;
  const int BORDER_WIDTH = 16;
  Mat multi_layer_image(ori_image.rows + BORDER_WIDTH * 2, (ori_image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers), CV_8UC3);
  multi_layer_image.setTo(Scalar(255, 255, 255));
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat layer_image_with_border = Mat::zeros(ori_image.rows + BORDER_WIDTH * 2, ori_image.cols + BORDER_WIDTH * 2, CV_8UC3);
    Mat layer_image_region(layer_image_with_border, Rect(BORDER_WIDTH, BORDER_WIDTH, ori_image.cols, ori_image.rows));
    layer_mask_images[layer_index].copyTo(layer_image_region);
    
    Mat region(multi_layer_image, Rect((layer_image_with_border.cols + IMAGE_PADDING) * layer_index, 0, layer_image_with_border.cols, layer_image_with_border.rows));
    layer_image_with_border.copyTo(region);
  }
  stringstream multi_layer_image_filename;
  multi_layer_image_filename << FLAGS_result_folder << "/multi_layer_image_" << result_index << ".bmp";
  imwrite(multi_layer_image_filename.str().c_str(), multi_layer_image);
  
  
  Mat input_multi_layer_image(ori_image.rows + BORDER_WIDTH * 2, (ori_image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers + 2), CV_8UC3);
  input_multi_layer_image.setTo(Scalar(0, 0, 0));
  
  Mat image_region(input_multi_layer_image, Rect(BORDER_WIDTH, BORDER_WIDTH, ori_image.cols, ori_image.rows));
  ori_image.copyTo(image_region);
  Mat disp_image_region(input_multi_layer_image, Rect(BORDER_WIDTH + (ori_image.cols + BORDER_WIDTH * 2), BORDER_WIDTH, ori_image.cols, ori_image.rows));
  //imwrite("Test/ori_disp_image.bmp", drawDispImage(ori_point_cloud, ori_image.cols, ori_image.rows));
  
  Mat disp_image = imread(FLAGS_result_folder + "/ori_disp_image.bmp");
  disp_image.copyTo(disp_image_region);
  Mat multi_layer_image_region(input_multi_layer_image, Rect((ori_image.cols + BORDER_WIDTH * 2) * 2, 0, multi_layer_image.cols, multi_layer_image.rows));
  multi_layer_image.copyTo(multi_layer_image_region);
  
  stringstream input_multi_layer_image_filename;
  input_multi_layer_image_filename << FLAGS_result_folder << "/input_multi_layer_image.bmp";
  imwrite(input_multi_layer_image_filename.str().c_str(), input_multi_layer_image);
  

  stringstream segments_filename;
  segments_filename << FLAGS_cache_folder << "/segments_" << result_index << ".txt";
  ofstream segments_out_str(segments_filename.str());
  segments_out_str << solution_num_surfaces << endl;
  for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
    segments_out_str << surface_it->first << endl;
    segments_out_str << surface_it->second << endl;
  }
  segments_out_str.close();

  stringstream segment_GMMs_filename;
  segment_GMMs_filename << FLAGS_cache_folder << "/segment_GMMs_" << result_index << ".xml";
  FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::WRITE);
  for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
    stringstream segment_name;
    segment_name << "Segment" << surface_it->first;
    segment_GMMs_fs << segment_name.str() << "{";
    surface_it->second.getGMM()->write(segment_GMMs_fs);
    segment_GMMs_fs << "}";
  }
  segment_GMMs_fs.release();

  
  bool write_ply_files = false;
  if (write_ply_files == true) {
    vector<vector<int> > layer_visible_pixel_segment_map(num_layers, vector<int>(NUM_PIXELS, -1));
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
	int surface_id = layer_surface_ids[layer_index][pixel];
	if (surface_id < solution_num_surfaces) {
	  layer_visible_pixel_segment_map[layer_index][pixel] = surface_id;
	  break;
	}
      }
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      int num_points = 0;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        if (surface_ids[pixel] < solution_num_surfaces)
    	  num_points++;

      map<int, vector<int> > segment_fitted_pixels;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
        vector<int> fitted_pixels = segment_it->second.getSegmentPixels();
	vector<int> new_fitted_pixels;
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  if (layer_visible_pixel_segment_map[layer_index][*pixel_it] == segment_it->first)
	    new_fitted_pixels.push_back(*pixel_it);
        segment_fitted_pixels[segment_it->first] = new_fitted_pixels;
      }	

      int num_fitted_pixels = 0;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++)
	num_fitted_pixels += segment_it->second.size();
      
      vector<double> point_cloud_range(6);
      for (int c = 0; c < 3; c++) {
	point_cloud_range[c * 2] = 1000000;
	point_cloud_range[c * 2 + 1] = -1000000;
      }
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
	for (int c = 0; c < 3; c++) {
	  if (point[c] < point_cloud_range[c * 2])
	    point_cloud_range[c * 2] = point[c];
          if (point[c] > point_cloud_range[c * 2 + 1])
            point_cloud_range[c * 2 + 1] = point[c];
	}
      }
      
      map<int, vector<vector<double> > > segment_plane_vertices;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++)
	segment_plane_vertices[segment_it->first] = vector<vector<double> >(4, vector<double>(3));
      stringstream layer_ply_filename;
      layer_ply_filename << FLAGS_result_folder << "/layer_ply_" << result_index << "_" << layer_index << ".ply";
      ofstream out_str(layer_ply_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_points << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "end_header" << endl;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int surface_id = surface_ids[pixel];
        if (surface_id == solution_num_surfaces)
    	  continue;
    	double depth = solution_segments.at(surface_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
	double x = pixel % image_width - camera_parameters[1];
	double y = pixel / image_width - camera_parameters[2];
	double X = -x / camera_parameters[0] * depth;
	double Y = -y / camera_parameters[0] * depth;
	out_str << X << ' ' << Y << ' ' << depth << endl;
      }
      out_str.close();

      stringstream layer_segment_fitting_filename;
      layer_segment_fitting_filename << FLAGS_result_folder << "/layer_segment_fitting_" << result_index << "_" << layer_index << ".ply";
      out_str.open(layer_segment_fitting_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_fitted_pixels + solution_segments.size() * 4 << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      out_str << "property uchar alpha" << endl;
      out_str << "element face " << segment_fitted_pixels.size() * 2 << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++) {
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        color_table[segment_it->first] = r * 256 * 256 + g * 256 + b;
        for (vector<int>::const_iterator pixel_it = segment_it->second.begin(); pixel_it != segment_it->second.end(); pixel_it++) {
          int pixel = *pixel_it;
	  
	  double X = -point_cloud[pixel * 3 + 0];
	  double Y = -point_cloud[pixel * 3 + 1];
	  double Z = point_cloud[pixel * 3 + 2];
          out_str << X << ' ' << Y << ' ' << Z << ' ' << r << ' ' << g << ' ' << b << " 255" << endl;
        }
      }
      for (map<int, vector<vector<double> > >::const_iterator segment_it = segment_plane_vertices.begin(); segment_it != segment_plane_vertices.end(); segment_it++) {
        int color = color_table[segment_it->first];
        int r = color / (256 * 256);
        int g = color / 256 % 256;
        int b = color % 256;
        for (vector<vector<double> >::const_iterator vertex_it = segment_it->second.begin(); vertex_it != segment_it->second.end(); vertex_it++)
          out_str << vertex_it->at(0) << ' ' << vertex_it->at(1) << ' ' << vertex_it->at(2) << ' ' << r << ' ' << g << ' ' << b << " 10" << endl;
      }
      for (int i = 0; i < segment_fitted_pixels.size(); i++) {
	out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 1 << ' ' << num_fitted_pixels + i * 4 + 2 << endl;
	out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 2 << ' ' << num_fitted_pixels + i * 4 + 3 << endl;
      }
      out_str.close();
    }
  }


  bool write_mesh_ply_files = true;
  if (write_mesh_ply_files == true) {
    map<int, int> layer_color_table;
    layer_color_table[0] = 255 * 256 * 256;
    layer_color_table[1] = 255 * 256;
    layer_color_table[2] = 255;
    layer_color_table[3] = 255 * 256 + 255;
    vector<vector<int> > layer_triangle_pixels_vec(num_layers);
    int num_non_empty_pixels = 0;
    vector<vector<int> > layer_pixel_index_map(num_layers);
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      vector<int> triangle_pixels_vec;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int x = pixel % image_width;
	int y = pixel / image_width;
	if (x == image_width - 1 || y == image_height - 1)
	  continue;
	vector<int> cell_pixels;
	if (surface_ids[pixel] < solution_num_surfaces)
	  cell_pixels.push_back(pixel);
	if (surface_ids[pixel + 1] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1);
        if (surface_ids[pixel + 1 + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1 + image_width);
        if (surface_ids[pixel + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + image_width);
	
	if (cell_pixels.size() == 3)
	  triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.end());
        if (cell_pixels.size() == 4) {
	  triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.begin() + 3);
	  triangle_pixels_vec.push_back(cell_pixels[0]);
	  triangle_pixels_vec.push_back(cell_pixels[2]);
	  triangle_pixels_vec.push_back(cell_pixels[3]);
	}
      }
      layer_triangle_pixels_vec[layer_index] = triangle_pixels_vec;
      
      vector<int> pixel_index_map(NUM_PIXELS, -1);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
	if (surface_ids[pixel] < solution_num_surfaces)
	  pixel_index_map[pixel] = num_non_empty_pixels++;
      layer_pixel_index_map[layer_index] = pixel_index_map;
    }

    {
      stringstream layer_segment_mesh_filename;
      layer_segment_mesh_filename << FLAGS_result_folder << "/layered_mesh.ply";
      ofstream out_str(layer_segment_mesh_filename.str());

      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_non_empty_pixels << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      int num_triangles = 0;
      for (int layer_index = 0; layer_index < num_layers; layer_index++)
	num_triangles += layer_triangle_pixels_vec[layer_index].size() / 3;
      out_str << "element face " << num_triangles << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  int segment_id = layer_surface_ids[layer_index][pixel];
	  if (segment_id == solution_num_surfaces)
	    continue;
	  
	  int x = pixel % image_width;
	  int y = pixel / image_width;
	  vector<int> neighbor_pixels;
	  if (x > 0)
	    neighbor_pixels.push_back(pixel - 1);
	  if (x < image_width - 1)
	    neighbor_pixels.push_back(pixel + 1);
	  if (y > 0)
	    neighbor_pixels.push_back(pixel - image_width);
	  if (y < image_height - 1)
	    neighbor_pixels.push_back(pixel + image_width);
	  if (x > 0 && y > 0)
	    neighbor_pixels.push_back(pixel - 1 - image_width);
	  if (x > 0 && y < image_height - 1)
	    neighbor_pixels.push_back(pixel - 1 + image_width);
	  if (x < image_width - 1 && y > 0)
	    neighbor_pixels.push_back(pixel + 1 - image_width);
	  if (x < image_width - 1 && y < image_height - 1)
	    neighbor_pixels.push_back(pixel + 1 + image_width);
	  bool on_boundary = false;
	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	    if (layer_surface_ids[layer_index][*neighbor_pixel_it] != segment_id) {
	      on_boundary = true;
	      break;
	    }
	  }
	
	  double depth = solution_segments.at(segment_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
	  double u = pixel % image_width - camera_parameters[1];
	  double v = pixel / image_width - camera_parameters[2];
	  double X = -u / camera_parameters[0] * depth;
	  double Y = -v / camera_parameters[0] * depth;
	  int color = layer_color_table[layer_index];
	  out_str << X << ' ' << Y << ' ' << depth << ' ' << color / (256 * 256) << ' ' << color / 256 % 256 << ' ' << color % 256 << endl;
        }
      }
    
      for (int layer_index = 0; layer_index < num_layers; layer_index++) { 
	for (int triangle_index = 0; triangle_index < layer_triangle_pixels_vec[layer_index].size() / 3; triangle_index++) {
	  out_str << "3";
	  for (int c = 0; c < 3; c++) {
	    int pixel = layer_triangle_pixels_vec[layer_index][triangle_index * 3 + c];
	    int pixel_index = layer_pixel_index_map[layer_index][pixel];

	    out_str << ' ' << pixel_index;
	  }
	  out_str << endl;
	}
      }
      out_str.close();
    }
  }
}

void LayerDepthRepresenter::writeRenderingInfo(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments)
{
  const int ORI_IMAGE_WIDTH = ori_image_.cols;
  const int ORI_IMAGE_HEIGHT = ori_image_.rows;
  const int ORI_NUM_PIXELS = ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT;
  
  
  vector<int> new_solution_labels;
  int new_solution_num_surfaces;
  map<int, Segment> new_solution_segments;
  upsampleSolution(solution_labels, solution_num_surfaces, solution_segments, new_solution_labels, new_solution_num_surfaces, new_solution_segments);
  
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    
    stringstream layer_depth_values_filename;
    layer_depth_values_filename << FLAGS_result_folder << "/depth_values_" << layer_index;
    vector<double> depths((ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1), 0);
    vector<int> counter((ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1), 0);
    for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
      int surface_id = new_solution_labels[pixel] / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      if (surface_id < new_solution_num_surfaces) {
	vector<int> corner_pixels(4);
	corner_pixels[0] = (pixel / ORI_IMAGE_WIDTH) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH);
	corner_pixels[1] = (pixel / ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH);
	corner_pixels[2] = (pixel / ORI_IMAGE_WIDTH) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH + 1);
	corner_pixels[3] = (pixel / ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH + 1);
	for (vector<int>::const_iterator corner_pixel_it = corner_pixels.begin(); corner_pixel_it != corner_pixels.end(); corner_pixel_it++) {
	  double depth = solution_segments.at(surface_id).getDepth(1.0 * (*corner_pixel_it % (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_WIDTH + 1), 1.0 * (*corner_pixel_it / (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_HEIGHT + 1));
          if (depth < 10) {
	    depths[*corner_pixel_it] += depth;
	    counter[*corner_pixel_it]++;
	  }
	}
      }
    }
    for (int pixel = 0; pixel < (ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1); pixel++) {
      if (counter[pixel] == 0)
	depths[pixel] = -1;
      else
	depths[pixel] /= counter[pixel];
    }
    
    ofstream depth_values_out_str(layer_depth_values_filename.str());
    depth_values_out_str << ORI_IMAGE_WIDTH + 1 << ' ' << ORI_IMAGE_HEIGHT + 1 << endl;
    for (int pixel = 0; pixel < depths.size(); pixel++) {
      double depth = depths[pixel];
      depth_values_out_str << depth << endl;
    }
    depth_values_out_str.close();
  }
  
  vector<vector<int> > layer_surface_ids(num_layers_, vector<int>(ORI_NUM_PIXELS, 0));
  vector<vector<int> > layer_visible_pixels(num_layers_);
  for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++) {
    int label = new_solution_labels[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      int surface_id = label / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      layer_surface_ids[layer_index][pixel] = surface_id;
      if (is_visible && surface_id < new_solution_num_surfaces) {
	layer_visible_pixels[layer_index].push_back(pixel);
	is_visible = false;
      }
    }
  }
  
  vector<vector<int> > segment_pixels_vec(new_solution_num_surfaces);
  vector<vector<int> > hole_pixels_vec(new_solution_num_surfaces);
  for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++) {
    
    int visible_layer_index = -1;
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      int surface_id = layer_surface_ids[layer_index][pixel];
      if (surface_id == new_solution_num_surfaces)
        continue;
      segment_pixels_vec[surface_id].push_back(pixel);
      if (visible_layer_index == -1)
        visible_layer_index = layer_index;
      else
	hole_pixels_vec[surface_id].push_back(pixel);
    }
  }
  
  Mat blurred_image;
  GaussianBlur(ori_image_, blurred_image, cv::Size(3, 3), 0, 0);
  Mat blurred_hsv_image;
  blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);
  
  const int NUM_EROSION_ITERATIONS = 2;
  const double COLOR_LIKELIHOOD_THRESHOLD = 1;
  for (int segment_id = 0; segment_id < new_solution_num_surfaces; segment_id++) {
    if (hole_pixels_vec[segment_id].size() == 0)
      continue;
    vector<bool> known_region_mask(ORI_NUM_PIXELS, false);
    for (vector<int>::const_iterator pixel_it = segment_pixels_vec[segment_id].begin(); pixel_it != segment_pixels_vec[segment_id].end(); pixel_it++)
      known_region_mask[*pixel_it] = true;
    for (vector<int>::const_iterator pixel_it = hole_pixels_vec[segment_id].begin(); pixel_it != hole_pixels_vec[segment_id].end(); pixel_it++)
      known_region_mask[*pixel_it] = false;    
    
    for (int iteration = 0; iteration < NUM_EROSION_ITERATIONS; iteration++) {
      vector<int> known_region_pixels;
      for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++)
        if (known_region_mask[pixel] == true)
          known_region_pixels.push_back(pixel);
      
      set<int> new_hole_pixel_indices;
      for (vector<int>::const_iterator pixel_it = known_region_pixels.begin(); pixel_it != known_region_pixels.end(); pixel_it++) {
        
        int pixel = *pixel_it;
	vector<int> neighbor_pixels;
	int x = pixel % ORI_IMAGE_WIDTH;
	int y = pixel / ORI_IMAGE_WIDTH;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < ORI_IMAGE_WIDTH - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - ORI_IMAGE_WIDTH);
	if (y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + ORI_IMAGE_WIDTH);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - ORI_IMAGE_WIDTH);
	if (x > 0 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel - 1 + ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + 1 + ORI_IMAGE_WIDTH);
	bool on_boundary = false;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (known_region_mask[*neighbor_pixel_it] == false) {
	    on_boundary = true;
	    break;
	  }
	}
	if (on_boundary == false)
	  continue;
	new_hole_pixel_indices.insert(pixel_it - known_region_pixels.begin());
      }
      if (new_hole_pixel_indices.size() == 0 || new_hole_pixel_indices.size() == segment_pixels_vec[segment_id].size())
	break;
      for (set<int>::const_iterator index_it = new_hole_pixel_indices.begin(); index_it != new_hole_pixel_indices.end(); index_it++) {
	hole_pixels_vec[segment_id].push_back(known_region_pixels[*index_it]);
	known_region_mask[known_region_pixels[*index_it]] = false;
      }
    }
  }
  
  
  map<int, Mat> completed_images;
  for (int segment_id = 0; segment_id < new_solution_num_surfaces; segment_id++) {
    stringstream completed_image_filename;
    completed_image_filename << FLAGS_cache_folder << "/completed_image_" << segment_id << ".bmp";
    if (imread(completed_image_filename.str()).empty()) {
      if (segment_pixels_vec[segment_id].size() == 0)
	continue;
      cout << "inpaint segment: " << segment_id << endl;
      Mat mask_image = Mat::ones(ori_image_.rows, ori_image_.cols, CV_8UC1) * 255;
      for (vector<int>::const_iterator pixel_it = segment_pixels_vec[segment_id].begin(); pixel_it != segment_pixels_vec[segment_id].end(); pixel_it++)
        mask_image.at<uchar>(*pixel_it / ori_image_.cols, *pixel_it % ori_image_.cols) = 128;
      for (vector<int>::const_iterator pixel_it = hole_pixels_vec[segment_id].begin(); pixel_it != hole_pixels_vec[segment_id].end(); pixel_it++)
	mask_image.at<uchar>(*pixel_it / ori_image_.cols, *pixel_it % ori_image_.cols) = 0;
      
      
      // {
      // 	stringstream mask_image_filename;
      // 	mask_image_filename << "Test/image_for_completion_" << segment_id << ".bmp";
      // 	Mat image_for_completion = ori_image_.clone();
      // 	for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
      // 	  int x = pixel % ORI_IMAGE_WIDTH;
      //     int y = pixel / ORI_IMAGE_WIDTH;
      // 	  if (mask_image.at<uchar>(y, x) > 200)
      // 	    image_for_completion.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
      //     if (mask_image.at<uchar>(y, x) < 100)
      // 	    image_for_completion.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
      // 	}
      // 	imwrite(mask_image_filename.str(), image_for_completion);
      // }
      
      Mat completed_image;
      imwrite(completed_image_filename.str(), completed_image);
      completed_images[segment_id] = completed_image.clone();
      
      for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++)
	if (mask_image.at<uchar>(pixel / ori_image_.cols, pixel % ori_image_.cols) > 200 && mask_image.at<uchar>(pixel / ori_image_.cols, pixel % ori_image_.cols) < 230)
	  completed_image.at<uchar>(pixel / ori_image_.cols, pixel % ori_image_.cols) = ori_image_.at<uchar>(pixel / ori_image_.cols, pixel % ori_image_.cols);
      
    } else
      completed_images[segment_id] = imread(completed_image_filename.str());
  }
  
  vector<Mat> texture_images(num_layers_);
  vector<Mat> static_texture_images(num_layers_);
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    Mat texture_image = Mat::zeros(ori_image_.size(), CV_8UC3);
    vector<int> surface_ids = layer_surface_ids[layer_index];
    for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
      int x = pixel % ORI_IMAGE_WIDTH;
      int y = pixel / ORI_IMAGE_WIDTH;
      int surface_id = surface_ids[pixel];
      if (surface_id < new_solution_num_surfaces)
	texture_image.at<Vec3b>(y, x) = completed_images[surface_id].at<Vec3b>(y, x);
      else {
	
	vector<int> neighbor_pixels;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < ORI_IMAGE_WIDTH - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - ORI_IMAGE_WIDTH);
	if (y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + ORI_IMAGE_WIDTH);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - ORI_IMAGE_WIDTH);
	if (x > 0 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel - 1 + ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + 1 + ORI_IMAGE_WIDTH);
	int num_valid_colors = 0;
	double b_sum = 0;
	double g_sum = 0;
	double r_sum = 0;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (surface_ids[*neighbor_pixel_it] != new_solution_num_surfaces) {
	    Vec3b color = completed_images[surface_ids[*neighbor_pixel_it]].at<Vec3b>(*neighbor_pixel_it / ORI_IMAGE_WIDTH, *neighbor_pixel_it % ORI_IMAGE_WIDTH);
	    b_sum += color[0];
	    g_sum += color[1];
	    r_sum += color[2];
	    num_valid_colors++;
	  }
	}
	if (num_valid_colors == 0)
	  texture_image.at<Vec3b>(pixel / ORI_IMAGE_WIDTH, pixel % ORI_IMAGE_WIDTH) = Vec3b(255, 0, 0);
	else
	  texture_image.at<Vec3b>(pixel / ORI_IMAGE_WIDTH, pixel % ORI_IMAGE_WIDTH) = Vec3b(b_sum / num_valid_colors, g_sum / num_valid_colors, r_sum / num_valid_colors);
      }
    }
    
    stringstream texture_image_filename;
    texture_image_filename << FLAGS_result_folder << "/texture_image_" << layer_index << ".bmp";
    imwrite(texture_image_filename.str().c_str(), texture_image);
    texture_images[layer_index] = texture_image.clone();
    
    vector<int> visible_pixels = layer_visible_pixels[layer_index];
    for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++) {
      texture_image.at<Vec3b>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH) = ori_image_.at<Vec3b>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH);
    }
    stringstream static_texture_image_filename;
    static_texture_image_filename << FLAGS_result_folder << "/static_texture_image_" << layer_index << ".bmp";
    imwrite(static_texture_image_filename.str().c_str(), texture_image);
    
    static_texture_images[layer_index] = texture_image.clone();
  }
  
  
  stringstream rendering_info_filename;
  rendering_info_filename << FLAGS_result_folder << "/rendering_info";
  ofstream rendering_info_out_str(rendering_info_filename.str());
  rendering_info_out_str << 512 << endl;
  rendering_info_out_str << ori_image_.cols << '\t' << ori_image_.rows << endl;
  rendering_info_out_str << num_layers_ << endl;
  rendering_info_out_str.close();
}

bool readLayers(const int image_width, const int image_height, const vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int num_layers, vector<int> &solution, int &solution_num_surfaces, map<int, Segment> &solution_segments, const int result_index)
{
  const int NUM_PIXELS = image_width * image_height;
  stringstream segments_filename;
  segments_filename << FLAGS_cache_folder << "/segments_" << result_index << ".txt";
  ifstream segments_in_str(segments_filename.str());
  if (!segments_in_str)
    return false;
  
  segments_in_str >> solution_num_surfaces;
  for (int i = 0; i < solution_num_surfaces; i++) {
    int segment_id;
    segments_in_str >> segment_id;
    assert(segment_id == i);
    Segment segment(image_width, image_height, camera_parameters, penalties, statistics);
    segments_in_str >> segment;
    solution_segments[i] = segment;
  }
  segments_in_str.close();
  if (solution_num_surfaces == 0)
    return false;

  stringstream segment_GMMs_filename;
  segment_GMMs_filename << FLAGS_cache_folder << "/segment_GMMs_" << result_index << ".xml";
  FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::READ);
  for (map<int, Segment>::iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
    stringstream segment_name;
    segment_name << "Segment" << surface_it->first;
    FileNode segment_GMM_file_node = segment_GMMs_fs[segment_name.str()];
    surface_it->second.setGMM(segment_GMM_file_node);
  }
  segment_GMMs_fs.release();

  solution = vector<int>(NUM_PIXELS, 0);
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    stringstream layer_image_filename;
    layer_image_filename << FLAGS_cache_folder << "/layer_image_raw_" << result_index << "_" << layer_index << ".bmp";
    Mat layer_image = imread(layer_image_filename.str().c_str(), 0);
    if (layer_image.empty())
      return false;
    
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;

      int surface_id = layer_image.at<uchar>(y, x);
      solution[pixel] += surface_id * pow(solution_num_surfaces + 1, num_layers - 1 - layer_index);
    }
  }
  return true;
}


void LayerDepthRepresenter::generateLayerImageHTML(const map<int, vector<double> > &iteration_statistics_map, const map<int, string> &iteration_proposal_type_map)
{
  stringstream html_filename;
  html_filename << FLAGS_result_folder << "/layer_images.html";
  ofstream html_out_str(html_filename.str());
  html_out_str << "<!DOCTYPE html><html><head></head><body>" << endl;
  double previous_energy = -1;
  for (map<int, string>::const_iterator iteration_it = iteration_proposal_type_map.begin(); iteration_it != iteration_proposal_type_map.end(); iteration_it++) {
    html_out_str << "<h3>iteration " << iteration_it->first << ": " << iteration_it->second << "</h3>" << endl;
    double energy = iteration_statistics_map.at(iteration_it->first)[0];
    if (previous_energy < 0 || energy < previous_energy) {
      stringstream image_filename;
      image_filename << "multi_layer_image_" << iteration_it->first << ".bmp";
      html_out_str << "<img src=\"" << image_filename.str() << "\" alt=\"" << image_filename.str() << "\" width=\"100%\" height=\"100%\">" << endl;
      previous_energy = energy;
    } else
      html_out_str << "<p>Energy increases.</p>";
  }
  html_out_str << "</body></html>";
  html_out_str.close();
}

void LayerDepthRepresenter::upsampleSolution(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments, vector<int> &upsampled_solution_labels, int &upsampled_solution_num_surfaces, map<int, Segment> &upsampled_solution_segments)
{
  
  const int ORI_IMAGE_WIDTH = ori_image_.cols;
  const int ORI_IMAGE_HEIGHT = ori_image_.rows;
  const int ORI_NUM_PIXELS = ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT;
  const int NUM_DILATION_ITERATIONS = 0;

  
  vector<double> ori_camera_parameters(3);
  cv_utils::estimateCameraParameters(ori_point_cloud_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_camera_parameters);
  
  
  map<int, map<int, vector<int> > > segment_layer_visible_pixels;
  vector<int> visible_layer_indices(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int solution_label = solution_labels[pixel];
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
      if (surface_id < solution_num_surfaces) {
        segment_layer_visible_pixels[surface_id][layer_index].push_back(pixel);
	visible_layer_indices[pixel] = layer_index;
        break;
      }
    }
  }
  vector<vector<bool> > layer_confident_pixel_mask(num_layers_, vector<bool>(NUM_PIXELS_, true));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> neighbor_pixels;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    if (x > 0)
      neighbor_pixels.push_back(pixel - 1);
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y > 0)
      neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y > 0)
      neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y > 0)
      neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (visible_layer_indices[pixel] < visible_layer_indices[*neighbor_pixel_it]) {
	layer_confident_pixel_mask[visible_layer_indices[pixel]][pixel] = false;
	layer_confident_pixel_mask[visible_layer_indices[pixel]][*neighbor_pixel_it] = false;
	break;
      }
    }
  }
  const int UNCONFIDENT_BOUNDARY_WIDTH = 2;
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    vector<bool> confident_pixel_mask = layer_confident_pixel_mask[layer_index];
    for (int i = 1; i < UNCONFIDENT_BOUNDARY_WIDTH; i++) {
      vector<bool> new_confident_pixel_mask = confident_pixel_mask;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	if (confident_pixel_mask[pixel] == true)
	  continue;
	vector<int> neighbor_pixels;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	  new_confident_pixel_mask[*neighbor_pixel_it] = false;
      }
      confident_pixel_mask = new_confident_pixel_mask;
    }
    layer_confident_pixel_mask[layer_index] = confident_pixel_mask;
  }
  
  
  vector<vector<double> > distance_map(ORI_NUM_PIXELS, vector<double>(9, 1000000));
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    int x = ori_pixel % ORI_IMAGE_WIDTH;
    int y = ori_pixel / ORI_IMAGE_WIDTH;
    for (int delta_x = -1; delta_x <= 1; delta_x++) {
      for (int delta_y = -1; delta_y <= 1; delta_y++) {
	if (x + delta_x >= 0 && x + delta_x < ORI_IMAGE_WIDTH && y + delta_y >= 0 && y + delta_y < ORI_IMAGE_HEIGHT) {
	  Vec3b color_1 = ori_image_.at<Vec3b>(y, x);
	  Vec3b color_2 = ori_image_.at<Vec3b>(y + delta_y, x + delta_x);
	  double distance = 0;
	  for (int c = 0; c < 3; c++)
	    distance += pow(color_1[c] - color_2[c], 2);
	  distance = sqrt(distance / 3);
	  distance_map[ori_pixel][(delta_y + 1) * 3 + (delta_x + 1)] = distance;
	}
      }
    }
  }
  const double DISTANCE_2D_WEIGHT = 0 * IMAGE_WIDTH_ / ORI_IMAGE_WIDTH;
  
  vector<int> solution_labels_high_res(ORI_NUM_PIXELS, 0);
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    double x = 1.0 * (ori_pixel % ORI_IMAGE_WIDTH) / ORI_IMAGE_WIDTH * IMAGE_WIDTH_;
    double y = 1.0 * (ori_pixel / ORI_IMAGE_WIDTH) / ORI_IMAGE_HEIGHT * IMAGE_HEIGHT_;
    vector<int> xs;
    xs.push_back(max(static_cast<int>(x - 1), 0));
    xs.push_back(min(max(static_cast<int>(x), 0), IMAGE_WIDTH_ - 1));
    xs.push_back(min(static_cast<int>(x) + 1, IMAGE_WIDTH_ - 1));
    xs.push_back(min(static_cast<int>(x) + 2, IMAGE_WIDTH_ - 1));
    vector<int> ys;
    ys.push_back(max(static_cast<int>(y - 1), 0));
    ys.push_back(min(max(static_cast<int>(y), 0), IMAGE_HEIGHT_ - 1));
    ys.push_back(min(static_cast<int>(y) + 1, IMAGE_HEIGHT_ - 1));
    ys.push_back(min(static_cast<int>(y) + 2, IMAGE_HEIGHT_ - 1));
    vector<int> vertex_pixels;
    for (vector<int>::const_iterator y_it = ys.begin(); y_it != ys.end(); y_it++)
      for (vector<int>::const_iterator x_it = xs.begin(); x_it != xs.end(); x_it++)
	vertex_pixels.push_back(*y_it * IMAGE_WIDTH_ + *x_it);
    
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      vector<int> segment_indices;
      for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++)
        segment_indices.push_back(solution_labels[*pixel_it] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1));
      
      map<int, map<int, int> > surface_occluding_relations;
      for (vector<int>::const_iterator segment_it_1 = segment_indices.begin(); segment_it_1 != segment_indices.end(); segment_it_1++) {
	if (*segment_it_1 == solution_num_surfaces)
	  continue;
	int vertex_pixel = vertex_pixels[segment_it_1 - segment_indices.begin()];
        for (vector<int>::const_iterator segment_it_2 = segment_indices.begin(); segment_it_2 != segment_indices.end(); segment_it_2++) {
	  if (*segment_it_2 == solution_num_surfaces || *segment_it_2 == *segment_it_1)
            continue;
	  if (solution_segments.at(*segment_it_2).getDepth(vertex_pixel) > solution_segments.at(*segment_it_1).getDepth(vertex_pixel) || solution_segments.at(*segment_it_2).getDepth(vertex_pixel) < 0)
	    surface_occluding_relations[*segment_it_1][*segment_it_2]++;
          else
	    surface_occluding_relations[*segment_it_1][*segment_it_2]--;
	}
      }
      
      int selected_segment_index = -1;
      int selected_vertex = -1;
      for (vector<int>::const_iterator segment_it = segment_indices.begin(); segment_it != segment_indices.end(); segment_it++) {
        if (selected_segment_index == -1 || selected_segment_index == solution_num_surfaces) {
	  selected_segment_index = *segment_it;
	  selected_vertex = segment_it - segment_indices.begin();
        } else if (*segment_it != solution_num_surfaces && *segment_it != selected_segment_index) {
	  if (surface_occluding_relations[*segment_it][selected_segment_index] + surface_occluding_relations[selected_segment_index][*segment_it] > 0) {
            if (solution_segments.at(*segment_it).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) < solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) ||
                solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) < 0) {
              selected_segment_index = *segment_it;
	      selected_vertex = segment_it - segment_indices.begin();
            }
          } else if (surface_occluding_relations[*segment_it][selected_segment_index] + surface_occluding_relations[selected_segment_index][*segment_it] < 0) {
            if (solution_segments.at(*segment_it).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) > solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_)) {
              selected_segment_index = *segment_it;
	      selected_vertex = segment_it - segment_indices.begin();
            }
          } else {
            if (sqrt(pow(x - vertex_pixels[segment_it - segment_indices.begin()] % IMAGE_WIDTH_, 2) + pow(y - vertex_pixels[segment_it - segment_indices.begin()] / IMAGE_WIDTH_, 2)) < sqrt(pow(x - vertex_pixels[selected_vertex] % IMAGE_WIDTH_, 2) + pow(y - vertex_pixels[selected_vertex] / IMAGE_WIDTH_, 2))) {
              selected_segment_index = *segment_it;
	      selected_vertex = segment_it - segment_indices.begin();
            } 
	  }
	}
      }

      bool has_empty_neighbor = false;
      for (vector<int>::const_iterator segment_it = segment_indices.begin(); segment_it != segment_indices.end(); segment_it++) {
	int vertex_pixel = vertex_pixels[segment_it - segment_indices.begin()];
        if (*segment_it == solution_num_surfaces && visible_layer_indices[vertex_pixel] > layer_index) {
	  has_empty_neighbor = true;
	  break;
	}
      }
      if (selected_segment_index < solution_num_surfaces && has_empty_neighbor) {
	set<int> selected_segment_pixels;
        set<int> empty_pixels;
        int window_size = 2;
        while (selected_segment_pixels.size() == 0 || empty_pixels.size() == 0) {
          for (int delta_x = -window_size; delta_x <= window_size; delta_x++) {
            for (int delta_y = -window_size; delta_y <= window_size; delta_y++) {
              int window_pixel = min(max(static_cast<int>(round(y)) + delta_y, 0), IMAGE_HEIGHT_ - 1) * IMAGE_WIDTH_ + min(max(static_cast<int>(round(x)) + delta_x, 0), IMAGE_WIDTH_ - 1);
              if (layer_confident_pixel_mask[layer_index][window_pixel] == false)
		continue;
              int window_surface_id = solution_labels[window_pixel] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
              if (window_surface_id == selected_segment_index) {
		selected_segment_pixels.insert(window_pixel);
              }
              if (window_surface_id == solution_num_surfaces) {
                empty_pixels.insert(window_pixel);
              }
            }
          }
          window_size++;
	  if (window_size == 5)
	    break;
        }
	if (window_size == 5) {
	  if (empty_pixels.size() >= selected_segment_pixels.size()) {
	    selected_segment_index = solution_num_surfaces;
	  }
	} else {
	  vector<int> ori_window_pixels;
	  for (set<int>::const_iterator selected_segment_pixel_it = selected_segment_pixels.begin(); selected_segment_pixel_it != selected_segment_pixels.end(); selected_segment_pixel_it++) {
	    double ori_selected_segment_pixel = min(static_cast<int>(round(1.0 * (*selected_segment_pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*selected_segment_pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	    ori_window_pixels.push_back(ori_selected_segment_pixel);
	  }
	  for (set<int>::const_iterator empty_pixel_it = empty_pixels.begin(); empty_pixel_it != empty_pixels.end(); empty_pixel_it++) {
	    double ori_empty_pixel = min(static_cast<int>(round(1.0 * (*empty_pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*empty_pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	    ori_window_pixels.push_back(ori_empty_pixel);
	  }
	  vector<double> distances = cv_utils::calcGeodesicDistances(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_window_pixels, DISTANCE_2D_WEIGHT);
	  double min_distance = 1000000;
	  int min_distance_index = -1;
	  for (vector<double>::const_iterator distance_it = distances.begin(); distance_it != distances.end(); distance_it++) {
	    if (*distance_it < min_distance) {
	      min_distance_index = distance_it - distances.begin();
	      min_distance = *distance_it;
	    }
	  }
	  if (min_distance_index >= selected_segment_pixels.size())
	    selected_segment_index = solution_num_surfaces;
	}
      }
      
      solution_labels_high_res[ori_pixel] += selected_segment_index * pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index);
    }
  }
  
  
  writeLayers(ori_image_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_point_cloud_, ori_camera_parameters, num_layers_, solution_labels_high_res, solution_num_surfaces, solution_segments, 20000, ori_image_, ori_point_cloud_);
  
  upsampled_solution_labels = solution_labels_high_res;
  upsampled_solution_num_surfaces = solution_num_surfaces;
}
