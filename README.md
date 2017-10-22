# Layered Scene Decomposition via the Occlusion-CRF

By Chen Liu, Yasutaka Furukawa, and Pushmeet Kohli

### Introduction

This paper proposes a novel layered depth map representation and its inference algorithm which is able to infer invisible surfaces behind occlusions. To learn more, please see our CVPR 2016 [paper](http://www.cse.wustl.edu/~furukawa/papers/2016-cvpr-layer.pdf) or visit our [project website](http://sites.wustl.edu/chenliu/layered-scene)

This code implements the algorithm described in our paper in C++.

### Requirements

0. OpenCV
1. PCL
2. gflags

### Usage

To compile the program:

0. mkdir build
1. cd build
2. cmake ..
3. make

To run the program on your own data:

./LayeredSceneDecomposition --image_path=*"your image path"* --point_cloud_path=*"your point cloud path"* --result_folder=*"where you want to save results"* --cache_folder=*"where you want to save cache"*

To run the program on the demo data:

./LayeredSceneDecomposition --image_path=../Input/image_01.txt --point_cloud_path=../Input/point_cloud_01.txt --result_folder=../Result --cache_folder=../Cache

Point cloud file format:

The point cloud file stores a 3D point cloud, each of which corresponds to one image pixel.
The number in the first row equals to image_width * image_height.
Then, each row stores 3D coordinates for a point which corresponds to a pixel (indexed by y * image_width + x).

### Contact

If you have any questions, please contact me at chenliu@wustl.edu.
