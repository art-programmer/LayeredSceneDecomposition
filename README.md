# Layered Scene Decomposition via the Occlusion-CRF

By Chen Liu, Yasutaka Furukawa, and Pushmeet Kohli

### Introduction

This paper proposes a novel layered depth map representation and its inference algorithm which is able to infer invisible surfaces behind occlusions. To learn more, please see our CVPR 2016 [paper](http://www.cse.wustl.edu/~furukawa/papers/2016-cvpr-layer.pdf) or visit our [project website](http://sites.wustl.edu/chenliu/LayeredScene)

This code implements the algorithm described in our paper in C++.

### Requirements:

0. OpenCV
1. PCL
2. gflags

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @article{ren15fasterrcnn,
        Author = {Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun},
        Title = {{Faster R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
        Journal = {arXiv preprint arXiv:1506.01497},
        Year = {2015}
    }

### Main Results
                          | training data                          | test data            | mAP   | time/img
------------------------- |:--------------------------------------:|:--------------------:|:-----:|:-----:
Faster RCNN, VGG-16       | VOC 2007 trainval                      | VOC 2007 test        | 69.9% | 198ms
Faster RCNN, VGG-16       | VOC 2007 trainval + 2012 trainval      | VOC 2007 test        | 73.2% | 198ms
Faster RCNN, VGG-16       | VOC 2012 trainval                      | VOC 2012 test        | 67.0% | 198ms
Faster RCNN, VGG-16       | VOC 2007 trainval&test + 2012 trainval | VOC 2012 test        | 70.4% | 198ms

**Note**: The mAP results are subject to random variations. We have run 5 times independently for ZF net, and the mAPs are 59.9 (as in the paper), 60.4, 59.5, 60.1, and 59.5, with a mean of 59.88 and std 0.39.


### Contents
0. [Requirements: software](#requirements-software)
0. [Requirements: hardware](#requirements-hardware)
0. [Preparation for Testing](#preparation-for-testing)
0. [Testing Demo](#testing-demo)
0. [Preparation for Training](#preparation-for-training)
0. [Training](#training)
0. [Resources](#resources)


