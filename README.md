# ECE 285 UCSD - Spring '19 - Final project - Team Graduate Descent

# Input New York City Walking Tour Videostream
[![Video stream that was used as input for detection](https://imgur.com/1hcwxrk.gif) ](https://www.youtube.com/watch?v=u68EWmtKZw0) 

# Clean Videostream object detected
![](demos/CLEAN_FINAL.gif)

# Noisy Videostream object detected
![](demos/NOISY_FINAL.gif)

# Denoised Videostream object detected
![](demos/DENOISED_FINAL.gif)


# Single Shot Detector 
A PyTorch implementation of the SSD Multibox Detector for image feature extraction, based on the 2016 [Arxiv](http://arxiv.org/abs/1512.02325) paper by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang, and Alexander C. Berg.1TP6yRMh.gif
## Table of contents
- Installation
- Datasets
- Training
- Evaluation
- Performance
- Demo notebook1hcwxrk.gif
- References
- Directory structure
### Installation
```pip install -r requirements.txt```
### Datasets
[2012 version](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) of Pascal VOC dataset - well known dataset for object detection/classification/segmentation. Contains 100k images for training and validation containing bounding boxes with 20 categories of objects.
### Training
Do ```python train.py``` with parameters listed in the file as a flag or pass your own parameters.
#### NOTE: NVIDIA/CUDA enabled GPU is recommended for speed.
### Evaluation
Do ```python eval.py``` with parameters listed in the file as a flag or pass your own parameters.
### Performance <br>
On UCSD Data Science and Machine Learning Cluster - [more info](https://datahub.ucsd.edu/hub/login):
- Training:
- Evaluation:
### Demo notebook
Visualization notebooks in the root directory with plots of descent of loss function and other evaluation metrics, and sample images with detected categories shown with overlaid bounding boxes.
### References <br>
Apart from links above for SSD Arxiv paper and VOC dataset documentation, we referred to:
- [Project problem statement document](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)
### Directory structure
- pycache/ - .pyc files for Python interpreter to compile the source to
- data/ - 
  - pycache/ - .pyc files for Python interpreter to compile the source to
  - init.py - contains instances:
    - function detection_collate - stack images in 0th dimension and list of tensors with annotations for image and return in tuple format, given tuple of tensor images and list of annotations
    - function base_transform - resize and mean-normalize image
    - class BaseTransform - call base_transform(image) iteratively
  - config.py - configures VOC dataset with source directory, mean values, color ranges and SSD parameters
  - voc0712.py - configures VOC dataset with labels considered, and contains instances:
    - class VOCAnnotationTransform - store dictionaries of classname:index mappings, with an option to discard difficult instances
    - class VOCDetection - update and store annotation based on input image, with functions to get item, pull item, image, annotation and tensor
- demos/ - demo gifs to show performance of SSD on noisy, clean and denoised video streams (source files for the .gifs shown above)
- denoising_experiments/ -
  - .ipynb_checkpoints/ - checkpoints folder for modular running of python notebooks
  - pycache/ - .pyc files for Python interpreter to compile the source to
  - NOISE_PARAMS.pkl - Pickle file for noise parameters
  - SSD_Denoise_Eval.ipynb - notebook for evaluating performance on denoised stream
  - SSD_Denoise_Experiments.ipynb - notebook for evaluating performance on denoised stream
  - SSD_Denoising.ipynb - notebook to visualize denoising algorithm
  - SSD_Noisy_Eval.ipynb - notebook for evaluating performance on noisy stream
  - nntools.py - class script for base classes to implement neural nets, evaluate performance, specify metrics etc.
- devkit_path / -
  - annotations_cache/ - 
    - annots.pkl - Pickle file for annotations
  - results/ - result files for each class
- eval/ -
  - test1.txt - ground truth bbox vales and predictions for a selected portion of VOC dataset
  - test1_Denoise.txt - ground truth bbox vales and predictions for a selected portion of the VOC dataset AFTER noising and denoising
- layers/ -
  - pycache/ - .pyc files for Python interpreter to compile the source to
  - functions/ -
    - pycache/ - .pyc files for Python interpreter to compile the source to
    - init.py - import all files in pwd
    - detection.py - contains instances:
      - class Detect - enable decoding of location predictions of bboxes and apply NMS based on confidence values and threshold; restrict to tok_k output predictions to reduce noise in results quality (not actual image noise)
        - function init - allocate memory and initialize
        - function forward - forward propagation to update layers given input location prediction, confidence and prior data from their respective layers
    - prior_box.py - contains instances:
      - class PriorBox - collate and store priorbox coordinates in center-offset form and tie it to each source feature map
        - function init - allocate memory and initialize
        - forward - forward propagation through priorbox layers
  - modules/ -
    - pycache/ - .pyc files for Python interpreter to compile the source to
    - init.py - import all files in pwd
    - l2norm.py - contains instances:
      - class L2Norm - calculate L2 norm and normalize
        - function init - allocate memory and initialize
        - forward - compute the norm and return
    - multibox_loss.py - contains instances:
      - class MultiBoxLoss - compute targets for confidence and localization and apply HNM; using a loss function that is weighted between the cross entropy loss and a smooth L1 loss (weights were found during cross validation)
        - function init - allocate memory and initialize
        - function forward - forward propagate through multibox layers, given tuple of location and confidence predictions, prior box values and ground truth boxes and labels in tensor format
  - init.py - import all files in pwd
  - box_utils.py - contains instances:
    - function point_form - convert prior box values from center-size format for easy comparison to point form ground truth data
    - function center_size - convert prior box values to center-size format for easy comparison to center-size ground truth data
    - function intersect - compute area of intersection between two given boxes
    - function jaccard - compute jaccard overlap or intersection over union of two boxes
    - function match - match prior box with ground truth box (for all boxes) based on highest jaccard overlap, encode in required format (point-form or center-size), and return matching indices for the given confidence and location predictions
    - function encode - encode variances from prior box layers into ground truth boxes
    - function decode - decode locations from priors and locations and return bbox predictions
    - function log_sum_exp - compute log of sum of exponent of difference between current tensor and maximum value of tensor, for unaveraged confidence loss
    - function nms - compute non-maximum suppression to avoid too many overlapping bboxes that highlight nearly the same area
  - optimization_experiments/ - 
