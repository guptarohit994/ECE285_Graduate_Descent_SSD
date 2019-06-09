# ECE 285 UCSD - Spring '19 - Final project - Team Graduate Descent

# Input New York City Walking Tour Videostream
[![Video stream that was used as input for detection](https://imgur.com/1hcwxrk.gif) ](https://www.youtube.com/watch?v=u68EWmtKZw0) 

# Clean Videostream object detected
[![Video stream that was used as input for detection](https://imgur.com/5BLCHJh.gif) ](https://imgur.com/GJDkbfu.gif) 

# Noisy Videostream object detected
[![Watch the video](https://i.imgur.com/1TP6yRMh.gif)](https://imgur.com/1TP6yRM)

# Denoised Videostream object detected
[![Watch the video](https://imgur.com/vrdP6Qa.gif)](https://imgur.com/9hJaIPe.gif)


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
```pip install -requirements.txt```
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
Apart from links above for SSD Arxiv paper and VOC dataset documentation:
- [Project problem statement document](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)
