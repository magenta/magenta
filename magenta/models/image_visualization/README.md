# Image Visualization

This repo contains the TensorFlow implementation of CNN visualization.  

In general, after neural networks are trained on image data they are used to classify new images into various classes (or types). Visualization is the task of using a trained model to produce an image that generalizes all images of its class. Because visualizations are generated from the learned parameters of neural networks, they are good representations of how the network perceives a certain class. Lots of foundational work on image visualization was published in the following papers:  
[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034). *Karen Simonyan, Andrea Vedaldi, Andrew Zisserman*.  
[Understanding Neural Networks Through Deep Visualization](http://www.evolvingai.org/files/2015_Yosinski_ICML.pdf). *Jason Yosinski et al.*.  

Google's DeepDream algorithm is a variation on image visualization where a normal image is passed through a neural network and is modified such that all activations are boosted at any layer. This creates a continuous loop that will continue to develop the classes visualized in the image over time. [DeepDream GitHub](https://github.com/google/deepdream)  

Visualization can be taken a step further by using it as a tool to generate art. For this, new techniques for eliminating noise, scaling up the image, and other image processing have been developed over the past few years. The code in this demo comes from the deepdream code by Google, the [code](https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb) from Kyle McDonald, [code](https://github.com/auduno/deepdraw) from Auduno for using GoogleNet, and the use of bilateral filtering from Mike Tyka.

Recently, the deepdream visualization method was implemented in TensorFlow by [InFoCusp in tf_cnnvis](https://github.com/InFoCusp/tf_cnnvis). Most of the code in this repo is based on this library.

So far, the demo uses Google's InceptionV5 CNN, which was trained on the ImageNet database.

# Setup and Visualization
To visualize images from InceptionV5, set up your [Magenta environment](/README.md).  

Also add the following packages to your python environment:
```
pip install h5py wget scikit-image six
```

In this repo, a 'params.py' file describes all of the parameters that 'class_visualization.py' will use in generating the visualization. Experimenting with these parameters, along with the other flags passed to 'class_visualization.py' us recommended. Here is a list of the existing parameters and default flag values:

```python
# params.py
octaves = [{
    'scale': 1.4,
    'iter_n': 190,
    'start_sigma': .44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 150,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }, {
    'scale': 1.4,
    'iter_n': 10,
    'start_sigma': 0.44,
    'end_sigma': 0.304,
    }]
background_color = np.float32([10, 90, 140])
classes = [1,2,3,4,5]

# flags
model="tensorflow_inception_graph.pb"                          # Model
layer="import/softmax2_pre_activation"                         # Layer name
params_path="."                                                # Path to folder containing python file 'params.py'
logdir_path="./Log/Inception5"                                 # Path to folder where logs will be stored
outdir_path="./Output/Inception5"                              # Path to folder where generated images will be stored")
use_bilateral=True                                             # Flag for whether or not to use bilateral filtering, which is dependent on start_sigma and end_sigma in 'params.py'
use_tv_bregman=True                                            # Flag for whether or not to perform total-variation denoising using split-Bregman optimization
```
