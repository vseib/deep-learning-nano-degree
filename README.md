# deep_learning_nano_degree

This repository contains exercises that I completed during [Udacity's Nano Degree program on Deep Learning](https://www.udacity.com/course/deep-learning-nanodegree--nd101). Some of these exercises were part of the program while others were additional things that I tried based on the given tasks.

## Description

The exercises contained in this repository are very diverse. I provide a detailed description in the README.md files of each of the subfolders along with my reasoning for the presented solutions. I also provide **additional material** for most lectures. All exercises were converted to Python-files. The original Jupyter notebooks can be found in Udacity's repository (see Acknowledgments).

The subfolders are prefixed with the lecture number and contain the following material:

* 02_neural_networks
    * lecture 2.5: project 1: predicting bike sharing patterns
        * additional material: PyTorch implementation
    * lecture 2.6: sentiment analysis
        * additional material: PyTorch implementation
    * lecture 2.7: introduction to PyTorch
        * additional material: 2 different MLPs for MNIST and FMNIST, evaluation of hyperparameters


## Acknowledgments

The original repository of this Nano Degree program can be found here

```
    https://github.com/udacity/deep-learning-v2-pytorch.git
```

Since I was learning PyTorch during that course, many of my solutions are embedded into the original task descriptions of Udacity. Further, my code is very similar in style and structure to what I learned during that program.

## Installation

* Download and install Anaconda from the [Anaconda website](https://www.anaconda.com/distribution/).
* Create and activate an environment (adjust python version if needed):

```
conda create -n nano_dl python=3.7
conda activate nano_dl
conda install numpy pandas matplotlib scikit-learn
conda install pytorch torchvision -c pytorch
```
This will already install the cudatoolkit 10 package for PyTorch. Note however that you still need to install Cuda to be able to use the GPU in PyTorch.

Since I do not have a GPU in my computer, the following instructions might be incomplete or not entirely correct. Consider them only as a hint.
In Ubuntu you might need additional packages after installing Cuda.

```
sudo apt install nvidia-390
```
Further, you need to activate the graphics drivers in the system settings.
Depending on the Cuda version available for your system, you might need to downgrade the installed PyTorch cudatoolkit, e.g. like this:

```
conda activate nano_dl
conda install cudatoolkit=9.0 -c pytorch
```

#### Test if the GPU is correctly found by PyTorch

```
conda activate nano_dl
python
```

* Inside the Python console type the following

```
import torch
torch.cuda.is_available()
```

* If this prints out 'True' then your GPU was correctly found by PyTorch

## License

This code is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for details.
