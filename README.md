# deep_learning_nano_degree

This repository contains exercises that I completed during [Udacity's Nano Degree program on Deep Learning](https://www.udacity.com/course/deep-learning-nanodegree--nd101). Some of these exercises were part of the program while others were additional things that I tried based on the given tasks.

## Description

The exercises contained in this repository are very diverse. I provide a detailed description in the README.md files of each of the subfolders along with my reasoning for the presented solutions.

## Acknowledgments

The original repository of this Nano Degree program can be found here

```
    https://github.com/udacity/deep-learning-v2-pytorch.git
```

Since I was learning pytorch during that course, many of my solutions are embedded into the original task descriptions of Udacity. Further, my code is very similar in style and structure to what I learned during that program.

## Installation

* Download and install Anaconda from the [Anaconda website](https://www.anaconda.com/distribution/).
* Create and activate an environment (note adjust python version if needed):

```
conda create -n nano_dl python=3.7
conda activate nano_dl
conda install numpy pandas matplotlib scikit-learn
conda install pytorch torchvision -c pytorch
```

* To use a GPU additionally perform the following steps (note that you might need to activate the GPU drivers in you system settings)

```
sudo apt install nvidia-384
conda activate nano_dl
conda install cudatoolkit=9.0 -c pytorch
```
* Test if the GPU is correctly found by PyTorch

```
conda activate nano_dl
python
```

* Inside the python console type the following

```
import torch
torch.cuda.is_available()
```

* If this prints out 'True' than your GPU was correctly found by PyTorch

## License

This code is released under the BSD-3-Clause license. See [LICENSE](LICENSE) for details.
