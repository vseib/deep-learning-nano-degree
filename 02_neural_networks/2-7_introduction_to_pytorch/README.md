# 2-7_introduction_to_pytorch

## Quick Start

### Datasets

The datasets will be downloaded automatically when running the scripts.

### Execute one of the provided scripts

This lecture was split into multiple small tasks that each introduced another feature of PyTorch. I summarized most of this steps in one script and included the detailed comments provided by Udacity. This script will train on the MNIST dataset for 1 epoch and display a classification result. To run this script do the following:

```
cd 2-7_intro
python introduction_training.py
```

The next folders contains a simple MLP with 3 layers and a more coplex MLP with 4 layers and dropout. The script will
    * train for 10 / 20 epochs, plot the losses, save the model
    * load the model, perform 1 inference, display results

Another difference between the two MLPs is that the more complex one is defined inside an own class.
Both MLPs are prepared for the MNIST and the Fashion MNIST datasets.
Additionally, there is a script to evaluate hyperparameters for each MLP and each dataset.

To use the simple version type:

```
cd 2-7_simple_mlp
```

And for the more complex one:

```
cd 2-7_new_mlp
```

Then choose one of the scripts:

```
python mnist_mlp.py
```

or

```
python fmnist_mlp.py
```

To test different hyperparameters either run

```
python mnist_mlp_optimizer_eval.py
```

or

```
python fmnist_mlp_optimizer_eval.py
```


## Description


