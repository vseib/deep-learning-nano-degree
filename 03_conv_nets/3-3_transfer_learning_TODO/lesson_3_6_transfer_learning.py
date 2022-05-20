### Transfer Learning

'''
Most of the time you won't want to train a whole convolutional network yourself. Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs.

    Instead, most people use a pretrained network either as a fixed feature extractor, or as an initial network to fine tune.

In this notebook, you'll be using VGGNet trained on the ImageNet dataset as a feature extractor. Below is a diagram of the VGGNet architecture, with a series of convolutional and maxpooling layers, then three fully-connected layers at the end that classify the 1000 classes found in the ImageNet database.

VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully-connected layer with our own classifier. This way we can use VGGNet as a fixed feature extractor for our images then easily train a simple classifier on top of that.

    Use all but the last fully-connected layer as a fixed feature extractor.
    Define a new, final classification layer and apply it to a task of our choice!

You can read more about transfer learning from the CS231n Stanford course notes.
    
    http://cs231n.github.io/transfer-learning/

'''

### Flower power

'''
Here we'll be using VGGNet to classify images of flowers. We'll start, as usual, by importing our usual resources. And checking if we can train our model on GPU.
'''

### Download Data

'''
The flower data is available in a zip file in this lesson's resources, for download to your local environment. In the case of this notebook, the data is already downloaded and in the directory flower_photos/.
'''

import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


### Load and Transform our Data

'''
We'll be using PyTorch's ImageFolder class which makes is very easy to load data from a directory. For example, the training images are all stored in a directory path that looks like this:

root/class_1/xxx.png
root/class_1/xxy.png
root/class_1/xxz.png

root/class_2/123.png
root/class_2/nsdf3.png
root/class_2/asd932_.png

Where, in this case, the root folder for training is flower_photos/train/ and the classes are the names of flower types.
'''

# NOTE: dataset available at https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
# however, no train-test splits are provided, therefore use first 70% of each flower for train, last 30% for test

# define training and test data directories
data_dir = 'flower_photos/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

# classes are folders in each directory with these names
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


### Transforming the Data
'''
When we perform transfer learning, we have to shape our input data into the shape that the pre-trained model expects. VGG16 expects 224-dim square images as input and so, we resize each flower image to fit this mold.
'''

# load and transform data using ImageFolder

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))


### DataLoaders and Data Visualization

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

# Visualize some sample data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
plt.show() 


### Define the Model

'''
To define a model for training we'll follow these steps:

    1. Load in a pre-trained VGG16 model
    2. "Freeze" all the parameters, so the net acts as a fixed feature extractor
    3. Remove the last layer
    4. Replace the last layer with a linear classifier of your own

Freezing simply means that the parameters in the pre-trained model will not change during training.
'''

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

# print out the model structure
print(vgg16)

print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features) 

# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False


#### Final Classifier Layer

'''
Once you have the pre-trained feature extractor, you just need to modify and/or add to the final, fully-connected classifier layers. In this case, we suggest that you repace the last layer in the vgg classifier group of layers.

    This layer should see as input the number of features produced by the portion of the network that you are not changing, and produce an appropriate number of outputs for the flower classification task.

You can access any layer in a pretrained network by name and (sometimes) number, i.e. vgg16.classifier[6] is the sixth layer in a group of layers named "classifier".

TODO: Replace the last fully-connected layer with one that produces the appropriate number of class scores.
'''

## TODO: add a last linear layer  that maps n_inputs -> 5 flower classes
## new layers automatically have requires_grad = True
import torch
import torch.nn as nn

vgg16.classifier[6] = nn.Linear(4096, 5, bias=True)

# after completing your model, if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16.cuda()

############################ NOTE: official solution ##########################
#import torch.nn as nn
#
#n_inputs = vgg16.classifier[6].in_features
#
## add last linear layer (n_inputs -> 5 flower classes)
## new layers automatically have requires_grad = True
#last_layer = nn.Linear(n_inputs, len(classes))
#
#vgg16.classifier[6] = last_layer
#
## if GPU is available, move the model to GPU
#if train_on_gpu:
#    vgg16.cuda()
#
## check to see that your last layer produces the expected number of outputs
#print(vgg16.classifier[6].out_features)
##print(vgg16)
###############################################################################


### Specify Loss Function and Optimizer

'''
Below we'll use cross-entropy loss and stochastic gradient descent with a small learning rate. Note that the optimizer accepts as input only the trainable parameters vgg.classifier.parameters().
'''

import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001

### after  5 epochs: Test Accuracy (Overall): 77% (419/540)
### after 10 epochs: Test Accuracy (Overall): 81% (439/540)
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)

### after  5 epochs: Test Accuracy (Overall): 77% (421/540)
### after 10 epochs: Test Accuracy (Overall): 82% (446/540)
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

### after  5 epochs: Test Accuracy (Overall): 83% (449/540)
### after 10 epochs: Test Accuracy (Overall): 84% (455/540)
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.01)

### Training

'''
Here, we'll train the network.

    Exercise: So far we've been providing the training code for you. Here, I'm going to give you a bit more of a challenge and have you write the code to train the network. Of course, you'll be able to see my solution if you need help.
'''

# number of epochs to train the model
n_epochs = 10

## TODO complete epoch and training batch loops
## These loops should update the classifier-weights of this model
## And track (and print out) the training loss over time

vgg16.train() # training mode

for e in range(1, n_epochs+1):
    
    train_loss_v1 = 0.0
    train_loss_v2 = 0.0
    
    for batch, labels in train_loader:
        # use gpu if available
        if train_on_gpu:
            batch, labels = batch.cuda(), labels.cuda()

        optimizer.zero_grad()
        output = vgg16(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss_v1 += loss.item()*batch.size(0)
        train_loss_v2 += loss.item()
        
    print("--------- after epoch ", e)
    print("Train loss v1: ", train_loss_v1/len(train_loader.dataset))
    print("Train loss v2: ", train_loss_v2/len(train_loader))

############################ NOTE: official solution ##########################
'''
# number of epochs to train the model
n_epochs = 2

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0
'''            
###############################################################################


#### Testing
'''
Below you see the test accuracy for each flower class.
'''

# track test loss 
# over 5 flower classes
test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))

vgg16.eval() # eval mode

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg16(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

#### Visualize Sample Test Results

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = vgg16(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx].cpu(), (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
plt.show() 

