### Convolutional Neural Networks

'''
In this notebook, we train a CNN to classify images from the CIFAR-10 database.

The images in this database are small color images that fall into one of ten classes; some example images are pictured below.
'''

'''
Test for CUDA

Since these are larger (32x32x3) images, it may prove useful to speed up your training time by using a GPU. CUDA is a parallel computing platform and CUDA Tensors are the same as typical Tensors, only they utilize GPU's for computation.
'''

import torch
import numpy as np

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

'''
Load the Data

Downloading may take a minute. We load in the training and test data, split the training data into a training and validation set, then create DataLoaders for each of these sets of data.
'''

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

##### alternatively with data augmentation
transform_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform_aug)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform_test)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#### Visualize a Batch of Training Data

import matplotlib.pyplot as plt
#%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()    

'''
View an Image in More Detail

Here, we look at the normalized red, green, and blue (RGB) color channels as three separate, grayscale intensity images.
'''

rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel']

fig = plt.figure(figsize = (36, 36)) 
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1, 3, idx + 1)
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(channels[idx])
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center', size=8,
                    color='white' if img[x][y]<thresh else 'black')
plt.show()

#### Define the Network Architecture
'''
This time, you'll define a CNN architecture. Instead of an MLP, which used linear, fully-connected layers, you'll use the following:

    Convolutional layers, which can be thought of as stack of filtered images.
    Maxpooling layers, which reduce the x-y size of an input, keeping only the most active pixels from the previous layer.
    The usual Linear + Dropout layers to avoid overfitting and produce a 10-dim output.

A network with 2 convolutional layers is shown in the image below and in the code, and you've been given starter code with one convolutional and one maxpooling layer.
TODO: Define a model with multiple convolutional layers, and define the feedforward network behavior.

The more convolutional layers you include, the more complex patterns in color and shape a model can detect. It's suggested that your final model include 2 or 3 convolutional layers as well as linear layers + dropout in between to avoid overfitting.

It's good practice to look at existing research and implementations of related models as a starting point for defining your own models. You may find it useful to look at this PyTorch classification example or this, more complex Keras example to help decide on a final structure.
Output volume for a convolutional layer

To compute the output size of a given convolutional layer we can perform the following calculation (taken from Stanford's cs231n course):

    We can compute the spatial size of the output volume as a function of the input volume size (W), the kernel/filter size (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. The correct formula for calculating how many neurons define the output_W is given by (W−F+2P)/S+1.

For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.
'''

### see here for the winning architecture:
# http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/

### see here for pytorch tutorial:
# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

#################### NOTE this ist version 1, the bigger model

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # setup
        num_classes = 10
        drop_p = 0.5
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layer
        self.fc1 = nn.Linear(1024, 256, bias=True)
        self.fc2 = nn.Linear(256, 64, bias=True)
        self.fc3 = nn.Linear(64, num_classes, bias=True)
        # dropout
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image, keep batch size
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


##################### NOTE this is version 2, the smaller model

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # setup
        num_classes = 10
        drop_p = 0.25
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layer
        self.fc1 = nn.Linear(512, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.fc3 = nn.Linear(64, num_classes, bias=True)
        # dropout
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image, keep batch size
        x = x.view(x.shape[0], -1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

##################### NOTE this is the official solution example, it is similar to version 1
# * Conv-layer sind wie bei mir
# * hat einen fc layer weniger als ich
# * hat einen dropout hinter dem letzten conv, das hatte ich nicht
# * hat eine relu hinter dem ersten fc, das hatte ich nicht

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layer
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image, keep batch size
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#################### NOTE this is version 3, after seeing the official solution

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # setup
        num_classes = 10
        drop_p = 0.5
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layer
        self.fc1 = nn.Linear(128 * 2 * 2, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, num_classes, bias=True)
        # dropout
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image, keep batch size
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


################################################################


# create a complete CNN
#model = Net1() # own bigger model
#model = Net2() # own smaller model
#model = Net()  # official solution example
model = Net3() # own final version after seeing the solution example

print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()


'''
Specify Loss Function and Optimizer

Decide on a loss and optimization function that is best suited for this classification task. The linked code examples from above, may be a good starting point; this PyTorch classification example or this, more complex Keras example. Pay close attention to the value for learning rate as this value determines how your model converges to a small error.
TODO: Define the loss and optimizer and see how these choices change the loss over time.
'''

import torch.optim as optim

# specify loss function
criterion = nn.NLLLoss()

# specify optimizer
# version 1, dropout = 0.50, mostly with 20 to 30 epochs
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 70% (7029/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 72% (7212/10000)
optimizer = optim.SGD(model.parameters(), lr=0.025)  # Test Accuracy (Overall): 71% (7172/10000)
# version 1, dropout = 0.25
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 71% (7185/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 71% (7121/10000)
optimizer = optim.SGD(model.parameters(), lr=0.025)  # Test Accuracy (Overall): 71% (7174/10000)

# version 2, dropout = 0.50
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 68% (6894/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 68% (6810/10000)
# version 2, dropout = 0.25
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 67% (6719/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 68% (6802/10000)

# version 3, dropout = 0.50
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 70% (7063/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 74% (7446/10000)
# version 3, dropout = 0.25
optimizer = optim.Adam(model.parameters(), lr=0.001) # Test Accuracy (Overall): 71% (7103/10000)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 74% (7463/10000)
# version 3, dropout = 0.25, with data augmentation, 35 epochs
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 77% (7702/10000)
# version 3, dropout = 0.25, with data augmentation, 60 epochs
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 79% (7925/10000)
# version 3 + dilation=2 in first 2 conv layers, dropout = 0.25, with data augmentation, 60 epochs
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Test Accuracy (Overall): 75% (7595/10000)


'''
Train the Network

Remember to look at how the training and validation loss decreases over time; if the validation loss ever increases it indicates possible overfitting.
'''

# number of epochs to train the model
n_epochs = 30 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

### Load the Model with the Lowest Validation Loss

model.load_state_dict(torch.load('model_cifar.pt'))

'''
Test the Trained Network

Test your trained model on previously unseen data! A "good" result will be a CNN that gets around 70% (or more, try your best!) accuracy on these test images.
'''

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
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

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

'''
Question: What are your model's weaknesses and how might they be improved?

Answer:
* Adam Optimizer does not work ... try other learning rates?
    * smaller learning rates work
* SGD works fine, but does not go beyond 71 % --> maybe simplify the architecture, use less params
    * less params works worse
    * best result: more complex model with dropout 0.5, SGD with lr=0.01 --> 72 % test accuracy

'''

### Visualize Sample Test Results

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx].cpu())
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
plt.show()                   
