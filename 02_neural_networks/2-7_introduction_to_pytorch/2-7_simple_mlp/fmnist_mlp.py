import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torch.nn.functional as F

from datetime import datetime
time_format_dt = '%Y-%m-%d %H:%M:%S'
time_format_t = '%H:%M:%S'

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),
                              ])

# Download and load the training data
trainset = datasets.FashionMNIST('../F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.FashionMNIST('../F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# move to gpu if available
model.to(device)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 10
train_losses, test_losses, test_acc = [None], [None], [None]

# main loop
for e in range(epochs):

    running_loss_train = 0
    model.train()

    for images, labels in trainloader:
        # move to gpu if available
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        images = images.view(images.shape[0], -1)   
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item()
    
    running_loss_test = 0
    total_accuracy = 0
    model.eval()

    for images, labels in testloader:
        # move to gpu if available
        images, labels = images.to(device), labels.to(device)

        images = images.view(images.shape[0], -1)   
        output = model.forward(images)

        # get probabilities and compute accuracy
        probs = F.softmax(output, dim=1)
        top_probs, top_classes = probs.topk(1, dim=1)
        equals = (top_classes == labels.view(*top_classes.shape))
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        loss = criterion(output, labels)
        running_loss_test += loss.item()
        total_accuracy += accuracy

    # store losses and accuracy
    train_losses.append(running_loss_train / len(trainloader))
    test_losses.append(running_loss_test / len(testloader))
    test_acc.append(total_accuracy / len(testloader))

    time_stamp = "["+datetime.now().strftime(time_format_t)+"]"

    print(time_stamp+"\t",
          "Epoch: {}/{}".format(e+1, epochs)+"\t",
          "Train Loss: {:.4f}".format(running_loss_train / len(trainloader))+"\t",
          "Test Loss: {:.4f}".format(running_loss_test / len(testloader))+"\t",
          "Accuracy: {:.4f}".format(total_accuracy / len(testloader)))

# save trained model
model_save_name = 'fmnist_model_simple.pth'
torch.save(model.state_dict(), model_save_name)

# plot training statistics
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.plot(test_acc, label="Accuracy")
plt.legend(frameon=False)
plt.show()

# load model
state_dict = torch.load(model_save_name)
model.load_state_dict(state_dict)
model.to(device)

# perform inference
model.eval()
dataiter = iter(testloader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

img = images[0]
img = img.view(1, 784)

output = model.forward(img)
probs = F.softmax(output, dim=1)
top_prob, top_class = probs.topk(1, dim=1)
print("Result:\t", "class:", top_class.item(), "\tprob: {:.4f}".format(top_prob.item()))

# Plot the image and probabilities
import helper
img = img.cpu()
probs = probs.cpu()
helper.view_classify(img.view(1, 28, 28), probs, version='Fashion')

