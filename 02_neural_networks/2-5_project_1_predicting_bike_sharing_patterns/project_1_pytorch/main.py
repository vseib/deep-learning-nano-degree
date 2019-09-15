'''
In this project, you'll build your first neural network and use it to predict daily bike rental ridership. We've provided some of the code, but left the implementation of the neural network up to you (for the most part). After you've submitted this project, feel free to explore the data and the model more.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############ LOAD AND PREPARE THE DATA #################

'''
A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!
'''

#print("Load and prepare data ...")
data_path = '../Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)

#print("Example rides:")
#print(rides.head())

############ CHECKING OUT THE DATA #################

'''
This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the cnt column. You can see the first few rows of the data above.

Below is a plot showing the number of bike riders over the first 10 days or so in the data set. (Some days don't have exactly 24 entries in the data set, so it's not exactly 10 days.) You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.
'''

rides[:24*10].plot(x='dteday', y='cnt', title='Data of first 10 days in dataset')
plt.show()

############ DUMMY VARIABLES #################

'''
Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to get_dummies().
'''

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
#print("Example data:")
#print(data.head())


############ SCALING TARGET VARIABLES ######################

'''
To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

The scaling factors are saved so we can go backwards when we use the network for predictions.
'''

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

'''
Splitting the data into training, testing, and validation sets

We'll save the data for the last approximately 21 days to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.
'''

# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


############ TIME TO BUILD THE NETWORK ######################


import sys
import torch
import torch.nn as nn
import numpy as np

from my_answers import NeuralNetwork

####################
### Set the hyperparameters in you myanswers.py file ###
####################

from my_answers import iterations, learning_rate, hidden_nodes, output_nodes

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
print(network)

from torch import optim
# specify loss
criterion = nn.MSELoss()
# specify optimizer
optimizer = optim.SGD(network.parameters(), lr = learning_rate)
#optimizer = optim.Adam(network.parameters(), lr = learning_rate)

losses = {'train':[], 'validation':[]}
last_lr = learning_rate
for ii in range(iterations):      

    # varying learning rate for SGD
    if ii == 1500:
        if isinstance(optimizer, optim.SGD):
            learning_rate = 0.1
            print("\n  reducing learning rate from", last_lr, "to", learning_rate)
            optimizer = optim.SGD(network.parameters(), lr = learning_rate)
            last_lr = learning_rate
    if ii == 5000:
        if isinstance(optimizer, optim.SGD):
            learning_rate = 0.05
            print("\n  reducing learning rate from", last_lr, "to", learning_rate)
            optimizer = optim.SGD(network.parameters(), lr = learning_rate)

    # Go through a random batch of 128 records from the training data set
    batch_idx = np.random.choice(train_features.index, size=128)
    data, targets = train_features.ix[batch_idx].values, train_targets.ix[batch_idx]['cnt']   

    ###################
    # train the model #
    ###################
    network.train()
    train_loss = 0

    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    data = data.astype(np.float32)
    output = network(torch.from_numpy(data))

    # calculate the loss
    targets = targets.astype(np.float32)
    targets_tensor = torch.from_numpy(targets.values)
    loss = criterion(output.squeeze(), targets_tensor)
    train_loss = loss.item()

    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()


    ###################
    #  eval the model #
    ###################
    network.eval()
    val_loss = 0

    # forward pass: compute predicted outputs by passing inputs to the model
    val_features = val_features.astype(np.float32)
    output = network(torch.from_numpy(val_features.values))

    # calculate the validation loss
    val_targets = val_targets.astype(np.float32)
    val_targets_tensor = torch.from_numpy(val_targets['cnt'].values)
    loss = criterion(output.squeeze(), val_targets_tensor)
    val_loss = loss.item()

    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

print("\n----------------- losses after last step ---------------")
print("training loss: {:.3f}".format(train_loss))
print("validation loss: {:.3f}".format(val_loss))

########### PLOT LOSSES ###################

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()

############ CHECK OUT PREDICTIONS ####################

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']

test_features = test_features.astype(np.float32)
output = network(torch.from_numpy(test_features.values))
output = output.detach().numpy().squeeze()
predictions = output * std + mean

ax.plot(predictions, label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()


