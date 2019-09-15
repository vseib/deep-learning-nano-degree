import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)
        self.lf = learning_rate
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)       
        return x


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000   
learning_rate = 0.25 # Adam: 0.02, SGD: 0.25 
hidden_nodes = 5     
output_nodes = 1
