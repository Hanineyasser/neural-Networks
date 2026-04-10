import torch
import torch.nn as nn
# nn.Module is the base class for all neural network modules in PyTorch.
# It is a class that provides a standard interface for all neural network modules in PyTorch.
class FNN(nn.Module):
    def __init__(self, layer_sizes):

       #E.g. [784, 128, 64, 10] means: Input=784, Hidden1=128, Hidden2=64, Output=10

        super(FNN, self).__init__()
        layers = []
        # Add hidden layers with ReLU
        for i in range(len(layer_sizes) - 2):
            # For each hidden layer, we add a Linear layer followed by a ReLU activation function.
            # it maps every nput to output 
            #nn.Linear--> performs a linear transformation on the input data
            #           fully connected layer
            #           y=w*x+b
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            #makes the model non-linear --> allows the model to learn complex patterns
            layers.append(nn.ReLU())
            
        # Add output layer (no activation here, CrossEntropyLoss handles softmax)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        # PyTorch automatically feeds the data through the first layer, 
        # takes the output, feeds it into the second layer, takes that output, 
        # feeds it into the third layer, and so on.
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten image
        # x.view(x.size(0), -1)--> flattens the input tensor x into a 1D vector
        # so that it camn be fed into a fully connected layer
        # x.size(0)--> number of samples in the batch-->batch_size
        # by setting trhe 1st argument to batch_size we are telling PyTorch 
        # to keep the batch dimension and do not merge different samples together
        # -1--> automatically calculates the number of features
        # It multiplies all the other dimensions (like Width*Height*Channels) 
        # and puts them into one single dimension.
        x = x.view(x.size(0), -1)
        return self.network(x)


class BonusCNN(nn.Module):
    def __init__(self):
        super(BonusCNN, self).__init__()
        # Input shape: (Batch, 1, 28, 28)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.LayerNorm([32, 28, 28]), # LayerNorm usage required by bonus
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (Batch, 32, 14, 14)
            nn.Dropout(p=0.25),          # Dropout usage required by bonus
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm([64, 14, 14]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (Batch, 64, 7, 7)
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
