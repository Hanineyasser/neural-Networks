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

# While the Perceptron you looked at earlier treats images as flat lists of numbers, 
# a CNN treats them as spatial grids. It looks for local patterns like edges, shapes.
class BonusCNN(nn.Module):
    def __init__(self):
        super(BonusCNN, self).__init__()
        # Input shape: (Batch, 1, 28, 28)-->Batch size, number of channels, height, width

        # feature extraction layers-->This part extracts spatial information from the pixels. 
                                    # It transforms the raw image into a set of "high-level features.
        #Sequential-->groups multiple layers together in a single module
        self.features = nn.Sequential(
            # Instead of every neuron connecting to every pixel, 
            # a small window (kernel) slides across the image.
            # 3 phases--> scanning - sliding - feature mapping
            # in_channels-->number of input channels-->1(grayscale)--MNIST dataset has 1 channel
            # out_channels-->scans the image and detects 32 different features
            # kernel_size-->size of the convolutional kernel-->3x3
            # padding-->padding around the input image-->1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # BatchNorm depends on other images in the same batch
            # LayerNorm treats every image as a stand alone only 
            # cares about the pixels within that specific sample.
            nn.LayerNorm([32, 28, 28]), # LayerNorm usage required by bonus
            # the Activation Function. 
            # It is the "gatekeeper" that decides which information is important 
            # enough to be passed to the next layer and which should be discarded.
            nn.ReLU(),
            # downsampling layer
            # reduces the spatial dimensions of the feature maps
            # It cuts the Height and Width in half (from 28x28-->14x14). 
            # This reduces computation and helps the model become "translation invariant"
            # (it can recognize a shape even if it's shifted slightly).
            nn.MaxPool2d(kernel_size=2), # Output: (Batch, 32, 14, 14)
            # Dropout is a regularization technique that helps prevent overfitting.
            # It works by randomly setting a fraction of the input units to 0 at each update.
            # 25% are deactivated randomly
            # It forces the model to be redundant. 
            # It can't rely on just one specific neuron to recognize a "gamma" particle; 
            # it has to learn multiple ways to see the same thing.
            nn.Dropout(p=0.25),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LayerNorm([64, 14, 14]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (Batch, 64, 7, 7)
            nn.Dropout(p=0.25)
        )
        # classifier --> This takes those high-level features and makes the final decision 
                       # (predicting one of 10 classes)
        self.classifier = nn.Sequential(
            # Flatten--> Converts the 3D feature maps (64x7x7) into a 1D vector (3136 elements)
            # It takes the 64 feature maps, each of size 7x7, and "unrolls" them into one long vector.
            # same as x.view(x.size(0), -1) in FNN
            nn.Flatten(),
            # size of input is 64*7*7=3136
            # performs --> y=w*x+b
            # each 128 neurons is connected to all 3136 neurons
            # 128 -->hyperparameter
            #too small-->underfitting--> forgets important details
            #too large-->overfitting
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            # we have over 400000 internal weights
            # Linear layers possess so much mathematical power
            # often 5/6 neurons make the heavy work and the rest are sleeping
            #To stop this laziness, we use a brutal 50% Dropout
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # extracting the features
        x = self.features(x)
        # classifying the features
        x = self.classifier(x)
        return x
