import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
# get_dataloaders--> loads the data from the dataset and splits it into training, validation, and testing sets
from data_loader import get_dataloaders
# FNN--> Fully Connected Neural Network
# BonusCNN--> Convolutional Neural Network for Bonus Part
from models import FNN, BonusCNN
# train_model--> trains the model
# evaluate_model--> evaluates the model
from trainer import train_model, evaluate_model
# plot_history--> plots the training and validation loss and accuracy
# plot_confusion_matrix--> plots the confusion matrix
from utils import plot_history, plot_confusion_matrix

def run_experiment(name, model, train_loader, val_loader, test_loader, lr=0.01, epochs=5, device='cpu'):
    print(f"\n{'='*50}\nRunning Experiment: {name}\n{'='*50}")
    # CrossEntropyLoss is a loss function that is used to calculate 
    # the loss between the predicted output and the actual output.
    criterion = nn.CrossEntropyLoss()
    # SGD is an optimization algorithm that is used to update the weights 
    # of the model.
    # model.parameters()--> returns an iterator over all the parameters
    #                       of the model that the optimizer will update
    # lr--> learning rate-->step size
    # If lr is too high (e.g., 5.0), the optimizer violently turns the knobs, 
    # constantly overshooting the correct answer, and the model never learns.
    # If lr is too tiny (e.g., 0.0000001), the optimizer barely turns the knobs 
    # at all, and your model will take a very long time to train.
    # SGD --> Stochastic Gradient Descent
    # optim--> PyTorch's toolbox holding all of the Optimization algorithms
    # If the Loss function(CrossEntropyLoss)tells us how bad the model's guess was, 
    # the Optimizer is the mechanic that actually goes into the neural network
    # and tweaks the weights to fix the problem!
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    trained_model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        num_epochs=epochs, 
        device=device
    )
    
    plot_history(history, experiment_name=name, save_dir='plots',epochs=epochs)
    # evaluate_model--> evaluates the model on the test_loader
    test_acc, labels, preds = evaluate_model(trained_model, test_loader, device=device)
    print(f"Test Accuracy for {name}: {test_acc:.4f}")
    
    plot_confusion_matrix(labels, preds, experiment_name=name, save_dir='plots')
    
    return test_acc

def main():
    # check if the running will be on the GPU or CPU
    # if GPU is available, use it, otherwise use CPU
    # torch.cuda.is_available()--> checks if a compatible NVIDIA GPU is installed
    # i don't have it so my program will run on CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # less than 5, the model won't learn well
    # Set epochs to 5 for quicker testing, but can be scaled up.
    # more than 5, the model will learn better but it will take more time to train
    epochs = 5  
    base_batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=base_batch_size)
    
    results = {}
    
    # 1. Base Model
    # Architecture: 784(In) -> 128(H1) -> 64(H2) -> 10(Out)
    # input-->28*28=784
    # output --> 10 classes-->[0,1,2,3,4,5,6,7,8,9]
    # Each of these 10 output neurons will produce a "score". 
    # When the network sees an image of a '3', we want the 4th output neuron
    # (index 3) to have the highest score out of the 10! 
    # PyTorch handles those scores automatically to guess the correct digit.
    results['Base'] = run_experiment('Base_Model', FNN([784, 128, 64, 10]), train_loader, val_loader, test_loader, lr=0.01, epochs=epochs, device=device)
    
    # 2. Learning Rate Analysis
    lrs_to_test = [0.1, 0.05, 0.005, 0.001]
    for lr in lrs_to_test:
        results[f'LR_{lr}'] = run_experiment(f'LR_{lr}', FNN([784, 128, 64, 10]), train_loader, val_loader, test_loader, lr=lr, epochs=epochs, device=device)
        
    # 3. Batch Size Analysis
    batch_sizes = [16, 32, 128, 256]
    for b_size in batch_sizes:
        # with different batches we need to reload the data
        # batch 16-->very noisy updates, slow convergence
        # batch 256-->very stable updates, fast convergence
        # batch 64-->perfect balance
        # batch 128-->good balance
        print("\n")
        t_loader, v_loader, ts_loader = get_dataloaders(batch_size=b_size)
        results[f'Batch_{b_size}'] = run_experiment(f'Batch_{b_size}', FNN([784, 128, 64, 10]), t_loader, v_loader, ts_loader, lr=0.01, epochs=epochs, device=device)
        
    # 4. Neurons Analysis
    neuron_configs = [
        [784, 64, 32, 10], 
        [784, 256, 128, 10], 
        [784, 512, 128, 10],
        [784, 512, 256, 10]
    ]
    #i-->index
    #config-->list of layer sizes
    #i+1-->index+1
    #config[1]-->number of neurons in the first hidden layer
    #config[2]-->number of neurons in the second hidden layer
    for i, config in enumerate(neuron_configs):
        config_name = f"Neurons_Config_{i+1}_H1_{config[1]}_H2_{config[2]}"
        results[config_name] = run_experiment(config_name, FNN(config), train_loader, val_loader, test_loader, lr=0.01, epochs=epochs, device=device)
        
    # 5. Layers Analysis
    layer_configs = {
        '3_Hidden': [784, 128, 64, 32, 10],
        '4_Hidden': [784, 128, 64, 32, 16, 10],
        '5_Hidden': [784, 256, 128, 64, 32, 16, 10],
        '6_Hidden': [784, 256, 128, 64, 32, 16, 16, 10]
    }
    #name--> name of the configuration
    #config--> list of layer sizes for that configuration
    for name, config in layer_configs.items():
        results[name] = run_experiment(name, FNN(config), train_loader, val_loader, test_loader, lr=0.01, epochs=epochs, device=device)
        
    # 6. Bonus: CNN Architecture
    results['Bonus_CNN'] = run_experiment('Bonus_CNN', BonusCNN(), train_loader, val_loader, test_loader, lr=0.01, epochs=epochs, device=device)
    
    print("\n\nAll Experiments Completed!")
    for k, v in results.items():
        print(f"{k}: Test Acc = {v:.4f}")
    
    with open('results_summary.json', 'w') as f:
        # json.dump()--> serializes the results dictionary into a 
        # JSON formatted string and writes it to the file 'results_summary.json'
        # indent=4--> makes the JSON file more readable by adding indentation
        #indent=4-->every new result will be indented in a new line, make the json file easier to read and understand.
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
