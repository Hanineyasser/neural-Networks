import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloaders(batch_size=64):
    # Transform to tensor
    # transform.Compose--> container to allows you to chain multiple image 
    #                   preprocessing steps together into a single pipeline.
    # image-->matix of pixels grid of integers from 0 to 255
    #         it was presented as (H,W,C)--> (height,width,channels)
    #TENSOR--> multi-dimensional array of numbers-->rank 3 tensor-->height,width,color channels(RGB)
    # ToTensor-->converts the image into a tensor to make data compatible with neural networks
    #             it is presented as (C,H,W)--> (channels,height,width)
    #             convert the image from integer array to floating point 
    #             converts the range of pixel values from [0,255] to [0,1]
    # optimization for mathematical operations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("Downloading and preparing MNIST dataset...")
    # Getting standard train and test splits to merge them
    # torchvision.datasets.MNIST-->downloads the MNIST dataset
    # root-->directory where the dataset is stored
    # train-->boolean value to specify whether to download the training or testing set
    # download-->boolean value to specify whether to download the dataset
    # transform-->transformation to be applied to the dataset
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Combine both to get all images
    # torch.utils.data.ConcatDataset--> concatenates multiple datasets into a single dataset
    #                                   without their targets
    concat_dataset = torch.utils.data.ConcatDataset([train_data, test_data])
    
    # Extract labels to allow stratified split
    # combine the targets
    # .numpy()--> converts the tensor to a numpy array to perform stratified split
    #               and cause it is easier to work with numpy arrays in sklearn and matplotlib 
    all_targets = torch.cat((train_data.targets, test_data.targets)).numpy()
    # np.arange()--> returns an array of integers from 0 to len(all_targets)-1  
    #               used to create an array of indices to split the dataset
    #the splitting happens on the indices of the dataset, not the dataset itself
    #this is done to maintain the same distribution of classes in each split
    #     to make it faster
    indices = np.arange(len(all_targets))
    
    print("Performing stratified train_test_split (60/20/20)...")
    # First split: 60% train, 40% temp (val + test)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, all_targets, test_size=0.4, stratify=all_targets
    )
    
    # Second split: 50% val, 50% test of the 40% temp -> 20% val, 20% test overall
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp
    )
    
    # Create Subsets from the exact original indices
    # taking from the concatinated dataset the indices that represent each of the sets
    train_dataset = Subset(concat_dataset, train_idx)
    val_dataset = Subset(concat_dataset, val_idx)
    test_dataset = Subset(concat_dataset, test_idx)
    
    # Create DataLoaders --> the engine that feeds the data to the model
    # we shuffle the training data to prevent the model from learning the order of the data
    # drop_last=True--> if the last batch is smaller than the batch size, it will be dropped
    #               this is done to ensures every update to your model is based on the same amount of evidence.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split complete: {len(train_dataset)} Train | {len(val_dataset)} Val | {len(test_dataset)} Test")
    return train_loader, val_loader, test_loader