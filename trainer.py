import torch
import copy
import time
# criterion is the loss that we wanna calculate
# optimizer is the algorithm that we wanna use to update the weights
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cpu'):
    # hardware management
    # model.to(device)--> moves the model to the device specified passed through the parameters
    # if device is 'cuda', it moves the model from the CPU to the GPU
    # if device is 'cpu', it moves the model to the CPU or keeps it in the CPU
    model.to(device)
    #losses
    train_losses, val_losses = [], []
    #accuracies
    train_accs, val_accs = [], []
    # making a deep copy of the weights of the best model to keeep update 
    # the best model we have to arrive to the very best at the end of all the iterations
    # state_dict()-->returns a dictionary containing the weights of the model
                    # dictionarey of ther oparameteres--> weights and bias
    best_model_weights = copy.deepcopy(model.state_dict())
    #initializing the best accuracy to 0.0
    best_acc = 0.0
    
    print(f"Training on {device} for {num_epochs} epochs...")
    #to calculate the time we needed to finish the training of this model
    start_time = time.time()
    #epoch-->pass
    for epoch in range(num_epochs):
        # built in function in the torch library to set the model to training mode
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        total_train = 0
        # inputs-->images = features --> for each batch
        # labels-->ground truth --> t --> y
        for inputs, labels in train_loader:
            # the whole date (features and labels )to the device used (CPU or GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            # this is done to prevent the accumulation of gradients from previous iterations
            # making optimizer = 0, cleaning the memory
            optimizer.zero_grad()
            # forward pass-->getting the outputs
            outputs = model(inputs)
            # calculating the loss-->cross entropy
            # returns an average loss over the batch
            loss = criterion(outputs, labels)
            # getting the predictions
            # torch.max-->returns the maximum value and the index of the maximum value
            # When you run outputs = model(inputs), the neural network doesn't 
            # just spit out a single number like "7". 
            # Because the output layer has 10 neurons, the network spits out 
            # an array of 10 different raw point scores for each image.
            # 1-->dim=1 means we are looking for the max in each row
            _, preds = torch.max(outputs, 1)
            # backward pass-->backpropagation
            # calculating the gradients
            # built in
            loss.backward()
            # updating the weights to arrive to the new weights 
            # built in
            optimizer.step()
            # sum up the loss and the number of correct predictions
            #.item()-->returns the value of the loss-->takes the decimal number out of the tensor
            # scalar number
            # it is like a weighted sum (loss * batch_size)
            # so that the loss is not affected by the batch size
            # example:::
                    #Class A has 10 students. The average score is 80.
                    #Class B has 50 students. The average score is 90.
                    #Class C has 5 students. The average score is 70.
                    # If you just blindly added the averages together without
                    # multiplying (80 + 90 + 70 = 240) and divided by 3 classes, 
                    # you would think the school average is 80. 
                    # But that is wrong! Class B has 50 students; their 90% average 
                    # should carry way more weight than Class C's tiny class of 5 students!  
                    #Class A: 80 (Average) * 10 (Students) = 800 total points
            running_loss += loss.item() * inputs.size(0)
            # sum up the number of correct predictions --> if the label we have is the same as the prediction we got from our calculations
            # torch.sum-->returns the sum of the correct guessed elements in the tensor
            running_corrects += torch.sum(preds == labels.data)
            # total number of training examples to arrive to the total training number of elements at the very last batch of inputs
            total_train += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train
        train_losses.append(epoch_train_loss)
        # .item() is used to get the value of the tensor cause we didnt get it out from it at the beginning like we did with the loss
        train_accs.append(epoch_train_acc.item())
        
        # built in function in the torch library to set the model to evaluation mode
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
                
        epoch_val_loss = running_loss / total_val
        epoch_val_acc = running_corrects.double() / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.item())
        # this protects the model from an overfitting 
        # cause we wanna see how good the model can generalize what it hads learnt
        # if the validation accuracy is decreasing, it means the model is overfitting
        # so we stop training
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            
    # Load best model weights based on validation performance
    # Forget everything you learned in the last few epochs where you started 
    # over-fitting. Go back to that specific version of yourself that was the 
    # most intelligent and generalized best to new data.
    # load_state_dict-->loads the weights of the model-->with the parameters u pass not the last parameters ther model had to prevent the overfitting
    model.load_state_dict(best_model_weights)
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s - Best val Acc: {best_acc:.4f}')
    
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    
    return model, history
# evaluate_model--> evaluates the model on the test_loader
def evaluate_model(model, test_loader, device='cpu'):
    # moves the model to the device(CPU or GPU) specified passed through the parameters
    model.to(device)
    # built in function in the torch library to set the model to evaluation mode
    model.eval()
    
    all_preds = []
    all_labels = []
    
    running_corrects = 0.0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            #inputs.size(0)-->number of inputs in the batch
            # 0-->batch_size 1-->channels 2-->height 3-->width
            total_test += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = running_corrects / total_test
    return accuracy.item(), all_labels, all_preds
