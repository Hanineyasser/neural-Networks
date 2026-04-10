import torch
import copy
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """
    Custom training loop per the assignment requirements. 
    """
    model.to(device)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f"Training on {device} for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects.double() / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc.item())
        
        # Validation Phase
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
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
    # Load best model weights based on validation performance
    model.load_state_dict(best_model_wts)
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s - Best val Acc: {best_acc:.4f}')
    
    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    
    return model, history

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates mode on the test_loader, returning accuracy and list of preds/labels.
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    running_corrects = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_test += inputs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = running_corrects.double() / total_test
    return accuracy.item(), all_labels, all_preds
