import matplotlib.pyplot as plt
# seaborn-->statistical data visualization library
# used to create heatmaps-->graphical representation of data
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_history(history, experiment_name, save_dir='plots',epochs=5):
    # creates a directory to save the plots
    # exist_ok=True-->if the directory already exists, it will not raise an error
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, epochs + 1)
    
    # Plot Loss
    #10-->width of the plot
    #5-->height of the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Loss vs Epochs - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_loss.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Accuracy vs Epochs - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_accuracy.png'))
    plt.close()

def plot_confusion_matrix(labels, preds, experiment_name, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    # heatmap-->visualizes the confusion matrix
    # annot=True-->show the values in the cells
    # fmt='d'-->format the values as integers
    # cmap='Blues'-->use the blues colormap
    # xticklabels=np.arange(10)-->set the x-axis labels to 0-9
    # yticklabels=np.arange(10)-->set the y-axis labels to 0-9
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, f'{experiment_name}_confusion_matrix.png'))
    plt.close()
