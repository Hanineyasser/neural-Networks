# Handwritten Digit Recognition System (MNIST)

A robust machine learning project built in PyTorch to classify handwritten digits from the MNIST dataset. This project includes a modular pipeline for data loading, a dynamic Feedforward Neural Network (FNN), and a Convolutional Neural Network (CNN). 

It features a comprehensive hyperparameter analysis covering learning rates, batch sizes, neuron configurations, and network depth.

## 📁 Project Structure

* **`main.py`**: The main execution script. It runs the entire suite of experiments, trains the models, and saves the results.
* **`models.py`**: Contains the neural network architectures.
  * `FNN`: A dynamic Feedforward Neural Network that allows for customizable layer sizes.
  * `BonusCNN`: A Convolutional Neural Network built with `Conv2d`, `LayerNorm`, `MaxPool2d`, and `Dropout` for superior spatial feature extraction.
* **`trainer.py`**: Contains the training loop, validation logic, and model evaluation (`train_model`). Handles PyTorch gradient tracking and deep-copying the best model weights.
* **`data_loader.py`**: Handles downloading the MNIST dataset, applying transformations (`transforms.ToTensor()`), and performing a stratified 60/20/20 train/val/test split.
* **`utils.py`**: Contains helper functions for visualizing results (e.g., plotting confusion matrices).
* **`results_summary.json`**: Auto-generated JSON file containing the final test accuracies for all experiments.

## 🛠️ Dependencies

Ensure you have the following installed:
* Python 3.x
* PyTorch (`torch`, `torchvision`)
* NumPy
* Scikit-Learn
* Matplotlib / Seaborn

Install dependencies using:
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

## 🚀 How to Run

Simply execute the `main.py` script. The script will automatically download the dataset (if not present), run all experiments, generate plots in the `plots/` directory, and output the results to `results_summary.json`.

```bash
python main.py
```

## 📊 Experiments & Key Findings

We conducted several experiments to analyze the impact of hyperparameters on model performance. 

### 1. Learning Rate Analysis
* **Best:** `LR = 0.1` achieved **96.3%** accuracy.
* **Worst:** `LR = 0.001` achieved only **51.7%** accuracy. 
* **Conclusion:** Too small of a learning rate prevents the optimizer from adjusting the weights enough within 5 epochs.

### 2. Batch Size Analysis
* **Best:** `Batch Size = 16` achieved **94.0%** accuracy.
* **Trend:** Smaller batch sizes resulted in higher accuracy in this setup, as they provide more frequent weight updates per epoch compared to larger batches like 256 (76.1%).

### 3. Network Depth (Hidden Layers)
* Adding more layers did *not* improve performance for the FNN. 
* Moving from 3 hidden layers (87.4%) down to 6 hidden layers (11.2%) caused the accuracy to completely crash (likely due to the vanishing gradient problem).

### 4. CNN vs. FNN
* **FNN (Base):** 90.0% accuracy.
* **Bonus CNN:** **97.8%** accuracy. 
* **Conclusion:** The Convolutional Neural Network significantly outperformed the FNN. By utilizing a 3x3 sliding kernel for spatial awareness, `MaxPool2d` for translation invariance, and `Dropout` to prevent overfitting, the CNN proved to be the superior architecture for image classification.
