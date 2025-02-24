# S4 Model Training Framework

## Overview
This repository contains a modular implementation of the S4 model training framework. It supports tabular and image datasets, integrates PyTorch for deep learning, and includes functionalities for dataset processing, training, evaluation, and result logging. This modifies the official implementation and experiments related to S4, including HiPPO, LSSL, SaShimi, DSS, HTTYH, S4D and S4ND from Albert Gu, Karen Goel, Khaled Saab and Chris Re. https://github.com/state-spaces/s4

## File Structure
```
../data/                      # Folder to store data for the models
../checkpoint/                # Folder to save checkpoints for the model training and final model
../results/                   # Folder to save results
s4model/
│── s4model.py                # Entry point for the script
│── config.py                 # Argument parsing and configuration
│── dataset.py                # Dataset handling and preprocessing
│── model.py                  # S4Model definition
│── train.py                  # Training and evaluation functions
│── output.py                 # Output handling
│── infer.py                  # Model Inference
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
src/                          # Utility functions that is called by the orignal models
models/                       # Original solid state modles - S4 including HiPPO, LSSL, SaShimi, DSS, HTTYH, S4D and S4ND
```

## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
To train the model, run:
```sh
python -m main --dataset cifar10 --modelname my_s4_model --epochs 20 --batch_size 64
```
For tabular data:
```sh
python -m main --dataset ../data/mydata.csv --modelname my_s4_model --tabulardata --dependent_variable label --independent_variables feature1 feature2 feature3
```

### Running Inference
After training, you can run inference on new data and save the results to a CSV file:
```sh
python infer.py --dataset new_data.csv --modelname my_s4_model --independent_variables feature1 feature2 feature3
```
The results will be saved in `../results/` as a CSV file.


## Arguments
| Argument | Description |
|----------|-------------|
| `--modelname` | Model name for saving checkpoints |
| `--dataset` | Dataset name or path (mnist, cifar10, or custom CSV) |
| `--trainvaltestsplit` | Train-validation-test split ratios |
| `--trainset` | Training dataset |
| `--valset` | Validation dataset |
| `--testset` | Test dataset |
| `--tabulardata` | Indicates dataset is tabular (CSV) |
| `--dependent_variable` | Dependent variable |
| `--independent_variables` | Independent variable |
| `--resume` | Resume from checkpoint |
| `--num_workers` | Number of subproceesses used for data loading in Dataloader. Higher number cna speed up data loading but cause increased memory usage |
| `--batch_size` | Batch size for training. Number of sample processed together |
| `--n_layers` | Number of layers in an neural network, such as convoluntional layers, fully conneted layers, or recurrent layers |
| `--d_model` | Model dimension - size of feature representations in the model. Larger dimensions allow the model to capture more complex patterns |
| `--dropout` | Dropout - Regularization technique where random neurons are dropped (set to zero) to prevent overfitting |
| `--prenorm` | Enable prenorm - applying layer nomralization before operations |
| `--epochs` | Number of training epochs - complete passes through the training dataset |
| `--patience` | Patience for learning rate schedulers used in early stopping during training. Specifies how many epochs to wait without improvement before halting |
| `--grayscale` | Use grayscale for images |
| `--lr` | Learning rate - step size used by an optimizer |
| `--weight_decay` | Weight Decay - regularization technique to reduce overfitting by penalizing large weights in the model |

## Features
- Supports image datasets (`CIFAR-10`, `MNIST`) and tabular datasets (CSV).
- Modular design for easy extensibility.
- Implements training, validation, and testing routines.
- Saves and logs model performance.
- Saves inference results to `../results/` folder.


## Model Architecture
The `S4Model` consists of:
- A linear encoder layer.
- Multiple stacked `S4D` layers for sequence modeling.
- A final linear decoder layer.

## Contributors
- **Prajwal Koirala**
- **John Hughes**

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE.