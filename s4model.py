#%%
import argparse

#Initialize the parameters for the model
parser = argparse.ArgumentParser(description='PyTorch S4 Training')

# Dataset
parser.add_argument('--modelname', default=None, type=str, required = True, help='Name for the model')
parser.add_argument('--dataset', default=None, type=str, required = False, help='Dataset - ["mnist", "cifar10", "customfullpath"]')
parser.add_argument('--trainvaltestsplit', type = float, nargs = 3, default=[0.7, 0.1, 0.2], help='Train, validation and test split [0.7, 0.1, 0.2]')
parser.add_argument('--trainset', default=None, type=str, help='Training Dataset. Provide full path for the dataset.')
parser.add_argument('--valset', default=None, type=str, help='Validation Dataset. Provide full path for the dataset.')
parser.add_argument('--testset', default=None, type=str, help='Test Dataset. Provide full path for the dataset.')                
parser.add_argument('--tabulardata', action = 'store_true', help='Is the dataset tabular(csv)? Needs transformation to tensor and proper dimensions to run the data. Defaults to True.')
parser.add_argument('--dependent_variable', default=None,  type=str, help='Dependent Variables')
parser.add_argument('--independent_variables', default=None, type=list, help='Independent Variables')

# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

# Dataloader
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')

# Model
parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_true', help='Prenorm')

# Scheduler
parser.add_argument('--epochs', default=20, type=int, help='Training epochs')
parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--grayscale', action='store_true', help='Use grayscale')

# Optimizer
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')

args, unknown = parser.parse_known_args()

# if args.dataset is None:
#     raise ValueError("Error: Please provide a 'dataset'. dataset cannot be None")


print('Training Inputs:', args)
print('Unknown Inputs:', unknown)

#%%
#import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, random_split, DataLoader

import torchvision
import torchvision.transforms as transforms

from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import time 
from datetime import datetime
import warnings


#%%

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

if device == 'cuda':
    cudnn.benchmark = True


#%%
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
dropout_fn = nn.Dropout1d # handles regularization

# %%
# Create the custom dataset class for CSV files
# Access data clements using this class
# In sequence models like S4Model, the input is expected to have a sequence length dimension because the model is designed to process sequential data. Even if your data is not sequential (e.g., features from a CSV file), you still need to provide this dimension, typically set to 1.
# If your input shape is (x, y), it means you are missing the sequence length dimension. In sequence models like your S4Model, the expected input shape is typically (B, L, d_input), where:
    # B is the batch size (x in your case).
    # L is the sequence length (usually 1 if you are not using sequential data).
    # d_input is the input dimension (y, representing your y features).

class CSVDataset(Dataset):
    def __init__(self, dataset, x_columns, y_column, transform=None):
        # Load the CSV file into a DataFrame
        self.data = dataset
        
        # Extract features and labels
        self.x = self.data[x_columns].values  # Convert to numpy array
        self.y = self.data[y_column].values   # Convert to numpy array
        
        # Optional: standardize features
        # self.scaler = StandardScaler()
        # self.x = self.scaler.fit_transform(self.x)
        
        # Convert labels to tensors (modify this for regression tasks if needed)
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)  # Use `dtype=torch.float32` for regression
        
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)

        # Unsqueeze to add the sequence length dimension
        x = x.unsqueeze(0)  # Shape becomes [1, 120]
            

        return x, y  # Return features and labels
    
#%%

# Split a given dataset into training, validation and test sets
def split_dataset(dataset, train_split=0.7, val_split=0.10, test_split=0.20, seed=42):
    """
    Splits a given dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_split (float): Proportion of the dataset for training.
        val_split (float): Proportion of the dataset for validation.
        test_split (float): Proportion of the dataset for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Check if the splits add up to 1.0
    total = train_split + val_split + test_split
    if not total == 1.0:
        raise ValueError("Train, validation, and test splits must add up to 1.0")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Calculate lengths for each split
    total_length = len(dataset)
    train_len = int(total_length * train_split)
    val_len = int(total_length * val_split)
    test_len = total_length - train_len - val_len  # Ensure the remainder goes to the test set

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    return train_dataset, val_dataset, test_dataset


#%%
# Data
print(f'==> Preparing data..')

# args.dataset = './data/precipitation/74486094789_tcn_precip_sum3hr_S4_test.csv'
# args.dependent_variable = 'label'
# Example cifar10, mnist datasets or process custom dataset from input

# Function to process image datasets
def process_image_dataset(dataset_name, grayscale):
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(122.6 / 255.0,) if grayscale else (0.4914, 0.4822, 0.4465),
                                 std=(61.0 / 255.0,) if grayscale else (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(1, 1024).t() if grayscale else x.view(3, 1024).t())
        ])

        dataset = torchvision.datasets.CIFAR10
        d_input, d_output = (1, 10) if grayscale else (3, 10)

    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(1, 784).t())
        ])

        dataset = torchvision.datasets.MNIST
        d_input, d_output = 1, 10
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
    
    trainset = dataset(root=f"./data/{dataset_name}", train=True, download=True, transform=transform)
    trainset, valset, _ = split_dataset(trainset, train_split=0.9, val_split=0.1, test_split=0)
    testset = dataset(root=f"./data/{dataset_name}", train=False, download=True, transform=transform)

    df, train_df, test_df = None, None, None
    
    return trainset, valset, testset, d_input, d_output, df, train_df, test_df

# Function to process tabular datasets
def process_tabular_data(dataset_path, dependent_variable, independent_variables):
    if dependent_variable is None:
        raise ValueError("Error: Please provide a dependent variable. Dependent variable cannot be None.")
    
    df = pd.read_csv(dataset_path).fillna(0)
    y_column = dependent_variable
    x_columns = independent_variables if independent_variables else [x for x in df.columns if x != y_column]
    
    csv_dataset = CSVDataset(df, x_columns, y_column)   
    
    d_input = len(x_columns)
    d_output = df[y_column].nunique()
    
    print(f"Input Features = {d_input}; Output Classes = {d_output}; Total Rows = {len(df)}")
    
    return csv_dataset, d_input, d_output, df

def dataprocessing(args):
    if args.dataset:
        if args.dataset in ["cifar10", "mnist"]:
            trainset, valset, testset, d_input, d_output = process_image_dataset(args.dataset, args.grayscale)
        elif args.tabulardata:
            trainset, valset, testset = None, None, None
            csv_dataset, d_input, d_output, df = process_tabular_data(args.dataset, args.dependent_variable, args.independent_variables)
            trainset, valset, testset = split_dataset(csv_dataset, train_split=args.trainvaltestsplit[0], val_split=args.trainvaltestsplit[1], test_split=args.trainvaltestsplit[2])
            train_df, test_df = None, None
        else:
            raise NotImplementedError("Dataset type not recognized.")
    elif args.trainset and args.testset:
        if not args.tabulardata:
            raise NotImplementedError("Only tabular data is supported for train/val/test splits.")
        
        df = None
        trainset, d_input, d_output, train_df = process_tabular_data(args.trainset, args.dependent_variable, args.independent_variables)
        testset, _, _, test_df = process_tabular_data(args.testset, args.dependent_variable, args.independent_variables)
        
        if not args.valset:
            warnings.warn("Warning: Validation dataset is not provided. Splitting training dataset 80/20 for validation.")
            trainset, valset, train_df = split_dataset(trainset, train_split=0.8, val_split=0.2, test_split=0)
        else:
            valset, _, _, val_df = process_tabular_data(args.valset, args.dependent_variable, args.independent_variables)        
    else:
        if args.dataset == None and args.trainset == None and args.testset == None and args.valset == None:
            raise ValueError("Error: Please provide a dataset.")
        else:
            raise ValueError("Error: Both trainset and testset must be provided.")
    
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return trainloader, valloader, testloader, d_input, d_output, trainset, valset, testset, df, train_df, test_df
    
trainloader, valloader, testloader, d_input, d_output, trainset, valset, testset, df, train_df, test_df = dataprocessing(args)

#%%
# S4Model class to run the S4 model - This is where the S4D is being used
class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        transposed = True,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=transposed, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

#%%

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)


model = model.to(device)
print(model)

#%%

#Checkpoints
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.modelname}ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

#%%

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)


#%%
###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        # print(f"Batch input shape: {inputs.shape}, Batch target shape: {targets.shape}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
   
    output_list_target = [] # To store target data
    output_list_predicted = [] # To store predicted data   
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()           

            output_list_target.extend(targets.cpu().numpy())  # Collect targets
            output_list_predicted.extend(predicted.cpu().numpy())  # Collect predictions                         

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )           
            
            

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/{args.modelname}ckpt.pth')
            best_acc = acc

        return acc, output_list_target, output_list_predicted


#%%    
# Reiterate thorugh epochs to train and validate the model
# 1st Batch - Train, # 2nd Batch - Validation, # 3rd Batch - Test
# args.epochs = 3
print('Start Time: ',datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
start_time = time.perf_counter()
pbar = tqdm(range(start_epoch,args.epochs))
for epoch in pbar:    
    train()
    val_acc, val_target, val_predicted = eval(epoch, valloader, checkpoint=True)
    test_acc, test_target, test_predicted = eval(epoch, testloader, checkpoint=True)    
    # print(test_acc)
    scheduler.step()
    # if epoch == 0:
    #     pbar.set_description('Epoch: %d' % (epoch))
    # else:
    pbar.set_description('Epoch: %d | Val acc: %.3f%% | Test acc: %.3f%%' % (epoch, val_acc, test_acc))

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print('End Time: ', datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
print(f"Process completed in {elapsed_time:.6f} seconds")  # Higher precision timing    

#%%
def combine_results_to_dataframe(dataloader, target_list, predicted_list, dependent_variable =None, valset=None, original_df=None, name = None):
    """
    Combines input features from a PyTorch DataLoader with target and predicted values into a DataFrame.
    Saves the results as a CSV file in the './results/' directory with a timestamped filename.

    Parameters:
        dataloader (torch.utils.data.DataLoader): Dataloader containing input features.
        target_list (list): List of target values.
        predicted_list (list): List of predicted values.
        dependent_variable (str): Name of the dependent variable to be removed from column names.
        valset (Dataset): Original validation dataset (if available).
        original_df (pd.DataFrame): Original DataFrame used to create the dataset.
        name (str): Name identifier for the output file.

    Returns:
        pd.DataFrame: Combined DataFrame with input features, target, and predicted values.
    """

    # Extract input features from dataloader
    input_list = []
    for inputs, _ in dataloader:  # Ignore targets since we have target_list separately
        batch_size = inputs.size(0)
        inputs_flat = inputs.view(batch_size, -1).cpu().numpy()  # Flatten input tensors
        input_list.extend(inputs_flat)  # Append batch data to list

    # Convert input data to DataFrame
    df_results = pd.DataFrame(input_list)

    # Determine column names while removing the dependent variable
    if hasattr(valset, 'columns'):  
        df_results.columns = [col for col in valset.columns.tolist() if col != dependent_variable]
    elif isinstance(original_df, pd.DataFrame):  
        df_results.columns = [col for col in original_df.columns.tolist() if col != dependent_variable][:df_results.shape[1]]
    else:  
        df_results.columns = [f'feature_{i}' for i in range(df_results.shape[1])]

    # Add the target and predicted values
    if dependent_variable is not None:
        target = dependent_variable
    else:
        target = 'Target'
    
    # Add the target and predicted values
    df_results[target] = target_list
    df_results["Predicted"] = predicted_list

    # Ensure the results directory exists
    results_dir = "./results/"
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
    filename = f"{results_dir}{timestamp}_{name}_results.csv"

    # Save DataFrame to CSV
    df_results.to_csv(filename, index=False)

    print(f"Dataset successfully saved as '{filename}'")

    return df_results

#%%
# args.modelname = 's4test'
#Save the validation and test datasets
if args.dataset == 'cifar10' or args.dataset == 'mnist':    
    dep_var = None
    val_name = None
    test_name = None
else:    
    dep_var = args.dependent_variable    
    val_name = train_df if df is None else df
    test_name = test_df if df is None else df


df_validation = combine_results_to_dataframe(valloader, val_target, val_predicted, dependent_variable =dep_var, valset=valset, original_df=val_name, name = f'{args.modelname}_Validation')
df_test = combine_results_to_dataframe(testloader, test_target, test_predicted, dependent_variable =dep_var, valset=testset, original_df=test_name, name = f'{args.modelname}_Test')
# %%
