#%%
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import torchvision
import warnings

#%%
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
        # Extract features and labels. convert to numpy array
        self.x = torch.tensor(self.data[x_columns].values, dtype=torch.float32)
        self.y = torch.tensor(self.data[y_column].values, dtype=torch.long) # Use `dtype=torch.float32` for regression

        # Optional: standardize features
        # self.scaler = StandardScaler()
        # self.x = self.scaler.fit_transform(self.x)

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
         
         if self.transform:
            return self.transform(self.x[idx]).unsqueeze(0), self.y[idx]
         
         return self.x[idx].unsqueeze(0), self.y[idx]  # Add sequence length dimension

#%%
# Split a given dataset into training, validation and test sets
def split_dataset(dataset, splits=[0.7, 0.1, 0.2], seed = 42):
    """
    Splits a given dataset into training, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        splits (lists of float): Proportion of the dataset for training, validation and testing.       
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """ 

    # Check if the splits add up to 1.0
    if not sum(splits) == 1.0:
        raise ValueError("Train, validation, and test splits must add up to 1.0")
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    train_len, val_len = int(len(dataset) * splits[0]), int(len(dataset) * splits[1])
    test_len = len(dataset) - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])

#%%
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
    
    trainset = dataset(root=f"../data/{dataset_name}", train=True, download=True, transform=transform)
    trainset, valset, _ = split_dataset(trainset, splits=[0.9, 0.1, 0])
    testset = dataset(root=f"../data/{dataset_name}", train=False, download=True, transform=transform)

    df, train_df, test_df = None, None, None
    
    return trainset, valset, testset, d_input, d_output, df, train_df, test_df

#%%
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

#%%
def load_data(args):
    if args.dataset:
        if args.dataset in ["cifar10", "mnist"]:
            trainset, valset, testset, d_input, d_output, df, train_df, test_df = process_image_dataset(args.dataset, args.grayscale)
        elif args.tabulardata:
            trainset, valset, testset = None, None, None
            csv_dataset, d_input, d_output, df = process_tabular_data(args.dataset, args.dependent_variable, args.independent_variables)
            trainset, valset, testset = split_dataset(csv_dataset, splits = args.trainvaltestsplit)
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