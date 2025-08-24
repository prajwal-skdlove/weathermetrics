#%%
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import torchvision
import warnings
import pyarrow.parquet as pq
import logging
import pandas as pd
import polars as pl
import os

# -------------------------------------------------------------------------
# Custom dataset for CSV/tabular data
# -------------------------------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, dataset, x_columns, y_column=None, modeltype=None, transform=None, inference=False):
        """
        Args:
            dataset (pd.DataFrame): DataFrame containing the data.
            x_columns (list): List of feature column names.
            y_column (str, optional): Target column. Not required for inference.
            modeltype (str, optional): 'classification' or 'regression'.
            transform (callable, optional): Transform applied to x.
            inference (bool): If True, y is not expected.
        """
        try:
            self.data = dataset
            self.x = torch.tensor(self.data[x_columns].values, dtype=torch.float32)
            self.transform = transform
            self.inference = inference

            if not inference and y_column is not None and modeltype is not None:
                if modeltype == "classification":
                    self.y = torch.tensor(self.data[y_column].values, dtype=torch.long)
                elif modeltype == "regression":
                    self.y = torch.tensor(self.data[y_column].values, dtype=torch.float32)
                else:
                    raise ValueError("modeltype must be 'classification' or 'regression'")
            else:
                self.y = None

            logging.info(f"Initialized CSVDataset with {len(self.data)} rows and {len(x_columns)} features.")

        except Exception as e:
            logging.exception(f"Error initializing CSVDataset: {e}")
            raise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        try:
            x_item = self.x[idx].unsqueeze(0)
            if self.transform:
                x_item = self.transform(x_item)
            if self.inference or self.y is None:
                return x_item
            else:
                return x_item, self.y[idx]
        except Exception as e:
            logging.exception(f"Error retrieving item at index {idx}: {e}")
            raise

# -------------------------------------------------------------------------
# Split dataset
# -------------------------------------------------------------------------
def split_dataset(dataset, splits=[0.7, 0.1, 0.2], seed=42):
    try:
        if not abs(sum(splits) - 1.0) < 1e-6:
            raise ValueError(f"Train/val/test splits must sum to 1.0. Got {splits}")

        torch.manual_seed(seed)
        train_len, val_len = int(len(dataset) * splits[0]), int(len(dataset) * splits[1])
        test_len = len(dataset) - train_len - val_len

        logging.info(f"Splitting dataset of {len(dataset)} samples into: "
                     f"{train_len} train, {val_len} val, {test_len} test.")
        return random_split(dataset, [train_len, val_len, test_len])
    except Exception as e:
        logging.exception(f"Error splitting dataset: {e}")
        raise

# -------------------------------------------------------------------------
# Split Time Series dataset
# -------------------------------------------------------------------------

def split_time_series_data(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Splits a time series dataset into training, validation, and test sets.
    Works with pandas.DataFrame, polars.DataFrame, and polars.LazyFrame.
    Uses lazy evaluation for polars LazyFrame.

    Parameters:
        df (pd.DataFrame | pl.DataFrame | pl.LazyFrame): Input dataset, ordered in descending dates.
        train_ratio (float): Proportion for training.
        val_ratio (float): Proportion for validation.
        test_ratio (float): Proportion for testing.

    Returns:
        tuple: (train, validation, test) of the same type as `df`.
    """
    try:
        logging.info("Starting time series split: train=%.2f, val=%.2f, test=%.2f",
                    train_ratio, val_ratio, test_ratio)

        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio:.4f})")

        # --- Handle pandas ---
        if isinstance(df, pd.DataFrame):
            n = len(df)
            if n == 0:
                raise ValueError("Input pandas DataFrame is empty")
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_data = df.iloc[:train_end].sort_index(ascending=True)
            val_data = df.iloc[train_end:val_end].sort_index(ascending=True)
            test_data = df.iloc[val_end:].sort_index(ascending=True)

            logging.info("Split pandas DataFrame: train=%d, val=%d, test=%d",
                        len(train_data), len(val_data), len(test_data))
            logging.debug("Train indices: 0-%d, Val indices: %d-%d, Test indices: %d-%d",
                         train_end-1, train_end, val_end-1, val_end, n-1)
            return train_data, val_data, test_data

        # --- Handle polars DataFrame ---
        elif isinstance(df, pl.DataFrame):
            n = df.height
            if n == 0:
                raise ValueError("Input Polars DataFrame is empty")
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_data = df.slice(0, train_end)
            val_data = df.slice(train_end, val_end - train_end)
            test_data = df.slice(val_end, n - val_end)

            logging.info("Split Polars DataFrame: train=%d, val=%d, test=%d",
                        train_data.height, val_data.height, test_data.height)
            return train_data, val_data, test_data

        # --- Handle polars LazyFrame ---
        elif isinstance(df, pl.LazyFrame):
            n = df.collect().height
            if n == 0:
                raise ValueError("Input Polars LazyFrame is empty")
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)

            train_data = df.slice(0, train_end)
            val_data = df.slice(train_end, val_end - train_end)
            test_data = df.slice(val_end, n - val_end)

            logging.info("Prepared lazy splits for Polars LazyFrame (rows=%d)", n)
            logging.debug("LazyFrame split indices: train=0-%d, val=%d-%d, test=%d-%d",
                         train_end-1, train_end, val_end-1, val_end, n-1)
            return train_data, val_data, test_data

        else:
            raise TypeError("Input must be a pandas.DataFrame, polars.DataFrame, or polars.LazyFrame")

    except Exception as e:
        logging.exception("Failed to split time series data: %s", e)
        raise

# -------------------------------------------------------------------------
# Process image datasets
# -------------------------------------------------------------------------
def process_image_dataset(dataset_name, grayscale):
    try:
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

        logging.info(f"Loaded image dataset {dataset_name} (grayscale={grayscale}, "
                     f"input={d_input}, output={d_output}).")

        return trainset, valset, testset, d_input, d_output, None, None, None
    except Exception as e:
        logging.exception(f"Error processing image dataset {dataset_name}: {e}")
        raise

# -------------------------------------------------------------------------
# Process tabular datasets
# -------------------------------------------------------------------------
def process_tabular_data(dataset_path, dependent_variable, independent_variables=None, modeltype="regression"):
    """
    Efficiently processes a tabular dataset for modeling using Polars.
    Handles CSV and Parquet files, fills nulls, selects features, and computes input/output dimensions.

    Parameters:
        dataset_path (str): Path to CSV or Parquet file.
        dependent_variable (str): Name of the target column.
        independent_variables (list, optional): List of feature columns. Defaults to all except target.
        modeltype (str): "classification" or "regression".

    Returns:
        tuple: (dataset, d_input, d_output, df, x_columns)
    """
    try:
        if dependent_variable is None:
            raise ValueError("Dependent variable must be provided.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        logging.info(f"Loading dataset: {dataset_path}")

        # Efficient reading using Polars
        if dataset_path.endswith(".csv"):
            df = pl.read_csv(dataset_path).fill_null(0)
        elif dataset_path.endswith(".parquet"):
            df = pl.read_parquet(dataset_path, use_pyarrow=True).fill_null(0)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        n_rows = df.height
        logging.info(f"Loaded dataset with {n_rows} rows and {len(df.columns)} columns.")

        # Determine feature columns
        x_columns = independent_variables if independent_variables else [c for c in df.columns if c != dependent_variable]

        if dependent_variable not in df.columns:
            raise ValueError(f"Dependent variable '{dependent_variable}' not found in dataset columns.")

        # Convert Polars DataFrame to pandas for CSVDataset if required
        pdf = df.to_pandas()

        # Initialize dataset
        csv_dataset = CSVDataset(pdf, x_columns, dependent_variable, modeltype)
        d_input = len(x_columns)

        # Determine output dimension
        if modeltype == "classification":
            d_output = pdf[dependent_variable].nunique()
        elif modeltype == "regression":
            d_output = 1
        else:
            raise ValueError(f"Invalid modeltype: {modeltype}")

        logging.info(f"Processed dataset: {n_rows} rows, {d_input} features, {d_output} outputs.")

        return csv_dataset, d_input, d_output, pdf, x_columns

    except Exception as e:
        logging.exception(f"Error processing tabular data '{dataset_path}': {e}")
        raise

# -------------------------------------------------------------------------
# Load data entrypoint
# -------------------------------------------------------------------------
def load_data(args):
    try:
        if args.dataset:
            if args.dataset in ["cifar10", "mnist"]:
                trainset, valset, testset, d_input, d_output, df, train_df, test_df = \
                    process_image_dataset(args.dataset, args.grayscale)
                x_columns = None
            elif args.tabulardata:
                csv_dataset, d_input, d_output, df, x_columns = \
                    process_tabular_data(args.dataset, args.dependent_variable, args.independent_variables, args.modeltype)
                trainset, valset, testset = split_dataset(csv_dataset, splits=args.trainvaltestsplit)
                train_df, test_df = None, None
            else:
                raise NotImplementedError("Dataset type not recognized.")
        elif args.trainset and args.testset:
            if not args.tabulardata:
                raise NotImplementedError("Only tabular data supported for train/val/test splits.")

            df = None
            trainset, d_input, d_output, train_df, x_columns = \
                process_tabular_data(args.trainset, args.dependent_variable, args.independent_variables, args.modeltype)
            testset, _, _, test_df, _ = \
                process_tabular_data(args.testset, args.dependent_variable, args.independent_variables, args.modeltype)

            if not args.valset:
                warnings.warn("Validation dataset not provided. Splitting training dataset 80/20.")
                trainset, valset, _ = split_dataset(trainset, splits=[0.8, 0.2, 0.0])
            else:
                valset, _, _, val_df, _ = \
                    process_tabular_data(args.valset, args.dependent_variable, args.independent_variables, args.modeltype)
        else:
            raise ValueError("No valid dataset arguments provided.")

        # Dataloaders
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        logging.info("DataLoaders created successfully.")

        return trainloader, valloader, testloader, d_input, d_output, trainset, valset, testset, df, None, None, x_columns
    except Exception as e:
        logging.exception(f"Error in load_data: {e}")
        raise

# -------------------------------------------------------------------------
# Sanity check utility for loaders
# -------------------------------------------------------------------------
def verify_loaders(trainloader, valloader, testloader, modeltype="classification"):
    """
    Verify DataLoaders by logging batch shapes and a sample of targets.

    Args:
        trainloader, valloader, testloader: DataLoader objects
        modeltype (str): "classification" or "regression"
    """
    try:
        for name, loader in [("Train", trainloader), ("Validation", valloader), ("Test", testloader)]:
            if loader is None:
                logging.warning(f"{name} loader is None. Skipping verification.")
                continue

            try:
                batch = next(iter(loader))
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                    logging.info(f"{name} loader: X batch shape = {tuple(x.shape)}, "
                                 f"Y batch shape = {tuple(y.shape)}")
                    if modeltype == "classification":
                        unique_classes = torch.unique(y)
                        logging.info(f"{name} targets sample: {y[:10].tolist()} "
                                     f"(unique classes in batch: {unique_classes.tolist()})")
                    elif modeltype == "regression":
                        logging.info(f"{name} targets sample (first 10): {y[:10].tolist()}")
                else:
                    x = batch
                    logging.info(f"{name} loader (inference mode): X batch shape = {tuple(x.shape)}")

            except StopIteration:
                logging.warning(f"{name} loader is empty.")
    except Exception as e:
        logging.exception(f"Error verifying loaders: {e}")
        raise
