#%%
import os
import time
import logging
from datetime import datetime
import argparse
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

# from config import get_args
from dataset import load_data, verify_loaders
from output import combine_results_to_dataframe
from model import S4Model, setup_optimizer
from train import train, eval, load_model

#%%
def setup_logging(log_dir="logs", log_file="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

#%%
def get_args():
    args = argparse.Namespace()
    args.modelname = "debug_model"
    args.modeltype = "regression"  # or 'regression' 'classification'
    args.dataset = None
    # args.traintestvalsplit = [0.7, 0.1, 0.2]
    args.trainset = "D:/CodeLibrary/Python/weathermetrics/data/weathermetrics/72206013889_DAYSUM_bin1_5_20_train.parquet"
    args.valset = "D:/CodeLibrary/Python/weathermetrics/data/weathermetrics/72206013889_DAYSUM_bin1_5_20_validation.parquet"   
    args.testset = "D:/CodeLibrary/Python/weathermetrics/data/weathermetrics/72206013889_DAYSUM_bin1_5_20_test.parquet"
    args.batch_size = 64
    args.num_workers = 0 
    args.n_layers = 4
    args.d_model = 128
    args.dropout = 0.1
    args.prenorm = False
    args.epochs = 1
    args.lr = 0.01
    args.weight_decay = 0.01
    args.grayscale = False
    args.tabulardata = True
    args.dependent_variable = "tgt_bin"
    args.independent_variables = None
    args.resume = False
    args.output_data = False
    unknown = []
    return args, unknown


#%%
def main():
    # Set up logging
    setup_logging()

    # Parse arguments
    args, unknown = get_args()

    logging.info(f"Training Inputs: {args}")
    logging.info(f"Unknown Inputs: {unknown}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = device == "cuda"
    logging.info(f"Using {device} device")

    best_acc = 0
    start_epoch = 0

    # Load dataset
    logging.info("Preparing data...")
    trainloader, valloader, testloader, d_input, d_output, trainset, valset, testset, df, train_df, test_df, args.independent_variables = load_data(args)
    verify_loaders(trainloader, valloader, testloader, modeltype=args.modeltype)
    args.d_input = d_input
    args.d_output = d_output

    # Initialize model, optimizer, and loss function
    logging.info("Building model...")
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        lr=args.lr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm
    ).to(device)
    logging.info(f"Model architecture:\n{model}")

    if args.modeltype == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif args.modeltype == 'regression':
        criterion = nn.MSELoss()
    else:
        logging.error(f"Unknown model type: {args.modeltype}")
        return

    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
    )

    # Resume from checkpoint if specified
    if getattr(args, "resume", False):
        logging.info("Resuming from checkpoint...")
        model, model_args, best_acc, start_epoch = load_model(S4Model, args.modelname)
    # Log all arguments except 'independent_variables'
    args_to_log = {k: v for k, v in vars(args).items() if k != "independent_variables"}
    logging.info(f"Arguments (excluding independent_variables): {args_to_log}")

    # Training loop
    logging.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    start_time = time.perf_counter()
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training")

    for epoch in pbar:
        avg_loss, acc, train_target, train_predicted, train_prob_list, train_input_list = train(
            model, trainloader, criterion, optimizer, device, modeltype=args.modeltype)
        if args.modeltype == "classification":
            pbar.set_description(f"Epoch: {epoch + 1} | Train acc: {acc:.3f}%")
        elif args.modeltype == "regression":
            pbar.set_description(f"Epoch: {epoch + 1} | Train MSE: {avg_loss:.3f}")
        val_metric, val_target, val_predicted, val_prob_list, val_input_list = eval(
            model, valloader, criterion, device, epoch, args.modelname, best_acc, args,
            modeltype=args.modeltype, checkpoint=True
        )
        test_metric, test_target, test_predicted, test_prob_list, test_input_list = eval(
            model, testloader, criterion, device, epoch, args.modelname, best_acc, args,
            modeltype=args.modeltype, checkpoint=True
        )
        scheduler.step()

        if args.modeltype == "classification":
            pbar.set_description(f"Epoch: {epoch + 1} | Val acc: {val_metric:.3f}% | Test acc: {test_metric:.3f}%")
        elif args.modeltype == "regression":
            pbar.set_description(f"Epoch: {epoch + 1} | Val MSE: {val_metric:.3f} | Test MSE: {test_metric:.3f}")

        logging.info(
            f"Epoch {epoch + 1}: Val metric: {val_metric:.4f}, Test metric: {test_metric:.4f}"
        )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time   
    logging.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    logging.info(f"Process completed in {elapsed_time:.2f} seconds")    

    # Save the validation and test datasets
    # logging.info("Saving validation and test datasets...")
    if args.dataset == 'cifar10' or args.dataset == 'mnist':    
        dep_var, independent_variables = None, None
        train_name = None
        val_name = None
        test_name = None
        train_input_list, val_input_list, test_input_list = None, None, None
        train_prob_list, val_prob_list, test_prob_list = [], [], []        
        logging.info("Dataset is CIFAR10 or MNIST; dependent variable and original DataFrames set to None.")
    else:    
        dep_var = args.dependent_variable 
        independent_variables = args.independent_variables   
        train_name = train_df if df is None else df
        val_name = test_df if df is None else df
        test_name = test_df if df is None else df
        if not args.output_data:
            train_input_list, val_input_list, test_input_list = None, None, None 
            logging.info("Output data flag not set; input data lists set to None.")
        else:
            logging.info("Output data flag set; including input data lists.")
        #     train_input_list, val_input_list, test_input_list = train_input_list, val_input_list, test_input_list        
        logging.info(f"Dependent variable: {dep_var}")
        logging.info(f"Training DataFrame: {'train_df' if df is None else 'df'}")
        logging.info(f"Validation DataFrame: {'test_df' if df is None else 'df'}")
        logging.info(f"Test DataFrame: {'test_df' if df is None else 'df'}")

    df_train = combine_results_to_dataframe(
        trainloader, train_input_list, train_target, train_predicted,
        dependent_variable=dep_var, independent_variables= independent_variables, 
        extra_features = train_prob_list, valset=trainset,
        original_df=train_name, name=f'{args.modelname}_Train'
    )
    logging.info(f"Training results DataFrame created with shape: {df_train.shape}")    
        
    df_validation = combine_results_to_dataframe(
        valloader, val_input_list, val_target, val_predicted,
        dependent_variable=dep_var, independent_variables=independent_variables, 
        extra_features = val_prob_list, valset=valset,
        original_df=val_name, name=f'{args.modelname}_Validation'
    )
    logging.info(f"Validation results DataFrame created with shape: {df_validation.shape}")

    df_test = combine_results_to_dataframe(
        testloader, test_input_list, test_target, test_predicted,
        dependent_variable=dep_var, independent_variables=independent_variables, 
        extra_features = test_prob_list, valset=testset,
        original_df=test_name, name=f'{args.modelname}_Test'
    )
    logging.info(f"Test results DataFrame created with shape: {df_test.shape}")

    # return train_target, train_predicted, train_prob_list, train_input_list, args.independent_variables
if __name__ == "__main__":
    main()
# %%
# print(len(values), values[0], values[1][:5], values[2][:5], values[3][:5])
# import pandas as pd
# import numpy as np

# #%%
# df_train = combine_results_to_dataframe(
#         dataloader = None, input_list = train_input_list, target_list = train_target, predicted_list = train_predicted,
#         dependent_variable=None, independent_variables= independent_variables, extra_features = train_prob_list, valset=None,
#         original_df=None, name='_Train'
#     )    
# %%
