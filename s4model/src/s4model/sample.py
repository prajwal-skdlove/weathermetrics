import os
import time
import logging
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tqdm import tqdm

import argparse
from dataset import load_data
from output import combine_results_to_dataframe, save_results
from model import S4Model, setup_optimizer
from train import load_model
from dataset import CSVDataset
from torch.utils.data import DataLoader

import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms
import torchvision
import warnings
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from output import save_results


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

def get_infer_args():
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument("--modelname", type=str, required=True, help="Name of the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    # parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    # Set up logging
    setup_logging()

    args, unknown = get_infer_args()
    logging.info(f"Arguments: {args}")

    # Load model and its configuration
    model, model_args, best_metric, _ = load_model(S4Model, args.modelname)
    

    args_to_log = {k: v for k, v in model_args.items() if k != "independent_variables"} 
    logging.info(f"Loaded model '{args.modelname}' with arguments (excluding independent_variables): {args_to_log}")

    modeltype = model_args.get("modeltype", None)
    independent_variables = model_args.get("independent_variables", None)
    batch_size = model_args.get("batch_size", 64)
    num_workers = model_args.get("num_workers", 0) 
    tabulardata = model_args.get("tabulardata", False)
    input_file = args.input_file   

    # logging.info(f"independent_variables: {independent_variables}")
    # logging.info(f"input_file: {input_file}")

    if tabulardata:
        if input_file.endswith('.csv'):
            input_df = pd.read_csv(input_file).fillna(0)
        elif input_file.endswith('.parquet'):        
            table = pq.read_table(input_file)
            input_df = table.to_pandas().fillna(0)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")
       
        dataset = CSVDataset(input_df, independent_variables, None, modeltype, inference=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logging.info(f"DataLoader created with batch size {batch_size} and num_workers {num_workers}")


    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = device == "cuda"
    logging.info(f"Using {device} device")

    model.to(device)       
    all_predictions = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, (inputs) in pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if modeltype == "regression":
                preds = outputs.squeeze().cpu().numpy()
                all_predictions.extend(preds)
            elif modeltype == "classification":
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                max_probs = np.max(probs, axis=1)
                max_classes = np.argmax(probs, axis=1)
                for prob, cls, prob_vec in zip(max_probs, max_classes, probs):
                    all_predictions.append({
                        "probability": prob,
                        "class": cls,
                        "probs": prob_vec.tolist()
                    })
            else:
                logging.error(f"Unsupported modeltype: {modeltype}")
                raise ValueError(f"Unsupported modeltype: {modeltype}")

    logging.info(f"Sample predictions: {all_predictions[0:5]}")   

    save_results(
        input_data=None,
        predictions=all_predictions,
        name=f"{args.modelname}_{modeltype}"
    )

    logging.info(f"Process Completed. Predictions saved for model '{args.modelname}'.")   


if __name__ == "__main__":
    main()

# python -m sample --modelname 72508014740 --modeltype classification --dataset ../data/weathermetrics/72508014740_validation.parquet  --tabulardata --dependent_variable 0 --epochs 1
# python -m sample --modelname 72508014740 --modeltype regression --dataset ../data/weathermetrics/72614594705_sample.parquet  --tabulardata --dependent_variable air_temp_degrees_cels_max --epochs 1

# python -m s4model --modelname 72508014740 --modeltype classification --dataset ../data/weathermetrics/72508014740_validation.parquet  --tabulardata --dependent_variable 0 --epochs 1
# python -m s4model --modelname 72508014740 --modeltype regression --dataset ../data/weathermetrics/72614594705_sample.parquet  --tabulardata --dependent_variable air_temp_degrees_cels_max --epochs 1

# python -m s4model --modelname fake --modeltype classification --dataset ../data/weathermetrics/fake_train.parquet  --tabulardata --dependent_variable target --epochs 1

# python -m s4model --modelname fake --modeltype classification --dataset ../data/weathermetrics/fake_train.parquet  --tabulardata --dependent_variable target --epochs 1
python -m s4model --modelname fake --modeltype classification --trainset ../data/weathermetrics/fake_train.parquet --valset ../data/weathermetrics/fake_validation.parquet --testset ../data/weathermetrics/fake_test.parquet --tabulardata --dependent_variable target --epochs 1
