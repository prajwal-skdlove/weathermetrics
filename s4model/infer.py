#%%
import torch
import pandas as pd
from dataset import CSVDataset
from model import S4Model
from train import load_model
from config import get_args
from output import save_results

#%%
# Load model and saved arguments

def infer(model, input_data):
    """Runs inference on a single input."""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()        
    return prediction, output


if __name__ == "__main__":
    args, unknown = get_args()
    # Load model
    model, model_args, best_acc, epochs = load_model(S4Model, args.modelname)
    # Now you have saved_args, so you don’t need to input args again
    print(f"Loaded model with arguments: {model_args}")

    # Load new data for inference
    df = pd.read_csv(args.dataset).fillna(0)
    y_column = model_args["dependent_variable"]
    x_columns = model_args["independent_variables"] if model_args["independent_variables"] else [x for x in df.columns if x != y_column]    

    
    dataset = CSVDataset(df, x_columns, y_column, transform=None)

   # Run inference
    predictions = []
    target_data = []
    input_data = []
    output_range = []
    
    for i in range(len(dataset)):
        sample = dataset[i][0].numpy()  
        target = dataset[i][1].numpy()      
        prediction, output = infer(model, sample)  
        predictions.append(prediction)
        input_data.append(sample.flatten().tolist()) 
        target_data.append(target)
        output_range.append(output.flatten().tolist())

    # Save results to CSV
    save_results(input_data, predictions, target_data, y_column,  df, f"{args.modelname}_inference")
    save_results(output_range, predictions, None, None, None, f"{args.modelname}_output_range")