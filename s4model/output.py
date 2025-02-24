import pandas as pd
import os
from datetime import datetime 

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
    results_dir = "../results/"
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
    filename = f"{results_dir}{name}_results_{timestamp}.csv"

    # Save DataFrame to CSV
    df_results.to_csv(filename, index=False)

    print(f"Dataset successfully saved as '{filename}'")

    return df_results

#%%

# Save the results of inference
def save_results(input_data, predictions, target_list = None, dependent_variable = None, input_df=None, name = None):
    """Saves inference results in a CSV file."""
    results_dir = "../results/"
    os.makedirs(results_dir, exist_ok=True)    
    timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
    filename = f"{results_dir}{name}_results_{timestamp}.csv"
    
    df_results = pd.DataFrame(input_data)

    # Determine column names while removing the dependent variable
    if isinstance(input_df, pd.DataFrame):  
        df_results.columns = [col for col in input_df.columns.tolist() if col != dependent_variable][:df_results.shape[1]]
    else:  
        df_results.columns = [f'feature_{i}' for i in range(df_results.shape[1])]

    # Add the target and predicted values
    if dependent_variable is not None:
        target = dependent_variable
    else:
        target = 'Target'
    
    df_results[target] = target_list
    df_results["Prediction"] = predictions
    df_results.to_csv(filename, index=False)
    print(f"Inference results saved to {filename}")
