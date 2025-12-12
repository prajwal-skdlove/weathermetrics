import pandas as pd
import os
from datetime import datetime
import logging
import pyarrow as pa
import pyarrow.parquet as pq


#%%
def combine_results_to_dataframe(
    dataloader=None,
    input_list=None,
    target_list=None,
    predicted_list=None,
    dependent_variable=None,
    independent_variables=None,
    extra_features=None,
    valset=None,
    original_df=None,
    name=None,
    csv = False
):
    """
    Combines input features with target and predicted values into a DataFrame.
    Saves the results as a CSV file in the '../results/' directory with a timestamped filename.

    Args:
        dataloader: Optional data loader object containing the dataset
        input_list: Optional list of input features
        target_list: List of target/actual values
        predicted_list: List of predicted values from the model
        dependent_variable: Name of the target variable column (default: 'Target')
        independent_variables: List of names for input feature columns (default: None)
        extra_features: List of lists/dictionaries containing additional features
        valset: Optional validation dataset
        original_df: Optional original DataFrame
        name: Name prefix for the saved file

    Returns:
        pandas.DataFrame: DataFrame containing combined results
        None: If an error occurs during processing

    Raises:
        ValueError: If target_list or predicted_list is not provided
    """

    try:
        if target_list is None or predicted_list is None:
            raise ValueError("target_list and predicted_list must be provided.")

        logging.info("Combining results into DataFrame...")

        # Determine target column name
        target_col = dependent_variable if dependent_variable else 'Target'

        expected_len = len(target_list)       
        if not isinstance(extra_features, list) or len(extra_features) != expected_len:
            # If extra_features was a list but wrong length, preserve existing items and
            # extend/truncate to match expected_len
            logging.warning("extra_features is not a list with the same length as targets; Adding empty list of the same length.")                          
            if isinstance(extra_features, list):
                if len(extra_features) < expected_len:
                    last = []
                    extra_features = extra_features + [last] * (expected_len - len(extra_features))
                elif len(extra_features) > expected_len:
                    extra_features = extra_features[:expected_len]
            else:
                # If it wasn't a list, replace with a list of None of the correct length
                extra_features = [[] for _ in range(expected_len)]
        elif extra_features is None:
            extra_features =  [[] for _ in range(expected_len)]
        
        # Create DataFrame with targets and predictions
        df_results = pd.DataFrame({
            target_col: target_list,
            "Predicted": predicted_list,
            "extra_features": extra_features
        })

        # Validate input_list length and add it to the results DataFrame
        if input_list is not None:
            if len(input_list) != expected_len:
                raise Warning("input_list must have the same length as target_list and predicted_list.")                 
            
            input_df = pd.DataFrame(input_list, columns=independent_variables)
            df_results = pd.concat([df_results, input_df], axis=1)
                

        # Ensure results directory exists
        results_dir = "../results/"
        os.makedirs(results_dir, exist_ok=True)
        logging.debug(f"Results directory verified at '{results_dir}'")

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
        filename = f"{results_dir}{name}_results_{timestamp}"

        if csv:
            filename += '.csv'
            df_results.to_csv(filename, index=False)         
        else:
            filename = filename + '.parquet'
            df_results.to_parquet(filename, index=False)           
        
        logging.info(f"Dataset successfully saved as '{filename}'")        

        return df_results

    except Exception as e:
        logging.exception(f"Error in combine_results_to_dataframe: {e}")
        print(f"Error in combine_results_to_dataframe: {e}")
        return None

# combine_results_to_dataframe(
#     dataloader=None,
#     input_list=None,
#     target_list=[0, 1, 0, 1],
#     predicted_list=[0.1, 0.9, 0.2, 0.8 ],
#     dependent_variable='tgt_bin',
#     independent_variables=None,
#     extra_features=   [
#                     {"0" : 0.245, "1" : 0.755, "2" : 0.345, "3" : 0.845},         
#                     {"0" : 0.15, "1" : 0.85, "2" : 0.25, "3" : 0.95}, 
#                      {"0" : 0.3, "1" : 0.7, "2" : 0.4, "3" : 0.6}, 
#                      {"0" : 0.05, "1" : 0.95, "2" : 0.15, "3" : 0.85}
#                      ],
#     valset=None,
#     original_df=None,
#     name = 'test'
# )
#%%

def save_results(
    input_data=None,
    predictions=None,
    target_list=None,
    dependent_variable=None,
    independent_variables=None,
    extra_features=None,
    input_df=None,
    name=None
):
    """
    Saves inference results in a pandas DataFrame and (optionally) as a CSV file.

    Args:
        input_data: Optional input data used for inference. send it as list
        predictions (list): List of predicted values from the model.
        target_list (list, optional): List of actual/target values. If None, target column will be empty.
        dependent_variable (str, optional): Name of the target variable column. Defaults to 'Target'.
        independent_variables (list, optional): List of names for input feature columns. Defaults to None.
        extra_features (list, optional): List of additional features (dicts/lists) for each sample.
        input_df (pd.DataFrame, optional): Original input DataFrame.
        name (str, optional): Prefix for the saved file name.

    Returns:
        pd.DataFrame: DataFrame containing targets, predictions, and extra features.
        None: If an error occurs during processing.

    Raises:
        ValueError: If predictions are not provided.

    Notes:
        - The function ensures extra_features matches the length of predictions/targets.
        - Logging is used for error and warning messages.
        - CSV saving code is commented out; uncomment to enable file saving.
    """

    try:
        if predictions is None:
            raise ValueError("Predictions must be provided.")

        target_col = dependent_variable if dependent_variable else 'Target'
        expected_len = len(target_list)       
        if not isinstance(extra_features, list) or len(extra_features) != expected_len:
            logging.warning("extra_features is not a list with the same length as targets; Adding empty list of the same length.")                          
            if isinstance(extra_features, list):
                if len(extra_features) < expected_len:
                    last = []
                    extra_features = extra_features + [last] * (expected_len - len(extra_features))
                elif len(extra_features) > expected_len:
                    extra_features = extra_features[:expected_len]
            else:
                extra_features = [[] for _ in range(expected_len)]
        elif extra_features is None:
            extra_features =  [[] for _ in range(expected_len)]

        df_results = pd.DataFrame({
            target_col: target_list if target_list is not None else [],
            "Predicted": predictions,
            "extra_features": extra_features if extra_features is not None else []
        })

         # Validate input_list length and add it to the results DataFrame
        if input_data is not None:
            if len(input_data) != expected_len:
                raise Warning("input_data must have the same length as target_list and predicted_list.")                              
            
            input_df = pd.DataFrame(input_data, columns=independent_variables)
            df_results = pd.concat([df_results, input_df], axis=1)

        return df_results

    except Exception as e:
        logging.exception(f"Error saving inference results: {e}")
        print(f"Error saving inference results: {e}")
        return None
    
# save_results(
#     input_data=None,
#     predictions=[0.1, 0.9, 0.2, 0.8 ],
#     target_list=[0, 1, 0, 1],
#     dependent_variable='tgt_bin',
#     independent_variables= None,
#     input_df=None,    
#     name = 'test',
#     extra_features = [{"0" : 0.245, "1" : 0.755, "2" : 0.345, "3" : 0.845}, 
#                      {"0" : 0.15, "1" : 0.85, "2" : 0.25, "3" : 0.95}, 
#     #                  {"0" : 0.3, "1" : 0.7, "2" : 0.4, "3" : 0.6}, 
#     #                  {"0" : 0.05, "1" : 0.95, "2" : 0.15, "3" : 0.85}
#                     ],
# )
