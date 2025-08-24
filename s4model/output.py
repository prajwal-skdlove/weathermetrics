import pandas as pd
import os
from datetime import datetime
import logging


#%%

def combine_results_to_dataframe(
    dataloader=None,
    target_list=None,
    predicted_list=None,
    dependent_variable=None,
    valset=None,
    original_df=None,
    name=None
):
    """
    Combines input features with target and predicted values into a DataFrame.
    Saves the results as a CSV file in the '../results/' directory with a timestamped filename.
    """

    try:
        if target_list is None or predicted_list is None:
            raise ValueError("target_list and predicted_list must be provided.")

        logging.info("Combining results into DataFrame...")

        # Determine target column name
        target_col = dependent_variable if dependent_variable else 'Target'

        # Create DataFrame with targets and predictions
        df_results = pd.DataFrame({
            target_col: target_list,
            "Predicted": predicted_list
        })

        # Ensure results directory exists
        results_dir = "../results/"
        os.makedirs(results_dir, exist_ok=True)
        logging.debug(f"Results directory verified at '{results_dir}'")

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
        filename = f"{results_dir}{name}_results_{timestamp}.csv"

        # Save to CSV
        df_results.to_csv(filename, index=False)
        logging.info(f"Dataset successfully saved as '{filename}'")
        print(f"Dataset successfully saved as '{filename}'")

        return df_results

    except Exception as e:
        logging.exception(f"Error in combine_results_to_dataframe: {e}")
        print(f"Error in combine_results_to_dataframe: {e}")
        return None

#%%

def save_results(
    input_data=None,
    predictions=None,
    target_list=None,
    dependent_variable=None,
    input_df=None,
    name=None
):
    """
    Saves inference results in a CSV file with robust logging and error handling.
    """

    results_dir = "../results/"
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create results directory '{results_dir}': {e}")
        print(f"Failed to create results directory '{results_dir}': {e}")
        return None

    try:
        if predictions is None:
            raise ValueError("Predictions must be provided.")

        target_col = dependent_variable if dependent_variable else 'Target'

        # Construct DataFrame
        df_results = pd.DataFrame({
            target_col: target_list if target_list is not None else [],
            "Predicted": predictions
        })

        # If target_list is None, drop the column
        if target_list is None:
            df_results.drop(columns=[target_col], inplace=True)
            logging.debug(f"No target_list provided; '{target_col}' column dropped.")

        # Timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%I%M%S%p")
        filename = f"{results_dir}{name}_results_{timestamp}.csv"

        # Save CSV
        df_results.to_csv(filename, index=False)
        logging.info(f"Inference results saved to '{filename}'")
        print(f"Inference results saved to '{filename}'")

        return df_results

    except Exception as e:
        logging.exception(f"Error saving inference results: {e}")
        print(f"Error saving inference results: {e}")
        return None
