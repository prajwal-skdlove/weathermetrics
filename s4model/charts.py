# %%

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Model Metric Charts")
    parser.add_argument("--df", required=True, type=str, help = "Full path of the csv file with actual and predicted values")    
    parser.add_argument("--actual", required=True, type=str, help = "Column name of actual values")
    parser.add_argument("--predicted", required=True, type=str, help = "Column name of predicted values")
    # parser.add_argument("--labels", type=str, help =  "Comma separated list of labels")
    # parser.add_argument("--title", type=str,help= "Title of the chart")
    # parser.add_argument("--xlabel", type=str, help= "X-axis label")
    # parser.add_argument("--ylabel", type=str, help= "Y-axis label")
    parser.add_argument("--chart_type", type=str, nargs='+', default=["c", "sb", 'am'], help="Type(s) of chart(s) to create. Options: s (scatter), m (mismatches), c (confusion_matrix), e (error_distribution), mh (misclassification_heatmap), sb (stacked_bar), am (accuracy_metrics).")
    args, unknown = parser.parse_known_args()

#%%
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
import numpy as np
import torch

#%%
def plot_target_vs_predicted(target_values, predicted_values, title="Actual vs Predicted", xlabel="Actual", ylabel="Predicted"):
    """
    Plots a scatter plot of target vs predicted values.

    Args:
        target_values (torch.Tensor or list or numpy.ndarray): The ground truth values.
        predicted_values (torch.Tensor or list or numpy.ndarray): The predicted values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Convert tensors to numpy if needed
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(target_values, predicted_values, alpha=0.6, edgecolor='k', label='Data points')
    plt.plot([min(target_values), max(target_values)], [min(target_values), max(target_values)], color='red', linestyle='--', label='Ideal fit')  # Diagonal line for reference
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

#%%
def plot_target_vs_predicted_mismatches(target_values, predicted_values, title="Actual vs Predicted (Mismatches)", xlabel="Target", ylabel="Predicted"):
    """
    Plots a scatter plot of target vs predicted values where target != predicted.

    Args:
        target_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Convert inputs to numpy arrays if they are not already    
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)

    # Filter mismatched values
    mismatches = target_values != predicted_values
    mismatched_targets = target_values[mismatches]
    mismatched_predictions = predicted_values[mismatches]

    if mismatched_targets.size == 0:
        print("No mismatches found. Nothing to plot.")
        return

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(mismatched_targets, mismatched_predictions, alpha=0.6, edgecolor='k', label='Mismatched points')
    plt.plot([min(mismatched_targets), max(mismatched_targets)], [min(mismatched_targets), max(mismatched_targets)], color='red', linestyle='--', label='Ideal fit')  # Diagonal line for reference
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
    
#%%
def plot_confusion_matrix(target_values, predicted_values, labels, title='Confusion Matrix', xlabel="Predicted", ylabel="Actual"):
    """
    Plots a confusion matrix to visualize classification errors.

    Args:
        target_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        labels (list): List of category labels for the confusion matrix.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.        
    """

    # Convert inputs to numpy arrays if they are not already    
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)

    cm = confusion_matrix(target_values, predicted_values, labels=sorted(labels))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=sorted(labels), yticklabels=sorted(labels))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.show()

#%%
def plot_error_distribution(target_values, predicted_values, title='Error Distribution by Actual', xlabel='Actual', ylabel='Error Count'):
    """
    Plots the distribution of errors by actual category.

    Args:        
        target_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.     
    """
    # Convert inputs to numpy arrays if they are not already
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)

    # Identify mismatches
    mismatches = target_values[target_values != predicted_values]
    error_counts = pd.Series(mismatches).value_counts().sort_index()

    # Plot the error distribution
    plt.figure(figsize=(8, 6))
    error_counts.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#%%
def plot_misclassification_heatmap(target_values, predicted_values, labels, title='Misclassification Heatmap', xlabel="Predicted", ylabel="Actual"):
    """
    Plots a heatmap of misclassification patterns.

    Args:
        target_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        labels (list): List of category labels.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis. 
    """
    # Convert inputs to numpy arrays if they are not already
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)

    # Create a DataFrame for mismatched data
    mismatches = pd.DataFrame({
        'Actual': target_values,
        'Predicted': predicted_values
    })
    mismatches = mismatches[mismatches['Actual'] != mismatches['Predicted']]

    # Create a crosstab for mismatched data
    heatmap_data = pd.crosstab(
        mismatches['Actual'],  # Actual values (rows)
        mismatches['Predicted'],  # Predicted values (columns)
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )

    # Ensure all categories are represented in rows and columns
    heatmap_data = heatmap_data.reindex(index=sorted(labels), columns=sorted(labels), fill_value=0)


    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Reds", xticklabels=sorted(labels), yticklabels=sorted(labels))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#%%
def plot_stacked_bar(target_values, predicted_values, labels, title='Proportion of Correct vs. Incorrect Predictions', xlabel="Actual", ylabel="Proportion"):
    """
    Plots a stacked bar chart of correct and incorrect predictions for each category.

    Args:
        target_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        labels (list): List of category labels.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Convert inputs to numpy arrays if they are not already
    if isinstance(target_values, torch.Tensor):
        target_values = target_values.cpu().numpy()
    else:
        target_values = np.array(target_values)
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().numpy()
    else:
        predicted_values = np.array(predicted_values)

    # Create a DataFrame for analysis
    df = pd.DataFrame({
        'Actual': target_values,
        'Predicted': predicted_values
    })
    df['Correct'] = (df['Actual'] == df['Predicted']).astype(int)

    # Calculate proportions
    stacked_data = pd.crosstab(df['Actual'], df['Correct'], normalize='index')
    stacked_data = stacked_data.reindex(index=sorted(labels), fill_value=0)
    stacked_data.columns = ['Incorrect', 'Correct']

    # Reorder columns to make 'Correct' the bottom stack
    stacked_data = stacked_data[['Correct', 'Incorrect']]

    # Plot the stacked bar chart
    ax = stacked_data.plot(
        kind='bar',
        stacked=True,
        color=['#006450', '#BC0024'],  # Warm red for Incorrect, warm green for Correct
        figsize=(10, 6),
        edgecolor='black'
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    # Annotate the bars with percentages (rounded to 0 digits)
    for i, (index, row) in enumerate(stacked_data.iterrows()):
        correct_pct = round(row['Correct'] * 100)
        incorrect_pct = round(row['Incorrect'] * 100)
        ax.text(i, row['Correct'] / 2, f"{correct_pct}%", ha='center', va='center', color='white', fontsize=12)
        ax.text(i, row['Correct'] + row['Incorrect'] / 2, f"{incorrect_pct}%", ha='center', va='center', color='black', fontsize=12)

    plt.legend(title='Prediction', loc='upper right')
    plt.show()

# %%

def calculate_class_wise_metrics(actual_values, predicted_values, labels):
    """
    Calculates class-wise precision, recall, F1-score, support, accuracy, TPR, and FPR for a multi-class classification problem.

    Args:
        actual_values (list or numpy.ndarray or pandas.Series): The ground truth values.
        predicted_values (list or numpy.ndarray or pandas.Series): The predicted values.
        labels (list): List of unique class labels.

    Returns:
        pd.DataFrame: A DataFrame containing precision, recall, F1-score, support, TPR, FPR, and specificity for each class.
        float: Overall accuracy.
    """
    # Calculate precision, recall, F1-score, and support
    precision, recall, f1_score, support = precision_recall_fscore_support(
        actual_values, predicted_values, labels=labels, zero_division=0
    )

    # Calculate confusion matrix
    cm = confusion_matrix(actual_values, predicted_values, labels=labels)

    # Calculate overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # Calculate TPR, FPR, and specificity for each class
    tpr = {}
    fpr = {}
    specificity = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        tpr[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[label] = fp / (fp + tn) if (fp + tn) > 0 else 0
        specificity[label] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Create a DataFrame for better visualization
    overall_metrics = pd.DataFrame({
        'TPR': tpr,
        'FPR': fpr,
        'Specificity': specificity,
    })

    overall_metrics = overall_metrics.sort_index().reset_index(names="Class")

    metrics_df = pd.DataFrame({
        'Class': labels,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    })

    # Merge and sort the DataFrame by class
    metrics_df = metrics_df.sort_values(by='Class').reset_index(drop=True).merge(overall_metrics, on='Class')
    metrics_df = metrics_df[['Class', 'Precision', 'Recall', 'Specificity', 'TPR', 'FPR', 'F1-Score', 'Support']].round(4)

    return accuracy, metrics_df

# %%
if __name__ == "__main__":
    df = pd.read_csv(args.df)
    actual_values = df[args.actual]
    predicted_values = df[args.predicted]
    labels = df[args.actual].unique()

    for chart_type in args.chart_type:
        if chart_type in ["s", "scatter"]:
            plot_target_vs_predicted(actual_values, predicted_values)
        elif chart_type in ["m", "mismatches"]: 
            plot_target_vs_predicted_mismatches(actual_values, predicted_values)
        elif chart_type in ["c", "confusion_matrix"]: 
            plot_confusion_matrix(actual_values, predicted_values, labels)
        elif chart_type in ["e", "error_distribution"]:
            plot_error_distribution(actual_values, predicted_values)
        elif chart_type in ["mh", "misclassification_heatmap"]:  
            plot_misclassification_heatmap(actual_values, predicted_values, labels)
        elif chart_type in ["sb", "stacked_bar"]:
            plot_stacked_bar(actual_values, predicted_values, labels=labels)
        elif chart_type in ["am", "accuracy_metrics"]:
            accuracy,metrics_df = calculate_class_wise_metrics(actual_values, predicted_values, labels)
            print("Accuracy = ", round(accuracy, 4))            
            print("Class-wise Metrics (Sorted by Class):")            
            print(metrics_df)
            # Formulas for reference
            print("\nFormulas:")
            print("Accuracy = (True Positives + True Negatives) / Total Samples")            
            print("Precision = True Positives / (True Positives + False Positives)")
            print("Recall = True Positives / (True Positives + False Negatives)")
            print("F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
            print("Specificity = True Negatives / (True Negatives + False Positives)")            
            print("True Positive Rate (TPR) = True Positives / (True Positives + False Negatives)")
            print("False Positive Rate (FPR) = False Positives / (False Positives + True Negatives)")            
            print("Support = Number of actual instances for each class")
        else:
            print(f"Invalid chart type: {chart_type}. Please choose from: s (scatter), m (mismatches), c (confusion_matrix), e (error_distribution), mh (misclassification_heatmap), sb (stacked_bar).")

# %%
# df = pd.read_csv("../results/74486094789_Test_results_20250414_103853PM.csv")
# actual_values = df['0']
# predicted_values = df['Predicted']
# labels = df['0'].unique() 
# print(labels)
# plot_confusion_matrix(actual_values, predicted_values, labels)
