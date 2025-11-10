import torch
import torch.optim as optim
from tqdm import tqdm
import os
import logging
import numpy as np
from model import log_gradients


###############################################################################
# Standard PyTorch training and evaluation utilities with error handling
###############################################################################

def train(model, trainloader, criterion, optimizer, device, modeltype="classification"):
    """
    Train the model for one epoch with logging and error handling.

    Args:
        model: PyTorch model to be trained.
        trainloader: DataLoader providing training batches.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Target device ("cpu" or "cuda").
        modeltype: "classification" or "regression".

    Returns:
        (avg_loss, accuracy) for classification
        (avg_loss,) for regression
    """

    if modeltype not in ["classification", "regression"]:
        logging.error(f"Invalid modeltype: {modeltype}")
        raise ValueError(f"Invalid modeltype '{modeltype}'. Use 'classification' or 'regression'.")

    model.train()
    train_loss, correct, total = 0, 0, 0

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc="Training")
    for batch_idx, (inputs, targets) in pbar:
        try:
            # Move inputs/targets to device
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Handle regression reshaping
            if modeltype == "regression":
                if targets.dim() == 1 and outputs.dim() == 2 and outputs.size(1) == 1:
                    targets = targets.view(-1, 1)

            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()

            # Log gradient stats
            log_gradients(model)

            optimizer.step()

            # Track loss
            train_loss += loss.item()
            total += targets.size(0)

            if modeltype == "classification":
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total if total > 0 else 0.0
                desc = (f"Batch {batch_idx}/{len(trainloader)} | "
                        f"Loss: {train_loss/(batch_idx+1):.3f} | Acc: {acc:.2f}% ({correct}/{total})")
            else:  # regression
                desc = (f"Batch {batch_idx}/{len(trainloader)} | "
                        f"Loss: {loss.item():.3f} (MSE)")

            pbar.set_description(desc)

        except Exception as e:
            logging.exception(f"Error in training loop at batch {batch_idx}: {e}")
            continue  # Skip problematic batch instead of crashing

    # ---- Final logging ----
    avg_loss = train_loss / (batch_idx + 1 if batch_idx >= 0 else 1)

    if modeltype == "classification":
        acc = 100. * correct / total if total > 0 else 0.0
        logging.info(f"Training complete — Avg Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        return avg_loss, acc
    else:
        logging.info(f"Training complete — Avg Loss (MSE): {avg_loss:.4f}")
        return avg_loss,


def eval(model, dataloader, criterion, device, epoch, modelname, best_acc,
         args, modeltype="classification", checkpoint=False):
    """
    Evaluate the model safely with logging and error handling.

    Args:
        model: PyTorch model.
        dataloader: DataLoader for evaluation data.
        criterion: Loss function.
        device: "cpu" or "cuda".
        epoch: Current epoch.
        modelname: Name prefix for checkpoints.
        best_acc: Best accuracy seen so far (classification only).
        args: Namespace or dict of model args (saved for checkpoints).
        modeltype: "classification" or "regression".
        checkpoint: Save checkpoint if True.

    Returns:
    Returns:
        (metric, targets, predictions, prob_list)
        For classification:
            - metric: accuracy 
            - prob_list: list of dicts per-sample mapping class index (as str) -> probability

        For regression:            
            - metric: mean squared error
            - prob_list: empty list           
        
    """
    if modeltype not in ["classification", "regression"]:
        logging.error(f"Invalid modeltype: {modeltype}")
        raise ValueError(f"Invalid modeltype '{modeltype}'. Use 'classification' or 'regression'.")

    output_list_target, output_list_predicted, prob_list = [], [], []
    model.eval()
    eval_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
        for batch_idx, (inputs, targets) in pbar:
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Handle regression reshaping
                if modeltype == "regression":
                    if targets.dim() == 1 and outputs.dim() == 2 and outputs.size(1) == 1:
                        targets = targets.view(-1, 1)

                # Compute loss
                loss = criterion(outputs, targets)
                eval_loss += loss.item()
                total += targets.size(0)

                if modeltype == "classification":
                    # Compute probabilities
                    if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                        # Binary logits or single-output: use sigmoid -> produce two-class probs
                        probs_pos = torch.sigmoid(outputs.squeeze())
                        # Ensure shape (N,2)
                        probs_batch = torch.stack([1.0 - probs_pos, probs_pos], dim=1)
                    else:
                        probs_batch = torch.softmax(outputs, dim=1)

                    # Predicted class from probabilities
                    predicted = probs_batch.argmax(dim=1)
                    correct += predicted.eq(targets).sum().item()
                        
                    # _, predicted = outputs.max(1)
                    # correct += predicted.eq(targets).sum().item()

                    # Ensure extend always gets a list/array
                    output_list_target.extend(np.atleast_1d(targets.cpu().numpy()))
                    output_list_predicted.extend(np.atleast_1d(predicted.cpu().numpy()))
                    # Build per-sample probability dicts
                    for pb in probs_batch.cpu().numpy():
                        prob_list.append({str(i): float(pb[i]) for i in range(len(pb))})

                    acc = 100. * correct / total if total > 0 else 0.0
                    desc = (f"Batch {batch_idx}/{len(dataloader)} | "
                            f"Loss: {eval_loss/(batch_idx+1):.3f} | Acc: {acc:.2f}% ({correct}/{total})")

                else:  # regression
                    output_list_target.extend(np.atleast_1d(targets.cpu().numpy()))
                    output_list_predicted.extend(np.atleast_1d(outputs.squeeze().cpu().numpy()))

                    desc = (f"Batch {batch_idx}/{len(dataloader)} | "
                            f"Loss: {loss.item():.3f} (MSE)")

                pbar.set_description(desc)

            except Exception as e:
                logging.exception(f"Error in evaluation loop at batch {batch_idx}: {e}")
                continue  # Skip bad batch instead of crashing

    # ---- Checkpoint saving ----
    try:
        if checkpoint:
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')

            if modeltype == "classification":
                acc = 100. * correct / total if total > 0 else 0.0
                if acc > best_acc:
                    state = {
                        "model": model.state_dict(),
                        "acc": acc,
                        "epoch": epoch,
                        "args": vars(args) if not isinstance(args, dict) else args
                    }
                    torch.save(state, f'../checkpoint/{modelname}ckpt.pth')
                    logging.info(f"Checkpoint saved: {modelname}ckpt.pth (Acc: {acc:.2f}%)")
                    best_acc = acc
                return acc, output_list_target, output_list_predicted, prob_list

            else:  # regression
                mse = eval_loss / (batch_idx + 1)
                state = {
                    "model": model.state_dict(),
                    "mse": mse,
                    "epoch": epoch,
                    "args": vars(args) if not isinstance(args, dict) else args
                }
                torch.save(state, f'../checkpoint/{modelname}ckpt.pth')
                logging.info(f"Checkpoint saved: {modelname}ckpt.pth (MSE: {mse:.4f})")
                return mse, output_list_target, output_list_predicted, prob_list

    except Exception as e:
        logging.exception(f"Error while saving checkpoint: {e}")

    # ---- Final metric return ----
    if modeltype == "classification":
        acc = 100. * correct / total if total > 0 else 0.0
        logging.info(f"Evaluation complete — Accuracy: {acc:.2f}%")
        return acc, output_list_target, output_list_predicted, prob_list
    else:
        mse = eval_loss / (batch_idx + 1 if batch_idx >= 0 else 1)
        logging.info(f"Evaluation complete — MSE: {mse:.4f}")
        return mse, output_list_target, output_list_predicted, prob_list

def load_model(model_class, modelname, device="cpu"):
    """
    Load model and arguments from a checkpoint with logging and error handling.

    Args:
        model_class: The class of the model to initialize.
        modelname: Checkpoint file prefix (string).
        device: Device to load model on ("cpu" or "cuda").

    Returns:
        model, model_args, best_metric, start_epoch
    """
    checkpoint_dir = "../checkpoint"
    checkpoint_path = os.path.join(checkpoint_dir, f"{modelname}ckpt.pth")

    # Check directory
    if not os.path.isdir(checkpoint_dir):
        logging.error(f"Checkpoint directory not found: {checkpoint_dir}")
        raise FileNotFoundError(f"No checkpoint directory at {checkpoint_dir}")

    # Check file
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        logging.exception(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

    try:
        # Restore model args
        model_args = checkpoint.get("args", {})
        if not model_args:
            logging.warning("No 'args' found in checkpoint — using defaults.")
            model_args = {}

        # Initialize model with args (keys missing -> None)
        model = model_class(
            d_input=model_args.get("d_input"),
            d_output=model_args.get("d_output"),
            lr=model_args.get("lr"),
            d_model=model_args.get("d_model"),
            n_layers=model_args.get("n_layers"),
            dropout=model_args.get("dropout"),
            prenorm=model_args.get("prenorm"),
        )

        # Load weights
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        # Detect metric type
        if "acc" in checkpoint:
            best_metric, metric_name = checkpoint["acc"], "Accuracy"
        elif "mse" in checkpoint:
            best_metric, metric_name = checkpoint["mse"], "MSE"
        else:
            best_metric, metric_name = None, "Unknown"

        start_epoch = checkpoint.get("epoch", None)

        logging.info(f"Model successfully loaded from {checkpoint_path}")
        logging.info(f"{metric_name}: {best_metric if best_metric is not None else 'N/A'}")
        if start_epoch is not None:
            logging.info(f"Resuming from epoch {start_epoch}")

        return model, model_args, best_metric, start_epoch

    except KeyError as e:
        logging.exception(f"Missing key in checkpoint: {e}")
        raise
        raise
###############################################################################