import torch
import torch.optim as optim
from tqdm import tqdm
import os

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(model, trainloader, criterion, optimizer, device, modeltype="classification"):
    """
    modeltype: "classification" or "regression"
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        # print(f"Batch input shape: {inputs.shape}, Batch target shape: {targets.shape}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if modeltype == "regression":
                if targets.dim() == 1 and outputs.dim() == 2 and outputs.size(1) == 1:
                    targets = targets.view(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

        if modeltype == "classification":
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            desc = (
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(trainloader), train_loss/(batch_idx+1), acc, correct, total)
            )
        elif modeltype == "regression":  # regression
            mse = loss.item()
            desc = (
                'Batch Idx: (%d/%d) | Loss: %.3f (MSE)' %
                (batch_idx, len(trainloader), mse)
            )
        pbar.set_description(desc)


def eval(model, dataloader, criterion, device, epoch, modelname, best_acc, args, modeltype="classification", checkpoint=False):
    """
    Evaluates the model.
    modeltype: "classification" or "regression"
    """
    output_list_target = []
    output_list_predicted = []
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if modeltype == "regression":
                if targets.dim() == 1 and outputs.dim() == 2 and outputs.size(1) == 1:
                    targets = targets.view(-1, 1)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            total += targets.size(0)

            if modeltype == "classification":
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                output_list_target.extend(targets.cpu().numpy())
                output_list_predicted.extend(predicted.cpu().numpy())
                acc = 100. * correct / total
                desc = (
                    'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (batch_idx, len(dataloader), eval_loss/(batch_idx+1), acc, correct, total)
                )
            elif modeltype == "regression":  # regression
                output_list_target.extend(targets.cpu().numpy())
                output_list_predicted.extend(outputs.squeeze().cpu().numpy())
                mse = loss.item()
                desc = (
                    'Batch Idx: (%d/%d) | Loss: %.3f (MSE)' %
                    (batch_idx, len(dataloader), mse)
                )
            pbar.set_description(desc)

    # Save checkpoint.
    if checkpoint and modeltype == "classification":
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                "args": vars(args)
            }
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            torch.save(state, f'../checkpoint/{modelname}ckpt.pth')
            best_acc = acc
        return acc, output_list_target, output_list_predicted
    elif checkpoint and modeltype == "regression":
        mse = eval_loss / (batch_idx + 1)
        state = {
            'model': model.state_dict(),
            'mse': mse,
            'epoch': epoch,
            "args": vars(args)
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, f'../checkpoint/{modelname}ckpt.pth')
        return mse, output_list_target, output_list_predicted
    else:
        if modeltype == "classification":
            acc = 100.*correct/total
            return acc, output_list_target, output_list_predicted
        elif modeltype == "regression":
            mse = eval_loss / (batch_idx + 1)
            return mse, output_list_target, output_list_predicted

#%%
#Loads the model for inference
def load_model(model_class, modelname, device="cpu"):
    """Loads the model and its arguments from a checkpoint."""
    checkpoint_path = f"../checkpoint/{modelname}ckpt.pth"
    
    assert os.path.isdir('../checkpoint/'), 'Error: no checkpoint directory found!'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model arguments
    model_args = checkpoint["args"]
    
    # Initialize model with saved args
    model = model_class(
        d_input=model_args["d_input"],
        d_output=model_args["d_output"],
        lr =model_args["lr"],
        d_model=model_args["d_model"],
        n_layers=model_args["n_layers"],
        dropout=model_args["dropout"],
        prenorm=model_args["prenorm"],
    )
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    # Infer metric type and print
    if "acc" in checkpoint:
        best_metric = checkpoint["acc"]
        metric_name = "Accuracy"
    elif "mse" in checkpoint:
        best_metric = checkpoint["mse"]
        metric_name = "MSE"
    else:
        best_metric = None
        metric_name = "Unknown metric"

    start_epoch = checkpoint.get('epoch', None)

    print(f"Model and arguments loaded from {checkpoint_path}")
    print(f"{metric_name}: {best_metric}")
    return model, model_args, best_metric, start_epoch


