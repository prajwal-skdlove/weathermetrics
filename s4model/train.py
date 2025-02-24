import torch
import torch.optim as optim
from tqdm import tqdm
import os

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(model, trainloader, criterion, optimizer, device):
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
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )


def eval(model, dataloader, criterion, device, epoch, modelname, best_acc, args, checkpoint=False):
   
    output_list_target = [] # To store target data
    output_list_predicted = [] # To store predicted data   
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()           

            output_list_target.extend(targets.cpu().numpy())  # Collect targets
            output_list_predicted.extend(predicted.cpu().numpy())  # Collect predictions                         

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )           
            
            

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                "args": vars(args)
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'../checkpoint/{modelname}ckpt.pth')
            best_acc = acc

        return acc, output_list_target, output_list_predicted


#%%
#Loads the model for inference
def load_model(model_class, modelname, device="cpu"):
    """Loads the model and its arguments from a checkpoint."""
    checkpoint_path = f"../checkpoint/{modelname}ckpt.pth"
    
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

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
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    print(f"Model and arguments loaded from {checkpoint_path}")
    return model, model_args, best_acc, start_epoch


