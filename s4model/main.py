#%%
# Set up arguments first
from config import get_args
args, unknown = get_args()
print('Training Inputs:', args)
print('Unknown Inputs:', unknown)

#%%
#Import librariries
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import time 
from datetime import datetime
from tqdm import tqdm

from dataset import load_data
from output import combine_results_to_dataframe
from model import S4Model, setup_optimizer
from train import train, eval, load_model

#%%
#Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True if device == 'cuda' else False
print(f"Using {device} device")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
dropout_fn = nn.Dropout1d # handles regularization

#%%
# Load dataset
print(f'==> Preparing data..')
trainloader, valloader, testloader, d_input, d_output, trainset, valset, testset, df, train_df, test_df = load_data(args)

# Add input & output to the argparse namespace
args.d_input = d_input
args.d_output = d_output

#%%
# Initialize model, optimizer, and loss function
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    lr = args.lr,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm    
        ).to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)

#%%
#Checkpoints

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    model, model_args, best_acc, start_epoch = load_model(S4Model, args.modelname)
print(args)
#%%

# Reiterate thorugh epochs to train and validate the model
# 1st Batch - Train, # 2nd Batch - Validation, # 3rd Batch - Test
     
print('Start Time: ',datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
start_time = time.perf_counter()
pbar = tqdm(range(start_epoch,args.epochs))
for epoch in pbar:    
    train(model, trainloader, criterion, optimizer, device)
    val_acc, val_target, val_predicted = eval(model, valloader, criterion, device, epoch, args.modelname, best_acc, args, checkpoint=True)    
    test_acc, test_target, test_predicted = eval(model, testloader, criterion, device, epoch, args.modelname, best_acc, args, checkpoint=True)        
    scheduler.step()    
    pbar.set_description('Epoch: %d | Val acc: %.3f%% | Test acc: %.3f%%' % (epoch + 1, val_acc, test_acc))

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print('End Time: ', datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
print(f"Process completed in {elapsed_time:.6f} seconds")  # Higher precision timing    

#%%
#Save the validation and test datasets
if args.dataset == 'cifar10' or args.dataset == 'mnist':    
    dep_var = None
    val_name = None
    test_name = None
else:    
    dep_var = args.dependent_variable    
    val_name = train_df if df is None else df
    test_name = test_df if df is None else df


df_validation = combine_results_to_dataframe(valloader, val_target, val_predicted, dependent_variable =dep_var, valset=valset, original_df=val_name, name = f'{args.modelname}_Validation')
df_test = combine_results_to_dataframe(testloader, test_target, test_predicted, dependent_variable =dep_var, valset=testset, original_df=test_name, name = f'{args.modelname}_Test')
#%%
# Inference