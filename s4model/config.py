import argparse

def get_args():
        parser = argparse.ArgumentParser(description="PyTorch S4 Training")

        # Dataset
        parser.add_argument('--modelname', type=str, required=True, help="Name for the model")
        parser.add_argument('--dataset', type=str, help='Dataset ["mnist", "cifar10", "customfullpath"]')
        parser.add_argument('--trainvaltestsplit', type=float, nargs=3, default=[0.7, 0.1, 0.2], help="Train, val, test split")
        parser.add_argument('--trainset', type=str, help="Training dataset path")
        parser.add_argument('--valset', type=str, help="Validation dataset path")
        parser.add_argument('--testset', type=str, help="Test dataset path")
        parser.add_argument('--tabulardata', action='store_true', help="Indicates dataset is tabular (CSV)")
        parser.add_argument('--dependent_variable', type=str, help="Dependent variable name")
        parser.add_argument('--independent_variables', type=str, nargs='+', help="Independent variables")

        # General
        parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
                        
        # Dataloader
        parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
        parser.add_argument('--batch_size', type=int, default=64, help="Batch size")

        # Model
        parser.add_argument('--n_layers', type=int, default=4, help="Number of layers")
        parser.add_argument('--d_model', type=int, default=128, help="Model dimension")
        parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
        parser.add_argument('--prenorm', action='store_true', help="Enable prenorm")

        # Training
        parser.add_argument('--epochs', type=int, default=20, help="Training epochs")
        parser.add_argument('--patience', type=float, default=10, help='Patience for learning rate scheduler')
        parser.add_argument('--grayscale', action='store_true', help='Use grayscale')
        parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
        parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")

        args, unknown = parser.parse_known_args()

        return args, unknown

