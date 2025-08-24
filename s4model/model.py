import torch
import torch.nn as nn
import torch.optim as optim
from models.s4.s4d import S4D
from models.s4.s4 import S4Block as S4  # optional full version
import logging



class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        lr,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        transposed=True,
        prenorm=False,
        dropout_fn=nn.Dropout1d
    ):
        try:
            super().__init__()
            self.prenorm = prenorm

            logging.info(f"Initializing S4Model with d_input={d_input}, d_output={d_output}, d_model={d_model}, n_layers={n_layers}")

            # Linear encoder
            self.encoder = nn.Linear(d_input, d_model)

            # Stack S4 layers as residual blocks
            self.s4_layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.dropouts = nn.ModuleList()

            for i in range(n_layers):
                self.s4_layers.append(
                    S4D(d_model, dropout=dropout, transposed=transposed, lr=min(0.001, lr))
                )
                self.norms.append(nn.LayerNorm(d_model))
                self.dropouts.append(dropout_fn(dropout))

            # Linear decoder
            self.decoder = nn.Linear(d_model, d_output)

            logging.info("S4Model initialized successfully.")

        except Exception as e:
            logging.exception(f"Error initializing S4Model: {e}")
            raise

    def forward(self, x):
        try:
            x = self.encoder(x)
            logging.debug(f"Encoder output shape: {x.shape}")

            x = x.transpose(-1, -2)  # (B, d_model, L)

            for i, (layer, norm, dropout) in enumerate(zip(self.s4_layers, self.norms, self.dropouts)):
                z = x
                if self.prenorm:
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)

                # Apply S4 block
                z, _ = layer(z)

                # Check for NaNs/Infs
                if torch.isnan(z).any() or torch.isinf(z).any():
                    logging.warning(f"NaN or Inf detected in S4 layer {i} output")

                # Dropout
                z = dropout(z)

                # Residual connection
                x = z + x

                if not self.prenorm:
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)

                logging.debug(f"S4 layer {i} output shape after residual: {x.shape}")

            x = x.transpose(-1, -2)
            x = x.mean(dim=1)  # average pooling
            logging.debug(f"Shape after pooling: {x.shape}")

            x = self.decoder(x)
            logging.debug(f"Decoder output shape: {x.shape}")

            return x

        except Exception as e:
            logging.exception(f"Error in forward pass of S4Model: {e}")
            raise

def setup_optimizer(model, lr, weight_decay, epochs):
    try:
        all_parameters = list(model.parameters())

        params = [p for p in all_parameters if not hasattr(p, "_optim")]
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            logging.info(
                ' | '.join([
                    f"Optimizer group {i}",
                    f"{len(g['params'])} tensors"
                ] + [f"{k} {v}" for k, v in group_hps.items()])
            )

        logging.info("Optimizer and scheduler setup completed successfully.")
        return optimizer, scheduler

    except Exception as e:
        logging.exception(f"Error setting up optimizer: {e}")
        raise

def log_gradients(model, threshold_nan=True, threshold_inf=True):
    """
    Logs gradient statistics for all parameters in the model.
    """
    try:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_mean = grad.mean().item()
                grad_max = grad.max().item()
                grad_min = grad.min().item()
                grad_norm = grad.norm().item()

                if threshold_nan and torch.isnan(grad).any():
                    logging.warning(f"NaN detected in gradient of {name}")
                if threshold_inf and torch.isinf(grad).any():
                    logging.warning(f"Inf detected in gradient of {name}")

                logging.debug(
                    f"Grad stats for {name} | mean: {grad_mean:.6f}, max: {grad_max:.6f}, "
                    f"min: {grad_min:.6f}, norm: {grad_norm:.6f}"
                )
    except Exception as e:
        logging.exception(f"Error logging gradients: {e}")


# def train_one_epoch(model, optimizer, scheduler, dataloader, device):
#     model.train()
#     for batch_idx, (x, y) in enumerate(dataloader):
#         try:
#             x, y = x.to(device), y.to(device)

#             optimizer.zero_grad()
#             output = model(x)
#             loss = nn.MSELoss()(output, y)  # example loss
#             loss.backward()

#             # Log gradient stats
#             log_gradients(model, logger)

#             optimizer.step()
#             scheduler.step()

#             logger.info(f"Batch {batch_idx}: loss = {loss.item():.6f}")

#         except Exception as e:
#             logger.exception(f"Error during training at batch {batch_idx}: {e}")
#             continue  # Skip batch but continue training
