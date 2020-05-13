
import config
import torch
import torch.nn as nn
from tqdm import tqdm

"""
LOGIC TO TRAIN AND EVALUATE 1 EPOCH
"""


# TPUs for PyTorch
if config.TPUs:
    import torch_xla.core.xla_model as xm


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def forward_pass(model, ids, mask, token_type_ids):
    """ 
    Abstract forward pass given inputs and model being used 
    """
    if config.MODEL == 'bert':
        return model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
    elif config.MODEL == 'distil-bert':
        return model(
        ids=ids,
        mask=mask
    )
    return None
    

def train_fn(data_loader, model, optimizer, device, scheduler):
    '''
    Train 1 epoch

    Args:
        - data_loader: PyTorch DataLoader  for Training set
        - model: PyTorch Model instance
        - optimizer: PyTorch Optim instance
        - device: Hardward for deployment
        - scheduler: PyTorch learning rate Scheduler instance

    Returns:
        - None
    '''
    model.train()

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data["ids"]
        token_type_ids = data["token_type_ids"]
        mask = data["mask"]
        targets = data["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        if config.TPUs:
            xm.optimizer.zero_grad(optimizer)
        else:
            optimizer.zero_grad()

        # Forward Pass
        outputs = forward_pass(model, ids, mask, token_type_ids)

        # Backward Pass
        loss = loss_fn(outputs, targets)
        loss.backward()
        if config.TPUs:
            xm.optimizer.step()
        else:
            optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    '''
    Evaluate 1 epoch

    Args:
        - data_loader: PyTorch DataLoader for Test set 
        - model: PyTorch Model instance
        - device: Hardward for deployment

    Returns:
        - fin_outputs: Predictions (torch.Tensor)
        - fin_targets: Ground truth (torch.Tensor)
    '''
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = data["ids"]
            token_type_ids = data["token_type_ids"]
            mask = data["mask"]
            targets = data["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets