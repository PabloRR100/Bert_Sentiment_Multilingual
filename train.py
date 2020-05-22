
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


# def loss_fn(outputs, targets):
#     return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


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
    

def train_fn(data_loader, model, optimizer, criterion, device, scheduler):
    '''
    Train 1 epoch

    Args:
        - data_loader: PyTorch DataLoader  for Training set
        - model: PyTorch Model instance
        - optimizer: PyTorch Optim instance
        - criterion: Loss Function
        - device: Hardward for deployment
        - scheduler: PyTorch learning rate Scheduler instance

    Returns:
        - None
    '''
    model.train()
    train_loss = 0
    total = correct = 0

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
        loss = criterion(outputs, targets)

        # Perfomance 
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        
        # Backward Pass
        loss.backward()

        if config.TPUs:
            xm.optimizer.step()
        else:
            optimizer.step()
        scheduler.step()

    train_accy = 100.*correct/total
    return train_loss, train_accy


def eval_fn(data_loader, model, criterion, device):
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
    test_loss = 0
    total = correct = 0

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
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accy = 100.*correct/total
    return test_loss, test_accy