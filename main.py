
import torch
import config
import dataset
from model import BERTBaseUncased
from train import train_fn, eval_fn
from utils import print_current_config
from beautifultable import BeautifulTable as BT

import numpy as np
import pandas as pd
from sklearn import metrics, model_selection

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler


""" CONFIG """
CUDA = torch.cuda.is_available()
N_GPU = torch.cuda.device_count()
DEVICE = 'cuda' if CUDA else 'cpu'
WORKERS = torch.multiprocessing.cpu_count()
print_current_config(CUDA, N_GPU, DEVICE, WORKERS)

exit()

# TPUs for PyTorch
if config.TPUs:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl

def run():
    '''
    Entire training loop
        - Create DataLoaders
        - Define Training Configuration
        - Launch Training Loop
    '''

    # Num of available TPU cores
    if config.TPUs:
        n_TPUs = xm.xrt_world_size()
        DEVICE = xm.xla_device()
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(DEVICE)
    
    # Read Data
    
    df1 = pd.read_csv('data/jigsaw-toxic-comment-train.csv', usecols=['comment_text', 'toxic'])
    df2 = pd.read_csv('data/jigsaw-unintended-bias-train.csv', usecols=['comment_text', 'toxic'])
    df_train = pd.concat([df1,df2], axis=0).reset_index(drop=True)
    df_valid = pd.read_csv('data/validation.csv')
    
    # Preprocess
    
    train_dataset = dataset.BERTDataset(
        comment=df_train.comment_text.values,
        target=df_train.toxic.values
    )

    valid_dataset = dataset.BERTDataset(
        comment=df_valid.comment_text.values,
        target=df_valid.toxic.values
    )

    drop_last=False
    train_sampler, valid_sampler = None, None
    if config.TPUs:
        drop_last=True
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=n_TPUs,
            rank=xm.get_ordinal(),
            shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_dataset, 
            num_replicas=n_TPUs,
            rank=xm.get_ordinal(),
            shuffle=True
        )


    # Create Data Loaders

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        drop_last=drop_last,
        sampler=train_sampler
    )


    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1,
        drop_last=drop_last,
        sampler=valid_sampler
    )

    # Machine Configuration

    model = BERTBaseUncased()
    model.to(device)
    
    # Optimizer Configuration 

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    lr = config.LR
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    # TODO: why do the LR increases because of a distributed training ?
    if config.TPUs:
        num_train_steps /= n_TPUs
        lr *= n_TPUs

    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    if not config.TPUs:
        model = nn.DataParallel(model)

    # Training loop

    best_score = 0
    
    for epoch in range(config.EPOCHS):
    
        if config.TPUs:
            train_loader = pl.ParallelLoader(train_data_loader, [device])
            valid_loader = pl.ParallelLoader(valid_data_loader, [device])
            train_fn(train_loader.per_device_loader(device), model, optimizer, device, scheduler)
            outputs, targets = eval_fn(valid_loader.per_device_loader(device), model, device)

        else:
            train_fn(train_data_loader, model, optimizer, device, scheduler)
            outputs, targets = eval_fn(valid_data_loader, model, device)
        
        targets = np.array(targets) >= 0.5 # TODO: why ?
        auc_score = metrics.roc_auc_score(targets, outputs)
            
        # Save if best
        print(f"AUC Score = {auc_score}")
        if auc_score > best_score:
            if not config.TPUs:
                torch.save(model.state_dict(), config.MODEL_PATH)
            else:
                xm.save(model.state_dict(), config.MODEL_PATH)
            best_score = auc_score


if __name__ == "__main__":
    run()