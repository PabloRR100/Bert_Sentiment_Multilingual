

import os
import sys
import time
import json
import yaml
import torch
from os.path import join as JP
from collections import OrderedDict
from beautifultable import BeautifulTable as BT


# =============================================================================
# General Utils
# ============================================================================= 

def print_current_config(CUDA, N_GPU, DEVICE, WORKERS):
    # CONFIG 
    MULTI_GPU = True if N_GPU > 1 else False
    # WORKERS = 1
    DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPUs'

    table = BT()
    table.append_row(['Python', sys.version[:5]])
    table.append_row(['PyTorch', torch.__version__])
    table.append_row(['GPUs', str(N_GPU)])
    table.append_row(['Cores', str(WORKERS)])
    table.append_row(['Device', str(DEVICE_NAME)])
    print('Environment Settings')
    print(table)



class Results(object):
    
    def __init__(self, name=None):
        self.name = name                
        self.time = list()
        self.epoch = list()
        self.train_loss = list()
        self.train_accy = list()
        self.valid_loss = list()
        self.valid_accy = list()