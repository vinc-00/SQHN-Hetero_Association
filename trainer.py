import Unit
import math
import torch
from torch import nn
import torchvision
import numpy as np
import pickle
import utilities
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import heteroassociate



''' 
This file contains function for running the tests presented in the paper. These functions are used
by the main function to run each test with the hyperparameters used in the paper. No hyperparam
searches or adjustments are needed.
'''


#Initial Auto and Hetero Associative Memory Tests
def modCompareAA():
    # Comparison to PCN MHN
    print(f'\n\n Moderate Corrupt ')
    heteroassociate.train(model_type=3, test_t=1, hip_sz=[1024], frcmsk=[.25], data=2, num_seeds=5)

    # Comparison to PCN MHN high corruption
    print(f'\n\n High Corrupt ')
    heteroassociate.train(model_type=3, test_t=1, hip_sz=[128], frcmsk=[.50], data=2, num_seeds=7, rec_thr=.001)
