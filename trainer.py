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
    heteroassociate.train(model_type=1, test_t=1, hip_sz=[1024], frcmsk=[.50], data=0, num_seeds=5)

