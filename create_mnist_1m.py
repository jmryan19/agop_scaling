from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from DDPM import *

N_CLASSES = 10
DEVICE = ''
TOTAL_SAMPS = 1_200_000

def create_batch(batch_size, model, w):
    """
    
    """
    n_samples = batchsize * N_CLASSES
    X, _ = model.sample(n_samples, (1, 28, 28), DEVICE, guide_w = w)
    return X

def save_pt(X, save_dir):
    pass

def create_1m(model_path, w, batch_size, save_dir, pt_size):
    n_feat = 256
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=N_CLASSES), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(DEVICE)

    ddpm.load_state_dict(torch.load(model_path, weights_only=True))

    current_pt = None
    for i in range((TOTAL_SAMPS // batch_size)):
        X = create_batch(batch_size, model, w)
        if current_pt is None:
            current_pt = X
        else:
            current_pt = torch.concat((current_pt, X), dim = 0)

        if current_pt.shape[0] == pt_size:
            save_pt(current_pt, save_dir)
            current_pt = None