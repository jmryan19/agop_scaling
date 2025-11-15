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
import os

N_CLASSES = 10
DEVICE = 'cuda:0'
TOTAL_SAMPS = 1_200_000

def create_batch(model, batch_size, w):
    """
    
    """
    with torch.no_grad():
        n_samples = batch_size * N_CLASSES
        X, _ = model.sample(n_samples, (1, 28, 28), DEVICE, guide_w = w)
        del _
    return X

def save_pt(X, save_dir, pt_num):
    labels = torch.arange(10).repeat(X.shape[0] // 10)
    torch.save({"images": X.cpu(), "labels": labels}, save_dir + f'mnist_{pt_num}.pt')

def create_1m(model_path, w, batch_size, save_dir, pt_size):
    n_feat = 256
    n_T = 500
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=N_CLASSES), betas=(1e-4, 0.02), n_T=n_T, device=DEVICE, drop_prob=0.1)
    ddpm.to(DEVICE)

    ddpm.load_state_dict(torch.load(model_path, weights_only=True))

    current_pt = None
    pt_num = 0
    for i in range((TOTAL_SAMPS // batch_size)):
        X = create_batch(ddpm, batch_size, w)
        if current_pt is None:
            current_pt = X
        else:
            current_pt = torch.concat((current_pt, X), dim = 0)

        if current_pt.shape[0] == pt_size:
            print(f'Saving {(pt_num + 1) * pt_size}')
            save_pt(current_pt, save_dir, pt_num)
            current_pt = None
            pt_num += 1

create_1m('model_39.pth', 0.15, 64, './mnist_pts/', 50_000)