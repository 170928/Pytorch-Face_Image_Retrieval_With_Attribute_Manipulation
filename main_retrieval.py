import os
import sys
import math
import logging 
import easydict
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torchmetrics import HingeLoss

from tqdm import tqdm
from datetime import datetime 

from dataset.retrieval import get_loaders
from models.Retrieval.model import OrthogonalBasis
from log.log import set_logger


def train_one_epoch(model, 
                    criterion,
                    data_loader, 
                    optimizer,
                    lr_scheduler,
                    device, 
                    epoch, 
                    save_path,
                    max_epoch):

    model.train()
    model.to(device)
    print_freq = 10
    count = 0
    Batch = len(data_loader)
    loss_mean = 0.0

    for _, samples, targets in data_loader:
        count +=1
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)

        for idx in range(18):
            if idx == 0:
                loss = criterion(outputs[:, idx, :], targets.type(outputs.dtype)) + model.l1_norm() * 0.005 # lambda * l1_norm 
            else:
                loss += criterion(outputs[:, idx, :], targets.type(outputs.dtype))

        loss_value = loss.item()
        loss_mean += loss_value 

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        if count % print_freq == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            loss_mean = 0.0

    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                }, f'{save_path}/checkpoint.pth')


@torch.no_grad()
def evaluate(data_loader, criterion, model, device, best, save_path):

    # switch to evaluation mode
    model.eval()
    loss_mean = 0.0
    Batch = len(data_loader)
    
    for paths, images, target in data_loader:
        images = images.to(device, non_blocking=True)
        targets = target.to(device, non_blocking=True)

        outputs = model(images)

        for idx in range(18):
            if idx == 0:
                loss = criterion(outputs[:, idx, :], targets.type(outputs.dtype)) + model.l1_norm() * 0.005 # lambda * l1_norm 
            else:
                loss += criterion(outputs[:, idx, :], targets.type(outputs.dtype))

        loss_mean += loss.item()

    for path, img, tar in zip(paths[0], images, targets):
        dist = model.distance(img, path)[-1]

    #print(f"Evaluation Loss : {loss_mean / Batch}")
    if best > loss_mean / Batch:
        best = loss_mean / Batch
        os.makedirs(f'{save_path}', exist_ok=True)
        torch.save({'model': model.state_dict()}, f'{save_path}/checkpoint.pth')
    return best

def argdict():
    args = easydict.EasyDict({
        'batch_size' : 32,
        'max_epoch' : 100000,
        'num_classes' : 9,
        'dim_style' : 512,
        'num_style' : 18, 
        'model_name' : 'retrieval',
        'save_path' : '/workspace/cv_engine/output/retrieval',
        'model_path' : '/workspace/cv_engine/pretrained/psp_ffhq_encode.pt',
        'experiment_name' : datetime.now().strftime("%Y-%d-%B-%p-%M-%I-%S")
    })
    return args

def execute():

    # get argument
    args = argdict()

    # generate folders for save results

    saved_model_path = f"{args.save_path}/{args.experiment_name}"
    print(f"Save Path is {saved_model_path} ...")
    os.makedirs(os.path.join(f'output/{args.model_name}', args.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(f'output/{args.model_name}', args.experiment_name, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(f'output/{args.model_name}', args.experiment_name, 'checkpoint', 'best'), exist_ok=True)

    set_logger(saved_model_path)

    # configure transformation; and loaders
    loaders = get_loaders(args.batch_size, 0, None, None, all=False, use_embed=True)
    
    # create model
    model = OrthogonalBasis(args).cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    lr = 0.001
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)

    criterion = HingeLoss()

    print("Start training")
    best = np.inf
    max_epoch = args.max_epoch
    for epoch in range(0, max_epoch):

        train_stats = train_one_epoch(
            model, 
            criterion, 
            loaders['train'],
            optimizer, 
            lr_scheduler,
            'cuda', 
            epoch, 
            save_path=os.path.join(f'output/{args.model_name}', args.experiment_name, 'checkpoint'),
            max_epoch=max_epoch
        )

        lr_scheduler.step()
        best = evaluate(loaders['valid'], 
                criterion,
                model,
                'cuda',
                best=best, 
                save_path=os.path.join(f'output/{args.model_name}', args.experiment_name, 'checkpoint', 'best'))



if __name__=="__main__":
    execute()


