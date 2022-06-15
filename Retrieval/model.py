import sys
import pickle
import pandas as pd
import numpy as np
import scipy.linalg

import torch 
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from torch.autograd import Function
from torch import linalg as LA
from scipy.linalg import sqrtm

from models.Retrieval.psp import pSp 

class OrthogonalBasis(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        self.args = args
        self.num_classes = args.num_classes # 9 
        self.dim_style = args.dim_style # 512 
        self.num_style = args.num_style # 18
        self.delta = 0.1

        # configure directions 'f' in Paper
        # [18, (M, 512)]
        self.directions = nn.Parameter(torch.randn(self.num_classes, self.dim_style * self.num_style)) # (M, 9126)
        self.bias = nn.Parameter(torch.randn(self.num_classes,)) # (M,)

        path = '/data/hair-retrieval/retrieval_embed_segment.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f) 
        df = pd.DataFrame(data, columns = ['path', 'label', 'embed'])
        self.embeds = torch.from_numpy(np.array(df.iloc[:, 2].tolist(), dtype=np.float))
        self.images = df.iloc[:, 0].tolist()

    def forward(self, x):
        '''
            x : (B, 18, 512)
            directions : (M, 9126)
        '''
        b, _, _ = x.shape 
        x = x.view(b, -1).unsqueeze(-1) # (B, 9126, 1)
        direc = self.directions.unsqueeze(0) # (1, M, 9126)
        direc = direc.repeat(b, 1, 1) # (B, M, 9126)
        direc = direc.type(x.dtype)
        result = torch.bmm(direc, x).squeeze(-1) # (B, M)
        bi = self.bias.unsqueeze(0) # (1, M) 
        bi = bi.repeat(b, 1) # (B, M)
        result = result + bi # (B, M)
        return result # (B, M)


    def algo_1(self):
        direc = self.directions # (M, 9126)
        cs, output = [], []
        for f in direc:
            # f : (9126,)
            f = f.unsqueeze(-1) # (9126, 1)
            c = LA.matrix_norm(f) 
            result = f / c # (9126, 1)
            output.append(result.transpose(1, 0)) # (1, 9126)
            cs.append(c)
        F = torch.concat(output, dim = 0).transpose(1, 0) # output : (9126, M)
        _temp = torch.mm(F.transpose(1, 0).detach().cpu(), F.detach().cpu()).numpy()
        r = sqrtm(_temp)
        r = torch.from_numpy(r)
        _F = torch.mm(F, r.to(F.device).type(F.dtype)) # (9126, M)
        temp = []
        for _f, _c in zip(_F.transpose(1, 0), cs):
            # _f : (9126, )
            # c : () real number
            temp.append((_f).unsqueeze(0))
        temp = torch.concat(temp, dim=0) # (M, 9126)
        return nn.Parameter(temp)


    def distance(self, wq, path):
        # embeds : (N, 18, 512)
        # directions : (18, M, 512)
        # wq : (18, 512)
        embeds = self.embeds

        wq = wq.view(1, -1) # (1, 9126)

        N, _, _ = embeds.shape
        embeds = embeds.view(N, -1) # (N, 9126)

        direc = self.directions.unsqueeze(0) # (1, M, 9126)
        direc = direc.repeat(N, 1, 1) # (N, M, 9126)
        input = embeds.unsqueeze(-1) # (N, 9126, 1)
        
        right = torch.bmm(direc, input.to(direc.device).type(direc.dtype)) # (N, M, 1)
        left = torch.bmm(self.directions.unsqueeze(0), wq.unsqueeze(-1).type(direc.dtype)) # (1, M, 1)
        right = right.squeeze(-1) # (N, M)
        left = left.squeeze(-1).repeat((N, 1)) # (N, M)
        df = (left - right).unsqueeze(-1) # (N, M, 1)

        F = torch.mm(self.directions.transpose(1, 0), self.directions) # (9126, 9126)
        I = torch.eye(9216, 9216)

        _wq = wq.repeat((N, 1)) # (N, 9126)
        _w = embeds # (N, 9126)
        right = (_wq - _w.to(_wq.device)).transpose(1, 0) # (9126, N)

        di = torch.mm((I.to(F.device).type(F.dtype) - F), right.type(F.dtype)) 
        dist = torch.bmm(df.transpose(2, 1), df).squeeze(-1).squeeze(-1) # + LA.matrix_norm(di) # (N)      
        return dist

    def l1_norm(self):
        '''
        regularization : https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        '''
        L1_reg = torch.tensor(0., requires_grad=True)
        direc = self.directions # (M, 9126)
        L1_reg = L1_reg + torch.linalg.norm(direc, 1)
        return L1_reg

class Embedder(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        self.args = args

        ckpt = torch.load(self.args.model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = args.model_path
        opts= Namespace(**opts)
        self.encoder = pSp(opts) 

    def forward(self, img):
        return self.encoder(img, return_latents=True, randomize_noise=False)

class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
