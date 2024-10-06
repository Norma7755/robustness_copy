import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from .src.models.sequence.ss.s4 import S4
from .src.models.sequence.ss.fairseq.models.lra.mega_lra_block import MegaLRAEncoder
from s5 import S5Block, S5
from mamba_ssm.modules.mamba_simple import Mamba

class SSM(nn.Module):
    def __init__(self,
                 d_input:int = 1,
                 d_model:int = 128,
                 d_output:int = 10, 
                 n_layers:int = 4, 
                 d_state:int = 64,
                 dropout:int = 0.2,
                 l_max:int = 1,
                 lr:float = 1e-3,
                 use_lyap:bool = False,
                 mode:str = 'nplr',
                 use_inject = False,
                 inject_method=0,
                 patch_size=None
                 ):
        super(SSM, self).__init__()  
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        self.l_max = l_max
        self.adjust = use_lyap
        # self.avg_vol = [0, 0, 0, 0]
        mapping_layers = []
        self.norms = nn.ModuleList()
        self.post_norm = nn.LayerNorm(d_model)
        
        self.use_inject = use_inject
        self.inject_method=inject_method
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)

        if use_lyap: 
            adjusts = []
        for _ in range(self.n_layers):  
            encoder_layer = S4(self.d_model,
                               self.d_state,
                               transposed = False,
                               activation = 'glu',
                               bidirectional=False,
                               mode = mode,
                               lr = lr,
                               use_inject=use_inject,
                               inject_method=self.inject_method)
            mapping_layers.append(encoder_layer)
            if use_lyap:
                adjusts.append(Lyap(d_model))
            self.norms.append(nn.LayerNorm(d_model))
            # mapping_layers.append(nn.LayerNorm(d_model))
        if use_lyap:
            self.adjusts = nn.Sequential(*adjusts)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)


    def forward(self, x, state = None, ret_lyap = False):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        # (B L C) -> (B L D)
        # ori = x
            x = self.input(x)
        # ori = x
        
        x = self.post_norm(x)
        if ret_lyap:
            lyap = 0.0
            
        y_injs = []
        num_layer = 0
        for layer in self.mapping_layers:
            residual = x
            if ret_lyap:
                if num_layer == 0:
                    x, state, y_ori = layer(x, state = state, rety = True)
                    lyap += self.adjusts[num_layer](y_ori, x)
                    # y_old = ori
                else:
                    x, state, y_ori = layer(x, state = state, rety = True)
                    lyap += self.adjusts[num_layer](y_ori, x)
                    # y_old = ori
            else:
                # if use_inject:
                #     x, state, y_injected = layer(x, state = state, \
                #         use_inject=True)
                #     y_injs.append(y_injected)
                # else:
                x, state = layer(x, state = state)
            # print(x.shape)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1
        # (B L D) -> (B D L) -> (B L D)
        # x = self.output_linear(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.output(x.mean(dim=1))
        if ret_lyap:
            return x, lyap/self.n_layers
        else:
            # if use_inject:
            #     return x, y_injs
            # else:
            return x
                
class S5_SSM(nn.Module):
    def __init__(self,
                 d_input:int = 1,
                 d_model:int = 128,
                 d_output:int = 10, 
                 n_layers:int = 4, 
                 d_state:int = 64,
                 dropout:int = 0.2,
                 patch_size:int=None
                 ):
        super(S5_SSM, self).__init__()  
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        mapping_layers = []
        self.norms = nn.ModuleList()
        self.post_norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)

        for _ in range(self.n_layers):  
            encoder_layer = S5(d_model, d_model)
            mapping_layers.append(encoder_layer)
            
            self.norms.append(nn.LayerNorm(d_model))
            # mapping_layers.append(nn.LayerNorm(d_model))
        
        # if patch_size is not None:
        #     self.patch_emb = nn.Conv2d(3, 3, kernel_size=patch_size, stride=patch_size)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.output = nn.Linear(d_model, d_output)


    def forward(self, x, state = None, ret_lyap = False):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        x = self.post_norm(x)
        num_layer = 0
        for layer in self.mapping_layers:
            residual = x
            x = layer(x)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1
        # (B L D) -> (B D L) -> (B L D)
        # x = self.output_linear(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.output(x.mean(dim=1))
        return x
    
    
class Lyap(nn.Module):
    def __init__(self,
                 d_model:int = 128,
                 seq_len:int = 1024, 
                 lmbda:float = 0.01
                 ):
        super(Lyap, self).__init__()  
        
        self.d_model = d_model
        self.lmbda = lmbda
        # self.time_catch = nn.Sequential(nn.Linear(seq_len, seq_len), nn.GELU(), nn.Linear(seq_len, 1), nn.Tanh())
    
        # self.output = nn.Parameter(0.01 * torch.rand((d_model, d_model)))
        # self.time_catch = nn.Sequential(nn.Conv1d(d_model, d_model, 3, 1, 1, bias=False))
        # self.encoder = nn.Sequential(nn.Linear(d_model, d_model, bias=False))
        self.time_catch = nn.Sequential(nn.Conv1d(3, d_model, 3, 1, 1, bias=False), nn.GELU(), nn.Conv1d(d_model, d_model, 3, 1, 1, bias=False))
        self.encoder = nn.Sequential(nn.Linear(d_model, d_model, bias=False), nn.GELU(), nn.Linear(d_model, d_model, bias=False))
    def forward(self, y_ori, x):
        # (B, L, D) -> (B, D, L)
        
        # x = x.transpose(-1, -2)
        # x = self.time_catch(x).squeeze(-1)
        # # print(x.shape)
        # lyap = torch.matmul(x, torch.matmul(self.output, self.output.t()) + torch.eye(self.d_model, device = x.device) * self.lmbda)
        # lyap = torch.matmul(lyap, x.t())
        # lyap = torch.diag(lyap).mean()
        # x = x.transpose(-1, -2)
        # x = self.time_catch(x).transpose(-1, -2)
        # x = self.encoder(x) + y_ori
        # lyap = ((x - y_ori)**2).mean(dim=(-1, -2))
        # lyap = (torch.std(x, dim=-1)**2).sum(-1)
        # lyap = (x.sum(dim=1)**2).mean(dim=1)
        
        lyap = torch.diff(y_ori, dim=1)**2
        lyap = torch.std(lyap, dim=(-2,-1))
        
        return lyap

    
class SSM_Individual_Head(nn.Module):
    def __init__(self,
                 d_input:int = 1,
                 d_model:int = 128,
                 d_output:int = 10, 
                 n_layers:int = 4, 
                 d_state:int = 64,
                 dropout:int = 0.0,
                 l_max:int = 1,
                 lr:float = 1e-3):
        super(SSM_Individual_Head, self).__init__()  
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.dropout = dropout
        self.l_max = l_max
        
        mapping_layers = []
        heads = []
        outputs = []
        self.norms = nn.ModuleList()
        self.input = nn.Linear(d_input, d_model)
        for _ in range(self.n_layers):  
            encoder_layer = S4(self.d_model,
                               self.d_state,
                               transposed = False,
                               activation = 'gelu',
                               bidirectional=False,
                               mode = 'real',
                               lr = lr)
            mapping_layers.append(encoder_layer)
            self.norms.append(nn.LayerNorm(d_model))
            heads.append(nn.Sequential(
            nn.Conv1d(d_model, 2*d_model, kernel_size=1),
            nn.GLU(dim=-2),
        ))
            outputs.append(nn.Linear(d_model, d_output))
            # mapping_layers.append(nn.LayerNorm(d_model))
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.heads = nn.Sequential(*heads)
        self.outputs = nn.Sequential(*outputs)

    def forward(self, x, state = None):
        # (B C H W) -> (B L=H*W C)
        B,C,H,W = x.shape
        x = x.view(B, H*W, C)
        # (B L C) -> (B L D)
        x = self.input(x)
        outputs = []
        num_layer = 0
        for layer in self.mapping_layers:
            residual = x
            x, state = layer(x, state = state)
            x = residual + x
            x = self.norms[num_layer](x)
            x_out = self.heads[num_layer](x.transpose(-1,-2)).transpose(-1,-2)
            x_out = self.outputs[num_layer](x_out).mean(dim=1)
            outputs.append(x_out)
            num_layer += 1
        # (B L D) -> (B D L) -> (B L D)
        return outputs

class Mega(nn.Module):
    def __init__(self, 
                 d_input:int = 3,
                 d_model:int = 128,
                 hidden_dim:int = 256,
                 d_output = 10,
                 n_layers:int = 4,
                 seq_len:int = 1024,
                 patch_size:int=None):
        super(Mega, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        mapping_layers = []
        for _ in range(self.n_layers):
            encoder_layer = MegaLRAEncoder(self.d_model,
                                   hidden_dim = hidden_dim,
                                   ffn_hidden_dim = hidden_dim,
                                   num_encoder_layers = 1,
                                   max_seq_len=seq_len,
                                   dropout = 0.0,
                                   chunk_size = -1,
                                   activation = 'silu'
                                   )
            mapping_layers.append(encoder_layer)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        # self.input = nn.Linear(d_input, d_model)
        self.output = nn.Linear(d_model, d_output)
    
    def forward(self, x, state = None):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        for layer in self.mapping_layers:
            x, _ = layer(x, state)
        output = self.output(x.mean(dim=1))
        return output
    

class S6_SSM(nn.Module):
    def __init__(self, 
                 d_input:int = 3,
                 d_model:int = 128,
                 d_state:int = 16,
                 d_output = 10,
                 n_layers:int = 4,
                 patch_size:int=None):
        super(S6_SSM, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        mapping_layers = []
        self.norms = nn.ModuleList()
        for _ in range(self.n_layers):
            encoder_layer = Mamba(self.d_model,
                               self.d_state,
                               device='cuda:0',
                               dtype=torch.float32,)
            mapping_layers.append(encoder_layer)
            self.norms.append(nn.LayerNorm(d_model))
            
        self.post_norm = nn.LayerNorm(d_model)
        self.patch_size = patch_size
        if patch_size is not None:
            self.input = nn.Conv2d(d_input, d_model, kernel_size=patch_size, stride=patch_size)
        else:
            self.input = nn.Linear(d_input, d_model)
        self.mapping_layers = nn.Sequential(*mapping_layers)
        # self.input = nn.Linear(d_input, d_model)
        self.output = nn.Linear(d_model, d_output)
    
    def forward(self, x, use_inject=False, state = None):
        if self.patch_size is not None:
            x = self.input(x)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
        else:
            # (B C H W) -> (B L=H*W C)
            B,C,H,W = x.shape
            x = x.view(B, H*W, C)
            x = self.input(x)

        x = self.post_norm(x)
        num_layer = 0
        
        for layer in self.mapping_layers:
            residual = x
            x = layer(x, state)
            x = residual + x
            x = self.norms[num_layer](x)
            num_layer += 1          
        
        output = self.output(x.mean(dim=1))
        # return output
        
        return output