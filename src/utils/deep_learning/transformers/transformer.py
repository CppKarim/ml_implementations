# Implementation of standard attention mechanisms 
# Cite papers/attention_iayn.pdf, or https://arxiv.org/pdf/1706.03762
from optparse import Option

import torch 
import torch.nn as nn
import torch.func as f
import numpy as np
from tqdm import tqdm

from typing import Optional,List,Tuple
import argparse
from enum import Enum
import time

class transformer_decoder(nn.Module):
    def __init__(self,num_layers,d_model,h,mha_imp,mha_kwargs=None):
        super().__init__()
        if mha_kwargs is None:
            mha_kwargs = {}
        self.model = nn.Sequential()
        for _ in range(num_layers):
            self.model.append(decoder_block(d_model,h,4,mha_imp,mha_kwargs))
    
    def forward(self,input_tuple:dict):
        return self.model(input_tuple)

class transformer_encoder(nn.Module):
    def __init__(self,num_layers,d_model,h,mha_imp,mha_kwargs=None):
        super().__init__()
        if mha_kwargs is None:
            mha_kwargs = {}
        self.model = nn.Sequential()
        for _ in range(num_layers):
            self.model.append(encoder_block(d_model,h,4,mha_imp,mha_kwargs))
    
    def forward(self,input_tuple:Tuple):
        return self.model(input_tuple)

class encoder_block(nn.Module):
    def __init__(self,d_model:int,h:int,ff_expansion:int,mha_imp,mha_kwargs):
        super().__init__()
        self.mha = mha_imp(d_model,h,causal=False,**mha_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model,d_model*ff_expansion),nn.ReLU(),nn.Linear(d_model*ff_expansion,d_model))
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self,input_tuple:Tuple[torch.Tensor,Optional[torch.Tensor]]):
        input_embedding,enc_mask=input_tuple
        # Q,K,V are of shape (batch, seq_len, d_model)
        b,l,d_m = input_embedding.shape
        
        mha_output = self.mha(X=input_embedding,mask=enc_mask)
        output = mha_output+input_embedding
        norm_output = self.norm1(output)
        
        output = self.ff(norm_output)
        output = output + norm_output
        output = self.norm2(output)
        return (output,enc_mask)

class decoder_block(nn.Module):
    def __init__(self,d_model:int,h:int,ff_expansion:int,mha_imp,mha_kwargs):
        super().__init__()
        self.mmha = mha_imp(d_model,h,causal=True,**mha_kwargs)
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = mha_imp(d_model,h,causal=False,**mha_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model,d_model*ff_expansion),nn.ReLU(),nn.Linear(d_model*ff_expansion,d_model))
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self,input_tuple:Tuple[torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]]):
        dec_embedding,enc_embedding,enc_mask,dec_mask = input_tuple
        # Q,K,V are of shape (batch, seq_len, d_model)
        b,l,d_m = dec_embedding.shape
        
        mmha_output = self.mmha(X=dec_embedding,mask=dec_mask)
        output = mmha_output+dec_embedding
        norm_output = self.norm1(output)
        
        mha_output = self.mha(X=enc_embedding,Q=norm_output,mask=enc_mask)
        output = mha_output+norm_output
        norm_output = self.norm2(output)
        
        output = self.ff(norm_output)
        output = output + norm_output
        output = self.norm3(output)

        return (output,enc_embedding,enc_mask,dec_mask)

class mha_vanilla(nn.Module):
    def __init__(self,d_model:int,h:int,causal:Optional[bool]=False):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model//h # d_k=d_v=d_model/h
        self.causal = causal
        assert h * self.d_k == d_model

        self.q_mat = nn.Linear(self.d_model,self.d_model)
        self.k_mat = nn.Linear(self.d_model,self.d_model)
        self.v_mat = nn.Linear(self.d_model,self.d_model)
        self.ll = nn.Linear(self.d_model,self.d_model,bias=True)
    
    def forward(self,X:torch.Tensor,mask:Optional[torch.Tensor]=None,Q:Optional[torch.Tensor]=None):
        # K,V are of shape (batch, seq_len_enc = l_enc, d_model)
        # Q is of shape (batch, seq_len_dec = l, d_model)
        b,l_enc,d_m = X.shape
        if Q is not None:
            _,l,_ = Q.shape
        else:
            l = l_enc
        assert d_m==self.d_model
        if self.causal and mask is None:
            row = torch.arange(0,l,1,device=X.device).unsqueeze(dim=0).unsqueeze(2)
            column = torch.arange(0,l_enc,1,device=X.device).unsqueeze(dim=0).unsqueeze(1)
            # Alternative
            # torch.ones(l, l_enc, device=X.device, dtype=torch.bool).triu(1).unsqueeze(0)
            mask = row<column

        if Q is None:
            Q = self.q_mat(X)
        else: 
            Q = self.q_mat(Q)
        Q = Q.reshape(b,l,self.h,self.d_k).transpose(1,2) # (b,l,h,d_k)
        K = self.k_mat(X)
        K = K.reshape(b,l_enc,self.h,self.d_k).transpose(1,2) # (b,l_enc,h,d_k)
        V = self.v_mat(X)
        V = V.reshape(b,l_enc,self.h,self.d_k).transpose(1,2) # (b,l_enc,h,d_k)
        sdpa_output = sdpa(Q,K,V,mask).transpose(-3,-2) # (b,l,h,d_k)
        concat_output = sdpa_output.reshape(b,l,-1) 
        linear_output = self.ll(concat_output)
        return linear_output

class mha_general(nn.Module):
    def __init__(self,d_model:int,h:int,d_k:int,d_v:int,causal:Optional[bool]=False):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.causal=causal
        assert d_model % h ==0

        self.q_mat = nn.Linear(self.d_model,self.h*self.d_k)
        self.k_mat = nn.Linear(self.d_model,self.h*self.d_k)
        self.v_mat = nn.Linear(self.d_model,self.h*self.d_v)
        self.ll = nn.Linear(self.h*self.d_v,self.d_model,bias=True)
    
    def forward(self,X:torch.Tensor,mask:Optional[torch.Tensor]=None,Q:Optional[torch.Tensor]=None):
        # K,V are of shape (batch, seq_len_enc = l_enc, d_model)
        # Q is of shape (batch, seq_len_dec = l, d_model)
        b,l_enc,d_m = X.shape
        if Q is not None:
            _,l,_ = Q.shape
        else:
            l = l_enc
        assert d_m==self.d_model
        if self.causal and mask is None:
            row = torch.arange(0,l,1,device=X.device).unsqueeze(dim=0).unsqueeze(2)
            column = torch.arange(0,l_enc,1,device=X.device).unsqueeze(dim=0).unsqueeze(1)
            # Alternative
            # torch.ones(l, l_enc, device=X.device, dtype=torch.bool).triu(1).unsqueeze(0)
            mask = row<column

        if Q is None:
            Q = self.q_mat(X)
        else: 
            Q = self.q_mat(Q)
        Q = Q.reshape(b,l,self.h,self.d_k).transpose(1,2) # (b,h,l,d_k)
        K = self.k_mat(X)
        K = K.reshape(b,l_enc,self.h,self.d_k).transpose(1,2) # (b,h,l_enc,d_k)
        V = self.v_mat(X)
        V = V.reshape(b,l_enc,self.h,self.d_v).transpose(1,2) # (b,h,l_enc,d_v)
        sdpa_output = sdpa(Q,K,V,mask).transpose(-3,-2) # (b,l,h,d_v)
        concat_output = sdpa_output.reshape(b,l,-1) 
        linear_output = self.ll(concat_output)
        return linear_output

def sdpa(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:Optional[torch.Tensor]=None):
    # Q is of shape (batch=b, h=h, seq_len_dec=l, d_k)
    # K is of shape (batch=b, h=h, seq_len_enc=l_enc, d_k)
    # V is of shape (batch=b, h=h, seq_len=l_enc, d_v)
    # Mask: True means positions is to be masked
    b,h,l,d_k = Q.shape
    logits = Q@K.transpose(-2,-1)/(d_k**0.5) # (b,h,l,l_enc)
    if mask is not None:
        logits = logits.masked_fill(mask,float('-inf')) 
    att_scores = torch.softmax(logits,dim=-1)
    output = att_scores@V # (b,h,l,d_v)
    return output



if __name__ == "__main__":
    # Initialization
    torch.manual_seed(42)
    torch.set_num_threads(16)
    np.random.seed(42)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    to_profile = False
    
    print("Testing vanilla transformer")
    b,l,d_model,h = 16,18,256,8
    layer = transformer_encoder(num_layers=4,d_model=d_model,h=h,mha_imp=mha_vanilla)
    data = torch.rand((b,l,d_model))
    output = layer((data,None))

    print("Testing general transformer")
    b,l,d_model,h,d_v,d_k = 16,22,256,8,48,52
    mha_kwargs = {"d_v":d_v, "d_k":d_k}
    layer = transformer_encoder(num_layers=4,d_model=d_model,h=h,mha_imp=mha_general,mha_kwargs=mha_kwargs)
    data = torch.rand((b,l,d_model))
    output = layer((data,None))

    print("Testing general transformer in decoder")
    b,l,d_model,h,d_v,d_k = 16,41,256,8,48,52
    mha_kwargs = {"d_v":d_v, "d_k":d_k}
    layer = transformer_decoder(num_layers=4,d_model=d_model,h=h,mha_imp=mha_general,mha_kwargs=mha_kwargs)
    data = torch.rand((b,l,d_model))
    output = layer((data,output[0],None,None))


    print(output)
