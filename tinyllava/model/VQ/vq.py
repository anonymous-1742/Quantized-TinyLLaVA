import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VQ_Encoder_img,VQ_Encoder_text
from .decoder import VQ_Decoder_img,VQ_Decoder_text,FSQ_Decoder
from .quantizer import ADSQ,FSQ,FSQ_old,TopKSparse
from .qlora import Qlora_quantize
from .splitfc import FWQ
import numpy as np
from .LearnableBinarization import LearnableBinarizationLayer

import os

class VQ_config:
    def __init__(self,token_dim,discrete_size,embed_dim=2048,code_dim=64,L_comm_cost=0.25,perp_cost=0.1,L_code_cost=0.25,sub_codebook_size=1):
        self.sub_codebook_size=sub_codebook_size
        self.token_dim=token_dim
        self.embed_dim=embed_dim
        self.code_dim=code_dim
        self.discrete_size=discrete_size
        self.comm_cost=L_comm_cost
        self.code_cost=L_code_cost
        self.perp_cost=perp_cost

def pad_sequence_to_length(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
    B, L, H = x.shape
    if L >= target_len:
        return x[:, :target_len, :]
    pad_len = target_len - L
    return F.pad(x, pad=(0, 0, 0, pad_len), value=pad_value)

class empty_VQ(nn.Module):
    def __init__(self):
        super(empty_VQ,self).__init__()
    def forward(self,inputs_embeds,mask=None,return_indice=False):
        return(inputs_embeds,0)

class ADSQ_block(nn.Module):
    def __init__(self,config):
        super(ADSQ_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.code_cost=config.code_cost
        self.quantizer=ADSQ(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        #inputs_embeds=self.encoder(inputs_embeds)
        quantized_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        output_embeds=quantized_embeds
        #output_embeds=self.decoder(quantized_embeds)
        #L_recon=F.mse_loss(inputs_embeds.detach(),output_embeds)
        vq_loss=self.comm_cost*L_comm+self.code_cost*L_code
        return(output_embeds,vq_loss)  
    
class FSQ_block(nn.Module):
    def __init__(self,config):
        super(FSQ_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=FSQ(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds,return_indice)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss)
    
class Qlora_block(nn.Module):
    def __init__(self,config):
        super(Qlora_block,self).__init__()
        bits=int(np.log2(config.discrete_size))
        self.quantizer=Qlora_quantize(bits=bits)
        
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds=self.quantizer(inputs_embeds)
        vq_loss=0
        return(output_embeds,vq_loss)

class TopKSparse_block(nn.Module):
    def __init__(self,config):
        super(TopKSparse_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=TopKSparse(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_mask=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds.to(dtype=inputs_embeds.dtype),vq_loss)  
    
class FSQ_old_block(nn.Module):
    def __init__(self,config):
        super(FSQ_old_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=FSQ_old(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss)
    
class FWQ_block(nn.Module):
    def __init__(self,config):
        super(FWQ_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.quantizer=FWQ(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds,L_code,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm
        return(output_embeds,vq_loss)  
    
class FSQ_AD_block(nn.Module):
    def __init__(self,config):
        super(FSQ_AD_block,self).__init__()
        self.quantizer=FSQ_AD(config.token_dim,config.code_dim,config.discrete_size)
    def forward(self,inputs_embeds):
        output_embeds,L_quant,L_comm=self.quantizer(inputs_embeds)
        vq_loss=self.comm_cost*L_comm+0.1*L_quant
        return(output_embeds,vq_loss)      
    
class FedBAT_block(nn.Module):
    def __init__(self,config):
        super(FedBAT_block,self).__init__()
        self.quantizer=LearnableBinarizationLayer()
    def forward(self,inputs_embeds,return_indice=False):
        output_embeds=self.quantizer(inputs_embeds)
        vq_loss=0
        return(output_embeds,vq_loss)        
    
class VQ(nn.Module):
    def __init__(self,VQ_type,config_text,config_image=None):
        super(VQ,self).__init__()
        self.VQ_type=VQ_type
        if(self.VQ_type=="single"):
            self.quantizer_single=ADSQ_block(config_text)
            self.quantizer_text=empty_VQ()
            self.quantizer_image=empty_VQ()
        elif(self.VQ_type=="double"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=ADSQ_block(config_text)
            self.quantizer_image=ADSQ_block(config_image)
        elif(self.VQ_type=="img"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=empty_VQ()
            self.quantizer_image=ADSQ_block(config_image)
        elif(self.VQ_type=="text"):
            self.quantizer_single=empty_VQ()
            #self.quantizer_text=ADSQ_block(config_text)
            self.quantizer_text=FSQ_block(config_text)
            self.quantizer_image=empty_VQ()
        elif(self.VQ_type=="FSQ"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=FSQ_block(config_text)
            self.quantizer_image=FSQ_block(config_image)
        elif(self.VQ_type=="FSQ_old"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=FSQ_old_block(config_text)
            self.quantizer_image=FSQ_old_block(config_image)
        elif(self.VQ_type=="FWQ"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=FWQ_block(config_text)
            self.quantizer_image=FWQ_block(config_image)
        elif(self.VQ_type=="Qlora"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text=Qlora_block(config_text)
            self.quantizer_image=Qlora_block(config_image)
        elif(self.VQ_type=="TopK"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text= TopKSparse_block(config_text)
            self.quantizer_image=TopKSparse_block(config_image)
        elif(self.VQ_type=="FedBAT"):
            self.quantizer_single=empty_VQ()
            self.quantizer_text= FedBAT_block(config_text)
            self.quantizer_image=FedBAT_block(config_image)
        else:
            self.quantizer_single=empty_VQ()
            self.quantizer_text=empty_VQ()
            self.quantizer_image=empty_VQ()

    def load_model(self,**kwargs):
        pretrained_vq_path = kwargs.get('pretrained_vq_path', None)
        if pretrained_vq_path is not None:
            pretrained_vq_path = os.path.join(pretrained_vq_path, 'pytorch_model.bin')
            vq_weights = torch.load(pretrained_vq_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.quantizer_single.load_state_dict(get_w(vq_weights, 'quantizer_single'))
            self.quantizer_text.load_state_dict(get_w(vq_weights, 'quantizer_text'))
            self.quantizer_image.load_state_dict(get_w(vq_weights, 'quantizer_image'))
            print(f'Loading quantizer from {pretrained_vq_path}...')
