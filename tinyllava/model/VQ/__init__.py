import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import VQ_Encoder_img,VQ_Encoder_text
from .decoder import VQ_Decoder_img,VQ_Decoder_text

class VQ_config:
    def __init__(self,token_dim,discrete_size,embed_dim=4096,nheads=10,dropout=0.1,num_layers=4,L_comm_cost=0.25,perp_cost=0.25,L_code_cost=1,recon_cost=1,n_res_layers=4):
        self.token_dim=token_dim
        self.embed_dim=token_dim*4
        self.discrete_size=discrete_size
        self.nheads=nheads
        self.dropout=dropout
        self.num_layers=num_layers
        self.comm_cost=L_comm_cost
        self.code_cost=L_code_cost
        self.perp_cost=perp_cost
        self.recon_cost=recon_cost
        self.n_res_layers=n_res_layers
        self.res_dim=token_dim
    def to_dict(self):
        return {
            'embed_dim': self.embed_dim,
            'discrete_size': self.discrete_size,
            'nheads': self.nheads,
            'dropout': self.dropout,
            'num_layers': self.num_layers
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            embed_dim=data['embed_dim'],
            discrete_size=data['discrete_size'],
            nheads=data.get('nheads', 10),          # 提供默认值
            dropout=data.get('dropout', 0.1),
            num_layers=data.get('num_layers', 8)
        )

class empty_VQ(nn.Module):
    def __init__(self):
        super(empty_VQ,self).__init__()
    def forward(self,inputs_embeds):
        return(inputs_embeds,0)
    
class VQ_block(nn.Module):
    def __init__(self,config):
        super(VQ_block,self).__init__()
        self.comm_cost=config.comm_cost
        self.code_cost=config.code_cost
        self.perp_cost=config.perp_cost
        self.recon_cost=config.recon_cost
        self.encoder=VQ_Encoder(config.token_dim,config.embed_dim,config.n_res_layers,config.res_dim)
        self.quantizer=Quantizer(config)
        self.decoder=VQ_Decoder(config.embed_dim,config.token_dim,config.n_res_layers,config.res_dim)
    def forward(self,inputs_embeds):
        original_iputs=inputs_embeds.detach()
        encoded_embeds=self.encoder(inputs_embeds)
        quantized_embeds,L_comm,L_code,L_perp=self.quantizer(encoded_embeds)
        output_embeds=self.decoder(quantized_embeds)
        L_recon=F.mse_loss(output_embeds,original_inputs)
        vq_loss=self.recon_cost*L_recon+self.comm_cost*L_comm+self.code_cost*L_code-self.perp_cost*L_perp
        return(output_embeds,vq_Loss)
    
class VQ_VAE(nn.Module):
    def __init__(self,VQ_type,config_text,config_image=None):
        super(VQ_VAE,self).__init__()
        self.VQ_type=VQ_type
        if(self.VQ_type=="single"):
            self.quantizer=VQ_block(config_text)
            self.quantizer_text=empty_VQ()
            self.quantizer_image=empty_VQ()
        elif(self.VQ_type=="double"):
            self.quantizer=empty_VQ()
            self.quantizer_text=VQ_block(config_text)
            self.quantizer_image=v(config_image)
        elif(self.VQ_type=="img"):
            self.quantizer=empty_VQ()
            self.quantizer_text=empty_VQ()
            self.quantizer_image=VQ_block(config_image)
        elif(self.VQ_type=="text"):
            self.quantizer=empty_VQ()
            self.quantizer_text=VQ_block(config_text)
            self.quantizer_image=empty_VQ()
        else:
            self.quantizer=empty_VQ()
            self.quantizer_text=empty_VQ()
            self.quantizer_image=empty_VQ()

    def load_model(self,**kwargs):
        pretrained_VQ_path = kwargs.get('pretrained_vq_path', None)
        if pretrained_vq_path is not None:
            pretrained_vq_path = os.path.join(pretrained_vq_path, 'pytorch_model.bin')
            vq_weights = torch.load(pretrained_vq_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.quantizer.load_state_dict(get_w(vq_weights, 'quantizer'))
            self.quantizer_text.load_state_dict(get_w(vq_weights, 'quantizer_text'))
            self.quantizer_image.load_state_dict(get_w(vq_weights, 'quantizer_image'))
            print(f'Loading connector from {pretrained_vq_path}...')
