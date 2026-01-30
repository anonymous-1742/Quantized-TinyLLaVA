from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
from torch.nn import CrossEntropyLoss
from . import LLMFactory, ConnectorFactory, VisionTowerFactory
from .configuration_tinyllava import TinyLlavaConfig
from ..utils.constants import *
from .VQ.vq_split import VQ_config,VQ_split
# from tinyllava.utils.data_utils import get_value_from_kwargs
from deepspeed.runtime.zero import GatheredParameters

    
def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None


            
class TinyLlavaPreTrainedModel(PreTrainedModel):
    config_class = TinyLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa

class TinyLlavaSplitClient(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        super().__init__(config)
        self.config = config

        self.language_model = LLMFactory(
            config.llm_model_name_or_path
        )[0](config.text_config)

        self.vision_tower = VisionTowerFactory(
            config.vision_model_name_or_path
        )(config.vision_config)

        self.connector = ConnectorFactory(config.connector_type)(config)

        self.vq_config_text = VQ_config(
            token_dim=config.hidden_size,
            code_dim=config.hidden_size,
            discrete_size=config.discrete_size
        )
        self.vq_config_image = VQ_config(
            token_dim=config.hidden_size,
            code_dim=config.hidden_size,
            discrete_size=config.discrete_size
        )

        self.VQ = VQ_split(
            config.vq_type,
            self.vq_config_text,
            self.vq_config_image
        ) 
                
        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        labels=None,
        images=None,
        image_sizes=None,
        return_indice=False,
        ):
        (
            input_ids,
            position_ids,
            attention_mask,
            _,
            new_input_embeds,
            new_labels,
            text_positions
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=labels,
            images=images,
            image_sizes=image_sizes,
            return_indice=return_indice,
        )
        text_embeds=new_input_embeds[text_positions] #original embedding
        image_embeds=new_input_embeds[~text_positions]
        image_embeds=self.VQ.quantizer_image.quantizer.encode(image_embeds)#projected and scaled embedding
        text_embeds=self.VQ.quantizer_image.quantizer.encode(text_embeds)
        
        image_payloads,image_aux,vq_image_loss=self.VQ.quantizer_image.quantizer.compress(image_embeds) #quantize payload
        vq_image_loss=vq_image_loss*self.VQ.quantizer_image.comm_cost
        text_payloads,text_aux,vq_text_loss=self.VQ.quantizer_text.quantizer.compress(text_embeds)
        vq_text_loss=vq_text_loss*self.VQ.quantizer_text.comm_cost
        transmission={
            "image_payloads": image_payloads,
            "text_payloads": text_payloads,
            "image_aux": image_aux,
            "text_aux": text_aux,
            "attention_mask": attention_mask.to("cpu"),
            "position_ids": position_ids,
            "labels": new_labels,
            "text_positions":text_positions
        }
        vq_loss=vq_image_loss+vq_text_loss
        
        return transmission,vq_loss,image_embeds,text_embeds
    
    def encode_images(self, images):
        kwargs = {}
        kwargs['vision_feature_layer'] = self.config.vision_feature_layer
        kwargs['vision_feature_select_strategy'] = self.config.vision_feature_select_strategy
        #with GatheredParameters(self.vision_tower.parameters(), modifier_rank=0):
        #    image_features = self.vision_tower(images, **kwargs)
        #device = next(self.connector.parameters()).device
        #image_features = image_features.to(device)
        image_features = self.vision_tower(images, **kwargs)
        image_features = self.connector(image_features)
        return image_features
    
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None,return_indice=True
    ):
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            print("model not executed")
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_features= self.encode_images(images)
        #image_features,vq_loss_img = self.VQ.quantizer_image(image_features,return_indice)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        vq_loss_text=0
        image_token_index=[]
        cur_image_idx = 0
        batch_size=0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            batch_size+=1
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            cur_image_token_index=[]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                
                #cur_token=torch.ones(cur_image_features.shape[0])
                #cur_image_token_index.append(cur_token)
                #with GatheredParameters(self.language_model.get_input_embeddings().parameters(), modifier_rank=0):
                #    cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                #cur_token=torch.zeros(cur_input_embeds_1.shape[0])
                #cur_image_token_index.append(cur_token)
                cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                cur_token=torch.zeros(cur_input_embeds_1.shape[0])
                image_token_index.append(cur_token)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            #with GatheredParameters(self.language_model.get_input_embeddings().parameters(), modifier_rank=0):
             #   cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_token=torch.zeros(cur_input_embeds_no_im[i].shape[0])
                cur_image_token_index.append(cur_token)
                
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    
                    cur_token=torch.ones(cur_image_features.shape[0])
                    cur_image_token_index.append(cur_token)
                    
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_image_token_index=torch.cat(cur_image_token_index)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            image_token_index.append(cur_image_token_index)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            image_token_index=[x[:tokenizer_model_max_length] for x in image_token_index]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        image_token_index_padded=[]
        for i, (cur_new_embed, cur_new_labels,cur_image_token_index) in enumerate(zip(new_input_embeds, new_labels,image_token_index)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                image_token_index_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len), dtype=cur_image_token_index.dtype, 
                                device=cur_image_token_index.device),
                    cur_image_token_index
                ),dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                image_token_index_padded.append(torch.cat((
                    cur_image_token_index,
                    torch.zeros((max_len - cur_len), dtype=cur_image_token_index.dtype,
                                device=cur_image_token_index.device)
                ),dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        image_token_index=torch.stack(image_token_index_padded,dim=0).bool()
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        text_token_index=~image_token_index
        text_embeds=new_input_embeds[text_token_index].view(new_input_embeds.shape[0],-1,new_input_embeds.shape[2])
        #quantized_text_embeds,vq_loss_text=self.VQ.quantizer_text(text_embeds,return_indice).contiguous().view(-1,new_input_embeds.shape[2])
        #new_input_embeds[text_token_index]=quantized_text_embeds
        return input_ids, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels,text_token_index
    
    
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        from_pretrain = get_value_from_kwargs(kwargs, 'from_pretrain')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)
        
        
    def load_vision_tower(self, **kwargs):
        vision_tower_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        print("vision tower name:",vision_tower_name)
        self.vision_tower.load_model(vision_tower_name, **kwargs)

        
    def load_connector(self, **kwargs):
        self.connector.load_model(**kwargs)
        
    def load_vq(self, **kwargs):
        self.VQ.load_model(**kwargs)

        
class TinyLlavaSplitServer(TinyLlavaPreTrainedModel):
    def __init__(self, config: TinyLlavaConfig):
        super().__init__(config)
        self.language_model = LLMFactory(
            config.llm_model_name_or_path
        )[0](config.text_config)
        self.vq_config_text = VQ_config(
            token_dim=config.hidden_size,
            code_dim=config.hidden_size,
            discrete_size=config.discrete_size
        )
        self.vq_config_image = VQ_config(
            token_dim=config.hidden_size,
            code_dim=config.hidden_size,
            discrete_size=config.discrete_size
        )

        self.VQ = VQ_split(
            config.vq_type,
            self.vq_config_text,
            self.vq_config_image
        )
        (Tokenizer, post_load) = LLMFactory(config.llm_model_name_or_path)[1]
        self.tokenizer = post_load(Tokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir = config.cache_dir,
            model_max_length = config.tokenizer_model_max_length,
            padding_side = config.tokenizer_padding_side,
            use_fast = config.tokenizer_use_fast,
        ))
    
    def decompress_embeds(self,image_payloads,image_aux,text_payloads,text_aux):
        image_embeds=self.VQ.quantizer_image.quantizer.decompress(image_payloads,image_aux) #dequantized embedding
        text_embeds=self.VQ.quantizer_text.quantizer.decompress(text_payloads,text_aux)
        return image_embeds,text_embeds
    
    def concat_embeds(self,image_embeds,text_embeds,text_positions):
        image_embeds=self.VQ.quantizer_image.quantizer.decode(image_embeds) #out porjected embedding
        text_embeds=self.VQ.quantizer_text.quantizer.decode(text_embeds)
        B, L_total = text_positions.shape
        H = image_embeds.shape[-1]
        input_embeds = torch.zeros((B, L_total, H), device=image_embeds.device, dtype=image_embeds.dtype)
        input_embeds[text_positions] = text_embeds.view(-1, H)#concated embedding
        input_embeds[~text_positions] = image_embeds.view(-1, H)

        return input_embeds
    
    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        labels=None,
        use_cache=None,
        output_hidden_states=None,
        output_attentions=None,
        ):
        outputs = self.language_model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return outputs
    def load_llm(self, **kwargs):
        language_model_name = get_value_from_kwargs(kwargs, 'model_name_or_path')
        pretrained_llm_path = get_value_from_kwargs(kwargs, 'pretrained_llm_path')
        from_pretrain = get_value_from_kwargs(kwargs, 'from_pretrain')
        if pretrained_llm_path is not None:
            language_model_name = pretrained_llm_path
        if language_model_name is not None:
            self.language_model = self.language_model.from_pretrained(
                language_model_name, **kwargs
            )
        print('loading language model from ', language_model_name)
        self.language_model.requires_grad_(False)
        
        self.config.text_config.torch_dtype = kwargs.get('torch_dtype', None)
        self.config.pad_token = getattr(self.tokenizer, 'pad_token', None)
        self.config.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        #self.config.tokenizer_padding_side = getattr(self.tokenizer, 'padding_side', None)
        #self.config.tokenizer_model_max_length =  getattr(self.tokenizer, 'model_max_length', None)

    def load_vq(self, **kwargs):
        self.VQ.load_model(**kwargs)

        
        
