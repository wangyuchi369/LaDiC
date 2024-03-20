import copy
import os

from configs.config import *
from torch import nn
from transformers import (
    BertForMaskedLM, BertConfig, RobertaForMaskedLM, RobertaConfig
)
from transformers.models.bert.modeling_bert import BertLowModel, BertHighModel
from my_utils.blip_util import create_vit, init_tokenizer, load_checkpoint
# from my_utils.util import PosEncoding
from einops import repeat
from diff_models.med import BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

def PosEncoding(d_model, length):
    """
    :param d_model: dimension of the diff_model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

t_emb_lookup = PosEncoding(IN_CHANNEL * 2, STEP_TOT).to(accelerator.device)

class Diffuser(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        super().__init__()
        #origin = BertForMaskedLM.from_pretrained("diff_models/diff_models/bert-base-uncased-local", local_files_only=True)  # .to(device)
        #self.space_encoder, self.space_decoder = copy.deepcopy(origin.bert.requires_grad_(False)), copy.deepcopy(origin.cls.requires_grad_(True))
        #self.space_encoder, self.space_decoder = origin.get_input_embeddings().requires_grad_(False), origin.cls.requires_grad_(False)
        low = BertLowModel.from_pretrained("bert-base-uncased")
        high = BertHighModel.from_pretrained("bert-base-uncased")
        lm_head = BertForMaskedLM.from_pretrained("bert-base-uncased").cls
        # low = BertLowModel.from_pretrained("bert-base-uncased-local")
        # high = BertHighModel.from_pretrained("bert-base-uncased-local")
        # lm_head = BertForMaskedLM.from_pretrained("bert-base-uncased-local").cls
        #bert = BertForMaskedLM.from_pretrained("diff_models/diff_models/bert-base-uncased-local", local_files_only=True).bert
        self.space_encoder, self.space_decoder, self.lm_head = copy.deepcopy(low.requires_grad_(False)), copy.deepcopy(high.requires_grad_(False)),copy.deepcopy(lm_head.requires_grad_(True))
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.load_visual_encoder_weights()
        self.visual_encoder.requires_grad_(False)
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.diffuser = BertModel(config=med_config, add_pooling_layer=False)
        self.diffuser.embeddings.word_embeddings.requires_grad_(False)

    def forward(self, image, x, attention_mask, t):
        bsz, seqlen, dim = x.shape
        if USING_TIME:
            t = t.squeeze()
            t_emb = t_emb_lookup.index_select(0, t) # (bsz, 768)
            # t_emb = self.time_emb_layer(t_emb)
            t_emb = repeat(t_emb, 'b d -> b n d', n = seqlen)
            x = x + t_emb
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.diffuser(inputs_embeds=x,
                                   #attention_mask=attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )
        return output.last_hidden_state[:, :MAX_LENGTH, :IN_CHANNEL]
    
    def load_visual_encoder_weights(self):
        visual_encoder_state_dict = torch.load('pretrained_ckpt/model_base_capfilt_large.pth', map_location='cpu')['model']
        self.load_state_dict(visual_encoder_state_dict, strict=False)

class Diffuser_with_LN(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 ):
        super().__init__()
        #origin = BertForMaskedLM.from_pretrained("diff_models/diff_models/bert-base-uncased-local", local_files_only=True)  # .to(device)
        #self.space_encoder, self.space_decoder = copy.deepcopy(origin.bert.requires_grad_(False)), copy.deepcopy(origin.cls.requires_grad_(True))
        #self.space_encoder, self.space_decoder = origin.get_input_embeddings().requires_grad_(False), origin.cls.requires_grad_(False)
        low = BertLowModel.from_pretrained("bert-base-uncased")
        high = BertHighModel.from_pretrained("bert-base-uncased")
        lm_head = BertForMaskedLM.from_pretrained("bert-base-uncased").cls
        # low = BertLowModel.from_pretrained("bert-base-uncased-local")
        # high = BertHighModel.from_pretrained("bert-base-uncased-local")
        # lm_head = BertForMaskedLM.from_pretrained("bert-base-uncased-local").cls
        #bert = BertForMaskedLM.from_pretrained("diff_models/diff_models/bert-base-uncased-local", local_files_only=True).bert
        self.space_encoder, self.space_decoder, self.lm_head = copy.deepcopy(low.requires_grad_(False)).eval(), copy.deepcopy(high.requires_grad_(False)).eval(),copy.deepcopy(lm_head.requires_grad_(True)).eval()
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.load_visual_encoder_weights()
        self.visual_encoder.requires_grad_(False)
        self.visual_encoder = self.visual_encoder.eval()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.diffuser = BertModel(config=med_config, add_pooling_layer=False)
        self.diffuser.embeddings.word_embeddings.requires_grad_(False)
        #self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.time_layer_norm = nn.LayerNorm(768 * 2, eps=1e-12)

    def forward(self, image, x, attention_mask, t):
        bsz, seqlen, dim = x.shape
        # X = self.layer_norm(x)
        if USING_TIME:
            t = t.squeeze()
            t_emb = t_emb_lookup.index_select(0, t) # (bsz, 768)
            # t_emb = self.time_emb_layer(t_emb)
            t_emb = repeat(t_emb, 'b d -> b n d', n = seqlen)
            t_emb = self.time_layer_norm(t_emb)
            x = x + t_emb
        # import time
        # visual_time = time.time()
        with torch.no_grad():
            image_embeds = self.visual_encoder(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        
        # visual_out = time.time()
        # print('visual time: ', visual_out - visual_time)
        # in_time = time.time()
        output = self.diffuser(inputs_embeds=x,
                                   #attention_mask=attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )
        # out_time = time.time()
        # print('diffuse time: ', out_time - in_time)
        return output.last_hidden_state[:, :MAX_LENGTH, :IN_CHANNEL]
    
    def load_visual_encoder_weights(self):
        visual_encoder_state_dict = torch.load('pretrained_ckpt/model_base_capfilt_large.pth', map_location='cpu')['model']
        self.load_state_dict(visual_encoder_state_dict, strict=False)

# model = Diffuser(image_size=224)
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# batch = tokenizer(['the picture in the day','.','how are you?'], return_tensors='pt', padding=True, truncation=True, max_length=10)
# # a = [tokenizer.decode(b) for b in batch['input_ids']]
# word_emb = model.space_encoder(batch['input_ids'])[0]
# # padding_mask = repeat(batch['attention_mask'], 'b s -> b s 768')
# # word_emb = word_emb * padding_mask
# # epsilon = 0.4
# # noise = (2 * torch.rand_like(word_emb) - torch.ones_like(word_emb)) * epsilon
# # print(noise.abs().mean())
# a1 = model.space_decoder(inputs_embeds = word_emb)[0]
# a2 = model.Bert(batch['input_ids'])[0]
# out = model.lm_head(a2)
# indexes = nn.functional.softmax(out, dim=-1).argmax(dim=-1)
#
# ans_strs = [tokenizer.decode(index) for index in indexes]
# print(ans_strs)