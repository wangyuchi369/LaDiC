import random

from configs.config import *
from torch import nn
import torch
from einops import repeat



def batch_loss(model, x_pred,x_t, x_tgt, x_0, mask, idx, loss_func):

    # if CLASSIFIER_FREE_PROB > 0:
    #     classifier_mask = (torch.rand(TRAIN_BATCH_SIZE) > CLASSIFIER_FREE_PROB).type(torch.float32).to(
    #         accelerator.device)  
    #     image = image * (repeat(classifier_mask,'b -> b c h w', c = 3, h = image.shape[2], w=image.shape[3]))
    # x_pred = torch.zeros_like(x_t)
    # object_emb_selfcond = torch.concat([object_emb, object_emb], dim=-1)
    # # add self attentioning
    # if SELF_COND and random.random() > SELF_COND_PROB:
    #     concat_x_t = torch.cat([x_t, x_pred], dim=-1)
    #     concat_x_t = torch.cat([concat_x_t, object_emb_selfcond], dim=-2)
    #     x_pred = model(image, concat_x_t, torch.concat([mask, object_mask], dim=-1), t)
    #     x_pred = x_pred.detach()
    # # x_t restore loss
    # x_pred = model(image, torch.concat([torch.cat([x_t, x_pred], dim=-1), object_emb_selfcond],dim=-2), torch.concat([mask, object_mask], dim=-1), t)
    if USE_X_T_LOSS:
        if X_0_PREDICTION:
            loss_mask = repeat(mask, 'b n -> b n d', d = IN_CHANNEL)
            x_t_loss = loss_func(x_pred, x_0)
            valid_token_loss = (loss_mask * x_pred - loss_mask * x_0).square().sum()/ loss_mask.sum()
            pad_loss = ((1-loss_mask) * x_pred - (1-loss_mask) * x_0).square().sum()/(1 - loss_mask).sum()
        else:
            assert x_tgt.shape == x_t.shape
            x_t_loss = loss_func(x_pred[:, :MAX_LENGTH, :], x_tgt)
    else:
        x_t_loss = 0

    # x_1 restore loss
    # x_1_hidden = model(image, x_1, mask,
    #                              torch.tensor([1], device=accelerator.device))
    # if USE_X_1_LOSS:
    #     x_1_loss = loss_func(x_1_hidden[:, :MAX_LENGTH, :], x_0)
    # else:
    x_1_loss = 0

    if USE_PROB_LOSS:
        input_ids = idx
        out = model.lm_head(model.space_decoder(inputs_embeds=x_pred * X_SIGMA + X_MEAN)[0])
        out = out.reshape(-1, out.shape[-1])
        new_idx = torch.where((input_ids==101)|(input_ids==102)|(input_ids==0), torch.tensor(1012).to(accelerator.device), input_ids).reshape(-1)
        x_t_prob_loss = nn.CrossEntropyLoss()(out, new_idx)
        x_1_prob_loss = 0
    else:
        x_t_prob_loss = 0
        x_1_prob_loss = 0

    return x_t_loss, x_1_loss, ROUNDING_WEIGHT * (x_t_prob_loss + x_1_prob_loss), valid_token_loss, pad_loss
