import torch
from tqdm import tqdm
import sys
import os
# from dataload.dataloader import val_set, val_loader
from diff_models.diffusion import *
from torch import nn
from diff_models.diffcap_model import Diffuser, Diffuser_with_LN
from my_utils.blip_util import load_checkpoint
import json
device = torch.device('cuda:0')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
from evaluate import load
torch.backends.cudnn.benchmark = False

import time

def inference(x, tokenizer, model, time_difference = 0):
    x_t = torch.randn((x["image"].shape[0], MAX_LENGTH , IN_CHANNEL), device=device)  # Gaussian noise (bsz, seqlen, 768)
    # each prediction involves multiple generation steps
    x_pred = torch.zeros_like(x_t, device=device)
    STEP = 200
    X_SIGMA.to(device)
    X_MEAN.to(device)
    time_start = time.time()
    t = STEP_TOT - 1
    flag = False
    while t > 0:
        t_diff = min(STEP_TOT - 1, t + time_difference)
        if not SELF_COND:
            x_pred = torch.zeros_like(x_t, device=device)
        cond_pred = model(x['image'].to(device), torch.cat([x_t, x_pred], dim=-1).to(device),
                              torch.ones((x["image"].shape[0], MAX_LENGTH), device=device),
                           torch.tensor([t_diff], device=device))
        # out1 = model.space_decoder(cond_noise)
        # indexes1 = nn.functional.softmax(out1, dim=-1).argmax(dim=-1)
        # cond_noise = model.space_encoder(indexes1)[0]
        uncond_pred = model( torch.zeros_like(x["image"], device=device), torch.cat([x_t, x_pred], dim=-1).to(device),
                                torch.ones((x["image"].shape[0], MAX_LENGTH), device=device),
                                # torch.tensor([1, 0], device=device).repeat(x["attention_mask"].shape[0], 1),
                                torch.tensor([t_diff], device=device))
        x_pred = (1 + CLASSIFIER_FREE_WEIGHT) * cond_pred - CLASSIFIER_FREE_WEIGHT * uncond_pred
        # x_pred = cond_pred
        if t < 600 and t > 300 and flag:
            tmp_out = model.lm_head(model.space_decoder(inputs_embeds=x_pred * X_SIGMA + X_MEAN)[0])
            softmax_tmp = nn.functional.softmax(tmp_out, dim=-1)
            # most_confident_token =softmax_tmp.max(dim=-1).values.argmax(dim=-1)
            confidence = softmax_tmp.max(dim=-1).values
            _, idx = torch.sort(confidence, descending=False)
            to_be_updated_idx = idx[:,:MAX_LENGTH//3].to(device)
            gaussian_noise = torch.randn_like(x_pred).to(x_pred.device)
            # x_pred[to_be_updated_idx, :] = gaussian_noise[to_be_updated_idx, :].clone()
            x_t = diffuse_t(x_pred, torch.tensor([t], device=device) - STEP)
            x_t[torch.arange(x_pred.shape[0])[:, None], to_be_updated_idx] = gaussian_noise[torch.arange(x_t.shape[0])[:, None], to_be_updated_idx].clone()
            # indexes1 = nn.functional.softmax(out1, dim=-1).argmax(dim=-1)
            # pred_x0 = (model.space_encoder(indexes1)[0] - X_MEAN)/X_SIGMA
            t = STEP_TOT - 1
            flag = False
        elif t > STEP:
            # noise = pred_x0
            x_t = diffuse_t(x_pred, torch.tensor([t], device=device) - STEP)
            #x_t = p_sample(x_t[:, :MAX_LENGTH, :], x_pred, torch.tensor([t], device=device) , STEP)
        t -= STEP
    cond_pred = x_pred * X_SIGMA + X_MEAN
    out = model.lm_head(model.space_decoder(inputs_embeds=cond_pred)[0])
    indexes = nn.functional.softmax(out, dim=-1).argmax(dim=-1)
    indexes = indexes.unique_consecutive(dim=-1)
    import itertools

    ans_strs = [tokenizer.decode(index) for index in indexes]
    time_end = time.time()
    # print('time cost', time_end - time_start, 's')
    ans_strs = [' '.join(k for k, _ in itertools.groupby(original_str.split())) for original_str in ans_strs]
    ans_strs = [original_str.strip('.').strip() + '.' for original_str in ans_strs]
    
    return ans_strs, x['img_id']


def model_evaluate(model, current_set, current_loader):

    #anns = current_set.annotation
    summary = sys.stdout
    tokenizer = current_set.tokenizer

    # from torchmetrics import BLEUScore
    model.eval()
    # metric = BLEUScore()
    # acc_bleu = 0
    # index_mapper = current_set.index_mapper
    with torch.no_grad():
        # with tqdm.tqdm(val_loader, unit="batch") as tepoch:
        #   for j, x in enumerate(tepoch):
        res = []
        for j, x in tqdm(enumerate(current_loader)):
            # if j==3:
            #     break
            captions, ids = inference(x, tokenizer, model, time_difference=5)
            if j==0:
                print(captions)
            for caption, img_id in zip(captions, ids):
                res.append({"image_id": img_id.item(), "caption": caption})
        result_file = f'result/{RESULT_FILE}.json'
        json.dump(res, open(result_file, 'w'))
        if not summary == sys.stdout:
            summary.close()
        # return bleu

def cal_bert_score(results_file, annotation_file):
    with open(annotation_file, 'r') as f:
        ann = json.load(f)
    with open(results_file, 'r') as g:
        res = json.load(g)
    res_list, ann_list = [], []
    i = 0
    annotations = ann['annotations']
    for each_res in res:
        res_list.append(each_res['caption'])
        image_id = each_res['image_id']
        cap_list = []
        while True:
            if i == len(annotations):
                break
            if annotations[i]['image_id'] == image_id:
                cap_list.append(annotations[i]['caption'])
                i += 1
            else:
                break
        ann_list.append(cap_list)

    bertscore = load("bertscore")
    results = bertscore.compute(predictions=res_list, references=ann_list, lang="en")
    import numpy as np
    return np.mean(results['f1'])

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val': 'coco_karpathy_val_gt.json', 'test': 'coco_karpathy_test_gt.json'}

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])
    bert_score = cal_bert_score(results_file, annotation_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # model_evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # model_evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    print(f'bert score: {bert_score:.3f}')
    return coco_eval
if __name__ == '__main__':
    MODEL_NAME = 'xxxxxxx'
    model = Diffuser_with_LN(image_size=224)
    PRETRAINED_DIR = 'pretrained_ckpt'
    RESULT_FILE = 'yyyy'
    if not os.path.exists(PRETRAINED_DIR):
        os.mkdir(PRETRAINED_DIR)
    model.visual_encoder, _ = load_checkpoint(model.visual_encoder, f'{PRETRAINED_DIR}/model_base_capfilt_large.pth')
    model.load_state_dict(torch.load(
        f"{MODEL_NAME}/acc_epoch_59/pytorch_model.bin", map_location=device))
    model = model.to(device)
    from dataload import create_dataset
    from torch.utils.data import DataLoader
    config = {'image_size': 224, 'ann_root': 'datasets/COCO/', 'image_root': 'datasets/COCO'}
    train_set, val_set, test_set = create_dataset('caption_coco', config)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=100, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=2, drop_last=False, num_workers=4)
    model_evaluate(model, val_set, val_loader)
    if not os.path.exists('result'):
        os.makedirs('result', exist_ok=True)
    coco_caption_eval('result/', f'result/{RESULT_FILE}.json', split='val')
