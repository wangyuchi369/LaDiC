from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
import clip
import torch
from .evaluate_clip import get_clip_score, get_refonlyclipscore
from zipfile import ZipFile
from urllib.request import urlretrieve
import pprint

# The cache dir is where we will store all of the temporary
# data for CLIP
CLIPDIR = os.path.dirname(__file__)


def print_progress(transferred_blocks, block_size, total_size):
    current_mb = transferred_blocks * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    percent = current_mb / total_mb
    progress_str = "Progress: {:5.1f}M / {:5.1f}M ({:6.1%})"
    print(progress_str.format(current_mb, total_mb, percent), end='\r')


class ClipScore:
    """
    Main Class to compute CLIPScore
    pip install git+https://github.com/openai/CLIP.git
    """

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('clipscore is using {}'.format(device))
        self.device = device
        if device == 'cpu':
            print('CLIP runs in full float32 on CPU. Results in CLIPScore paper were computed on GPU, which uses float16. '
                  'If you\'re reporting results on CPU, please note this when you report, though differences should be small. '
                  'To run in the GPU setting, please check out https://github.com/jmhessel/clipscore')
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        self.model = model
        cwd = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(cwd, CLIPDIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.mscoco_feats_path = os.path.join(CLIPDIR, 'mscoco_vitb_features/mscoco_image_features~ViT-B32.npy')
        self.mscoco_id2row_path = os.path.join(CLIPDIR, 'mscoco_vitb_features/mscoco_image_features~ViT-B32~im2row.json')
        if not os.path.exists(self.mscoco_feats_path):
            print('Downloading MSCOCO image features for CLIPScore ...')
            url = 'https://storage.googleapis.com/ai2-jack-public/clipscore/mscoco_vitb_features.zip'
            zip_file, headers = urlretrieve(url, reporthook=print_progress)
            for filef in ['mscoco_vitb_features/mscoco_image_features~ViT-B32~im2row.json', 'mscoco_vitb_features/mscoco_image_features~ViT-B32.npy']:
                ZipFile(zip_file).extract(filef, CLIPDIR)
        assert os.path.exists(self.mscoco_feats_path), 'download error, {} doesnt exist!'.format(self.mscoco_feats_path)
        assert os.path.exists(self.mscoco_id2row_path), 'download error, {} doesnt exist!'.format(self.mscoco_id2row_path)
        self.mscoco_feats = np.load(self.mscoco_feats_path)
        with open(self.mscoco_id2row_path) as f:
            self.mscoco_feats_id2row = json.load(f)

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })

        is_valid_clipscore = []
        image_feats = []
        for d in input_data:
            try:
                image_feats.append(self.mscoco_feats[self.mscoco_feats_id2row[str(d['image_id']).zfill(12)]])
            except:
                raise NotImplementedError(
                    'image ID "{}" not found in MSCOCO. If you\'re trying to run CLIPScore on a dataset that is not MSCOCO, '
                    'please consider using the command line utility here: https://github.com/jmhessel/clipscore'.format(
                        d['image_id']))

        image_feats = np.vstack(image_feats)
        is_valid_clipscore = np.array(is_valid_clipscore)

        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = get_clip_score(
            self.model, image_feats, [d['test'] for d in input_data], self.device)

        # get text-text clipscore
        _, per_instance_text_text = get_refonlyclipscore(
            self.model, [d['refs'] for d in input_data], candidate_feats, self.device)

        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
        # scores is a list of dictionaries
        scores = [{'CLIPScore': clipscore, 'RefCLIPScore': refclipscore}
                  for clipscore, refclipscore in zip(per_instance_image_text, refclipscores)]

        return [np.mean(per_instance_image_text), np.mean(refclipscores)], scores

    def method(self):
        return "CLIPScore"
