import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from dataload.utils import pre_caption
from configs.config import *

class coco_karpathy_train(Dataset):
    def __init__(self, transform, tokenizer, image_root, ann_root, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
        filename = 'coco_karpathy_train.json'

        download_url(url, ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.tokenizer = tokenizer

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        tokens = self.tokenizer(text=caption, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=MAX_LENGTH)
        # return image, caption, self.img_ids[ann['image_id']]
        return {'image': image,
                'text': caption,
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze()
        }

class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, tokenizer, image_root, ann_root, split):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}

        download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        return {'image': image,
                'img_id': int(img_id),
                'index':index
                }


# class coco_karpathy_retrieval_eval(Dataset):
#     def __init__(self, transform, image_root, ann_root, split, max_words=30):
#         '''
#         image_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         split (string): val or test
#         '''
#         urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
#                 'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
#         filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}
#
#         download_url(urls[split], ann_root)
#
#         self.annotation = json.load(open(os.path.join(ann_root, filenames[split]), 'r'))
#         self.transform = transform
#         self.image_root = image_root
#
#         self.text = []
#         self.image = []
#         self.txt2img = {}
#         self.img2txt = {}
#
#         txt_id = 0
#         for img_id, ann in enumerate(self.annotation):
#             self.image.append(ann['image'])
#             self.img2txt[img_id] = []
#             for i, caption in enumerate(ann['caption']):
#                 self.text.append(pre_caption(caption, max_words))
#                 self.img2txt[img_id].append(txt_id)
#                 self.txt2img[txt_id] = img_id
#                 txt_id += 1
#
#     def __len__(self):
#         return len(self.annotation)
#
#     def __getitem__(self, index):
#
#         image_path = os.path.join(self.image_root, self.annotation[index]['image'])
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)
#
#         return image, index
