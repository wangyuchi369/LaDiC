import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from dataload.utils import pre_caption
from configs.config import *

class para_train(Dataset):
    def __init__(self, transform, tokenizer, image_root, ann_root, max_words=200, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''


        self.annotation = json.load(open(os.path.join(ann_root, 'paragraphs_v1.json'), 'r'))
        self.img2paragraph = {}
        for each_img in self.annotation:
            image_id = each_img['image_id']
            each_paragraph = each_img['paragraph']
            self.img2paragraph[image_id] = each_paragraph
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.train_list = json.load(open(os.path.join(ann_root, 'train_split.json'), 'r'))


    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):

        img_id = self.train_list[index]

        image_path = os.path.join(self.image_root, f'{img_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # caption = pre_caption(ann['caption'], self.max_words)
        caption = pre_caption(self.img2paragraph[img_id])
        tokens = self.tokenizer(text=caption, return_tensors="pt", padding='max_length',
                                truncation=True, max_length=MAX_LENGTH)
        # return image, caption, self.img_ids[ann['image_id']]
        return {'image': image,
                'text': caption,
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze()
        }


class para_eval(Dataset):
    def __init__(self, transform, tokenizer, image_root, ann_root, split):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
        #         'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        # filenames = {'val': 'coco_karpathy_val.json', 'test': 'coco_karpathy_test.json'}
        #
        # download_url(urls[split], ann_root)

        self.annotation = json.load(open(os.path.join(ann_root, 'paragraphs_v1.json'), 'r'))
        self.imgid_list = json.load(open(os.path.join(ann_root, f'{split}_split.json'), 'r'))
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.img2paragraph = {}
        for each_img in self.annotation:
            image_id = each_img['image_id']
            each_paragraph = each_img['paragraph']
            self.img2paragraph[image_id] = each_paragraph
    def __len__(self):
        return len(self.imgid_list)

    def __getitem__(self, index):
        img_id = self.imgid_list[index]
        image_path = os.path.join(self.image_root, f'{img_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        gt_paragraph = self.img2paragraph[img_id]
        return {'image': image,
                'img_id': int(img_id),
                'index': index,
                'gt_paragraph': gt_paragraph
                }