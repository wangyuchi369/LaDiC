from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import random
from configs.config import *
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# url2 = "http://images.cocodataset.org/val2017/000000049269.jpg"
# image2 = Image.open(requests.get(url2, stream=True).raw)
# images = [image, image2]

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(accelerator.device)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def get_detr_objects(inputs):
    # expected a list of Image
    #inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(pixel_values=inputs.contiguous())

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([[256,256]] * inputs.shape[0])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)
    objects_list = []
    input_ids = []
    attention_mask = []

    for i in range(inputs.shape[0]):
        labels = results[i]['labels']
        objects = set()
        for j in range(labels.shape[-1]):
            objects.add(model.config.id2label[labels[j].item()])
            if len(objects) >= 5:
                break
        objects = list(objects)
        random.shuffle(objects)
        objects_list.append(objects)
        objects = " ".join(objects)
        object_token = tokenizer(text=objects, return_tensors="pt", padding='max_length',
                                      truncation=True, max_length=MAX_OBJECT_LENGTH)
        input_ids.append(object_token['input_ids'])
        attention_mask.append(object_token['attention_mask'])
    return objects_list, torch.cat(input_ids), torch.cat(attention_mask)
    #print(model.config.id2label[results[0]["labels"]])

