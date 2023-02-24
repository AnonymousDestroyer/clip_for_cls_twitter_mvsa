# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 2:36
# @Author  : Yujin Wang

import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import os
import clip

test_text = 'a vicious wolf'
test_image_path = './test_image/img.png'

# test_text = 'a baby said come on'
# test_image_path = './test_image/img_1.png'

# test_text = 'masked man'
# test_image_path = './test_image/img_2.png'

# pos pos
# test_text = '#escort We have a young and energetic team and we pride ourselves on offering the highes #hoer'
# test_image_path = './test_image/4.jpg'

# neg # neg
# test_text = '#depressed #depression #bullied #anxiety #overdosed #addict #drugs #pills #cuts #cutting #… '
# test_image_path = './test_image/19.jpg'

# load trained model
model_pth_path = 'model_pth/best.pth'
model = torch.load(model_pth_path)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
_, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
print('model loaded.')
label_class_names = ['negative', 'neutral', 'positive']


def map_idx_to_label(idx):
    """
    :param idx:
    :return:
    """
    return label_class_names[idx]


def inference_mm_rules(text_res, image_res):
    """
    inference multimodal data result using rules
    :param text_res:
    :param image_res:
    :return:
    """
    if text_res == 1 and image_res == 1:
        return 'neutral'
    elif text_res + image_res >= 3:
        return 'positive'
    else:
        return 'negative'


# images = torch.stack([preprocess(Image.fromarray(img)) for img in test_image_path], dim=0).to(device)
image = preprocess(Image.open(test_image_path)).unsqueeze(0).to(device)
text = clip.tokenize(texts=test_text).to(device)
label_names = torch.cat([clip.tokenize(f"{c}") for c in label_class_names]).to(device)

image_features = model.encode_image(image)
text_features = model.encode_text(text)
label_features = model.encode_text(label_names)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
label_features /= label_features.norm(dim=-1, keepdim=True)

probs_text = (100.0 * text_features @ label_features.T).softmax(dim=-1)
probs_image = (100.0 * image_features @ label_features.T).softmax(dim=-1)

# print('==> train ground truth:')
# print("labels image:{}".format(labels_image))
# print("labels text:{}".format(labels_text))

pred_text = torch.argmax(probs_text, dim=1).item()
pred_image = torch.argmax(probs_image, dim=1).item()
print('the text is {}, the image is {}, the text+image is {}'.format(map_idx_to_label(pred_text), map_idx_to_label(pred_image), inference_mm_rules(pred_text, pred_image)))
