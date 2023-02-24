# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 19:56
# @Author  : Yujin Wang

import os
import numpy
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import clip
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Subset
from torch.cuda.amp import autocast, GradScaler
from transformers import CLIPProcessor, CLIPModel
import json


class MVSADataSet(data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_dir = os.path.join(root_dir, 'img')
        self.text_dir = os.path.join(root_dir, 'txt')
        self.label_dir = os.path.join(root_dir, 'label_process.json')

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.txt')])
        self.label = json.load(open(self.label_dir))

        assert len(self.image_files) == len(self.text_files), "The number of images and text files must be the same!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        text_file = os.path.join(self.text_dir, self.text_files[idx])

        image = Image.open(image_file).convert('RGB')
        with open(text_file, 'r', errors='ignore') as f:
            text = str(f.read())
        text_label = self.label[self.image_files[idx].split('.')[0]]['text']
        image_label = self.label[self.image_files[idx].split('.')[0]]['image']

        if self.transform is not None:
            image = self.transform(image)

        return image, text, image_label, text_label

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()



# load clip
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clip_model_path = '../clip_models/ViT-B-32.pt'
model, processor = clip.load(clip_model_path, device=device, jit=False)  # Must set jit=False for training
print('==> model loaded.')


dataset = MVSADataSet(root_dir='data/MVSA_spl', transform=processor)
train_dataset_size = int(0.95 * (len(dataset)))
test_dataset_size = len(dataset) - train_dataset_size
train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_dataset_size, test_dataset_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)


# optimizer = optim.AdamW([{'params': clip.parameters()}, {'params': fc.parameters()}], lr=1e-4)
optimizer = optim.Adam(model.parameters(), lr=1e-6,weight_decay=1e-5)


# parameters
num_epochs = 50
loss_img = nn.CrossEntropyLoss().to(device)
loss_txt = nn.CrossEntropyLoss().to(device)
loss_lab = nn.CrossEntropyLoss().to(device)

label_class_names = ['negative', 'neutral', 'positive']
best_txt_val_acc = 0
best_img_val_acc = 0
best_txt_img_val_acc = 0

def map_txt_to_idx(txt):
    """
    :param txt:
    :return:
    """
    if txt == 'negative':
        return 0
    elif txt == 'neutral':
        return 1
    elif txt == 'positive':
        return 2


model_pth_path = 'model_pth/'
model_best_text_path = os.path.join(model_pth_path, 'best_text.pth')
model_best_image_path = os.path.join(model_pth_path, 'best_image.pth')
model_best_path = os.path.join(model_pth_path, 'best.pth')

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    print('Epoch {}:'.format(epoch))
    for i, (images, texts, labels_image, labels_text) in enumerate(train_loader):
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        labels_image = clip.tokenize(labels_image).to(device)
        labels_text = clip.tokenize(labels_text).to(device)
        logits_per_image, logits_per_text, logits_per_text_label, logits_per_image_label, logits_per_label_text, logits_per_label_image = model(
            images, texts, labels_image, labels_text)
        if device == "cpu":
            ground_truth = torch.arange(len(labels_image)).long().to(device)
        else:
            ground_truth = torch.arange(len(labels_image), dtype=torch.long, device=device)
        loss_txt_lab = (loss_lab(logits_per_text_label, ground_truth) + loss_txt(logits_per_label_text, ground_truth)) / 2
        loss_img_lab = (loss_lab(logits_per_image_label, ground_truth) + loss_img(logits_per_label_image, ground_truth)) / 2
        loss = (loss_txt_lab + loss_img_lab) / 2
        optimizer.zero_grad()
        loss.backward()
        convert_models_to_fp32(model)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print('Epoch {} loss: {:.4f}'.format(epoch + 1, epoch_loss))

    # test data
    model.eval()
    num_total = 0
    num_correct_text = 0
    num_correct_image = 0
    for i, (images, texts, labels_image, labels_text) in enumerate(test_loader):
        if i > 10:
            break
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        labels_image = list(map(map_txt_to_idx, labels_image))
        labels_text = list(map(map_txt_to_idx, labels_text))

        with torch.no_grad():
            label_names = torch.cat([clip.tokenize(f"{c}") for c in label_class_names]).to(device)  # 生成文字描述
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            label_features = model.encode_text(label_names)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        label_features /= label_features.norm(dim=-1, keepdim=True)

        probs_text = (100.0 * text_features @ label_features.T).softmax(dim=-1)
        probs_image = (100.0 * image_features @ label_features.T).softmax(dim=-1)

        # print('==> train ground truth:')
        # print("labels image:{}".format(labels_image))
        # print("labels text:{}".format(labels_text))

        pred_text = torch.argmax(probs_text, dim=1)
        pred_image = torch.argmax(probs_image, dim=1)
        # print('==> train pred label:')
        # print('pred text:{}'.format(pred_text))
        # print('pred image:{}'.format(pred_image))


        num_txt = sum(numpy.array(labels_text) == pred_text.detach().cpu().numpy())
        num_img = sum(numpy.array(labels_image) == pred_image.detach().cpu().numpy())
        num_correct_image += num_img
        num_correct_text += num_txt
        num_total += len(labels_image)

    val_acc_txt = num_correct_text / num_total
    val_acc_img = num_correct_image / num_total

    if (val_acc_txt + val_acc_img) > best_txt_img_val_acc:
        best_txt_img_val_acc = (val_acc_txt + val_acc_img)
        torch.save(model, model_best_path)
        print('saved model best')

    if val_acc_txt > best_txt_val_acc:
        best_txt_val_acc = val_acc_txt
        torch.save(model, model_best_text_path)
        print('saved model text best')

    if val_acc_img > best_img_val_acc:
        best_img_val_acc = val_acc_img
        torch.save(model, model_best_image_path)
        print('saved model image best')

    print('val txt:{}, val img:{}'.format(val_acc_txt, val_acc_img))
    print('best val txt；{}, best val img:{}'.format(best_txt_val_acc, best_img_val_acc))


# clip.eval()
# val_loss = 0.0
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, texts in val_loader:
#         images = images.to(device)
#         texts = [processor(text, return_tensors='pt', padding=True) for text in texts]
#         texts = {key: value.to(device) for key, value in texts[0].items()}
#
#         features = clip(image=images, text=texts)['last_hidden_state']
#         features = features[:, 0, :]
#         outputs = fc(features)
#         loss = criterion(outputs, labels)
#
#         val_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# val_loss /= len(val_loader)
# accuracy = 100 * correct / total
# print('Validation loss: {:.4f}'.format(val_loss))
# print('Accuracy: {:.2f}%'.format(accuracy))
