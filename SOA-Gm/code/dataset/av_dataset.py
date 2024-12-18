import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms
import librosa

class AVDataset_CD(Dataset):
    def __init__(self, mode='train'):
        classes = []
        self.data = []
        data2class = {}

        self.mode = mode
        self.visual_path = ''
        self.audio_path = ''
        self.stat_path = ""
        self.train_txt = ''
        self.test_txt = ''
        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                aud_path = os.path.join(self.audio_path,item[0]+'.pt')
                # print(os.path.exists(
                #         self.visual_path + '/' + item[0]))
                if item[1] in classes and os.path.exists(aud_path) and os.path.exists(
                        self.visual_path + '/' + item[0]):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]

        print('data load over')
        print(len(self.data))

        self.classes = sorted(classes)

        self.data2class = data2class
        self._init_atransform()
        print('# of files = %d ' % len(self.data))
        print('# of classes = %d' % len(self.classes))

        # Audio
        self.class_num = len(self.classes)

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        # Audio
        fbank = torch.load(os.path.join(self.audio_path , datum + '.pt')).unsqueeze(0)





        # Visual
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        folder_path = os.path.join(self.visual_path,datum)
        file_num = len(os.listdir(folder_path))
        file_list = os.listdir(folder_path)
        pick_num = 7
        # pick_num = 7
        seg = int(file_num / pick_num)

        if file_num < pick_num:
            selected_files = random.choices(file_list, k=pick_num)
        else:
            selected_files = random.sample(file_list, pick_num)

        image_arr = []
        for file in selected_files:
            path = os.path.join(folder_path, file)
            image_arr.append(transf(Image.open(path).convert('RGB')).unsqueeze(0))
        images = torch.cat(image_arr)
        # print(images.size())

        return fbank, images, self.classes.index(self.data2class[datum])