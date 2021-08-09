"""
UOW, Wed Feb 24 23:37:42 2021
Dependencies: torch>=1.1, torchvision>=0.3.0
"""
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms as T


# Read a general dataset from text files
class FaceDataset(Dataset):
    def __init__(self, dataset_dir,
                 data_list_file,
                 age_group=1,
                 num_classes=117,
                 input_shape=(64, 64)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.age_group = age_group

        with open(os.path.join(dataset_dir, data_list_file), 'r') as fd:
            imgs = fd.readlines()
        imgs = [dataset_dir + img[:-1] for img in imgs]  # img[:-1] to remove the last character '\n'
        self.imgs = np.random.permutation(imgs)

        self.transforms = T.Compose([
            T.Resize(self.input_shape, interpolation=Image.BILINEAR),
            # T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        img = Image.open(img_path).convert('RGB')  # avoid grayscale images in the FGNET dataset
        img = self.transforms(img)
        age = np.int32(splits[1])
        age = int(age / self.age_group)

        # For regression methods
        encode = [1] * age + [0] * (self.num_classes - age)
        encode = torch.tensor(encode, dtype=torch.float32)

        return img, age, encode

    def __len__(self):
        return len(self.imgs)


# Read the APPA dataset (Thanh's code)
class APPADataset(Dataset):
    def __init__(self, dataset_dir, data_type, img_size=128):
        assert (data_type in ('train', 'valid', 'test'))
        csv_path = dataset_dir + 'gt_avg_' + data_type + '.csv'
        img_dir = dataset_dir + data_type
        self.resize_raw = T.Resize((img_size, img_size), interpolation=Image.BILINEAR)
        self.to_tensor = T.ToTensor()

        self.x = []
        self.y = []
        df = pd.read_csv(str(csv_path))
        ignore_path = dataset_dir + 'ignore_list.csv'
        ignore_img_names = list(pd.read_csv(str(ignore_path))['img_name'].values)

        for _, row in df.iterrows():
            img_name = row['file_name']

            if img_name in ignore_img_names:
                continue

            img_path = img_dir + '/' + img_name + '_face.jpg'
            self.x.append(img_path)
            self.y.append(row['real_age'])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]
        img = Image.open(img_path)
        img = self.resize_raw(img)
        img = self.to_tensor(img)
        return img, np.clip(round(age), 0, 100)


# Soan's code (failed): try to deal with the problem of Unicode filename
# class FaceDataset_2(Dataset):
#
#     def __init__(self, dataset_dir,
#                  data_list_file,
#                  age_group=1,
#                  num_classes=117,
#                  input_shape=(64, 64)):
#         self.age_group = age_group
#         self.num_classes = num_classes
#         self.input_shape = input_shape
#
#         imgs = np.loadtxt(data_list_file, dtype=str, delimiter=' ')
#         self.imgs = []
#         for item in imgs:
#             img_path = '%s/%s' % (dataset_dir, item[0])
#             self.imgs.append([img_path, item[1]])
#         self.imgs = np.random.permutation(self.imgs)
#
#         self.transforms = T.Compose([
#             T.ToPILImage(),
#             T.Resize(self.input_shape),
#             T.RandomHorizontalFlip(p=0.5),
#             T.ToTensor()
#         ])
#
#     def __getitem__(self, index):
#         sample = self.imgs[index]
#         img_path = sample[0]
#         if img_path.isascii():
#             data = cv2.imread(img_path)
#         else:
#             # print(img_path)
#             stream = open(img_path, "rb")
#             numpyarray = np.asarray(bytearray(stream.read()), dtype=np.uint8)
#             data = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
#         data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#         data = self.transforms(data)
#         age = np.int32(sample[1])
#         age = int(age / self.age_group)
#
#         # For regression methods
#         encode = [1] * age + [0] * (self.num_classes - age)
#         encode = torch.tensor(encode, dtype=torch.float32)
#
#         return data.float(), age, encode
#
#     def __len__(self):
#         return len(self.imgs)
