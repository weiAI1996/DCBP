import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor
import sys


class SegmentationDataset(Dataset):
    def __init__(self, img_paths, input_shape, num_classes, train, root):
        super(SegmentationDataset, self).__init__()
        self.img_paths = img_paths
        self.length = len(img_paths)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.root = root

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if 'win' in sys.platform:
            name = img_path.split('.')[0].split('\\')[-1]
        else:
            name = img_path.split('.')[0].split('/')[-1]

        img = Image.open(img_path)
        label = Image.open(os.path.join(os.path.join(self.root, "labels"), name + ".png"))
        line_label = Image.open(os.path.join(os.path.join(self.root, "line_labels"), name + ".png"))

        img, label,line_label = self.get_random_data(img, label,line_label ,self.input_shape, random=self.train)

        img = np.transpose(preprocess_input(np.array(img, np.float64)), [2, 0, 1])
        label = np.array(label)
        line_label = np.array(line_label)
        label[label >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return img, label,line_label, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label,line_label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        line_label = Image.fromarray(np.array(line_label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((w, h), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((w, h), Image.NEAREST)
            line_label = line_label.resize((w, h), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_line_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            new_line_label.paste(line_label, ((w - nw) // 2, (h - nh) // 2))
            return image, label,line_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((w, h), Image.BICUBIC)
        label = label.resize((w, h), Image.NEAREST)
        line_label = line_label.resize((w, h), Image.NEAREST)
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            line_label = line_label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_line_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        new_line_label.paste(line_label, (dx, dy))

        image_data = np.array(image, np.uint8)

        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            line_label = cv2.warpAffine(np.array(line_label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))


        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label,line_label


def seg_dataset_collate(batch):
    images = []
    pngs = []
    line_pngs = []
    seg_labels = []
    for img, png,line_png, labels in batch:
        images.append(img)
        pngs.append(png)
        line_pngs.append(line_png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    line_pngs = torch.from_numpy(np.array(line_pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs,line_pngs, seg_labels
