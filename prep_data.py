from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision import transforms

IMAGE_PATH = '../data/img_align_celeba/img_align_celeba/'

class AttributeSet(Dataset):
	def __init__(self, train=True, transform=None):
		df = pd.read_csv('../data/list_attr_celeba.csv')
		df.replace(-1, 0, inplace=True)
		df = df.values
		np.random.shuffle(df)
		if train:
			self.df_size = 10048
		else:
			self.df_size = 5008
		names = df[:self.df_size, 0].tolist()
		self.attr = df[:self.df_size, 1:]
		self.transform = transform
		self.samples = []

		for i, name in enumerate(names):
			img = cv2.imread(IMAGE_PATH + name, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (150, 150))
			if self.transform:
				img = self.transform(img)
			img = img / 255.0
			self.samples.append((torch.from_numpy(img.astype(np.float64)), torch.from_numpy(self.attr[i].astype(np.int64))))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]
