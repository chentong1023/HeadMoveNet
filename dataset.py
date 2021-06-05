import os
import torch
import math
import numpy as np
import torch.utils.data as data


class FixDataset(data.Dataset):
	def __init__(self, head, gaze, seq_len):
		self._head = np.load(head)
		self._gaze = np.load(gaze)
		self._seq = seq_len
	
	def __len__(self):
		return len(self._head) - seq_len
	
	def __getitem__(self, idx):
		inp = torch.arange(0)
		for i in range(seq_len):
			inp = np.concatenate([inp, self._gaze[i + idx]])
		tar = self._head[idx + seq_len]
		return inp, tar

class UnityDataset(data.Dataset):
	def __init__(self, img_dir, label_dir, training, seq_len=4):
		self._img_dir = img_dir
		self._label_dir = label_dir
		self._train = training
		
		self._imgs = []
		img_files = os.listdir(img_dir)
		img_npy = [f for f in img_files if f.endswith(".npy")]
		img_npy = sorted(img_npy, key = lambda x: int(x[10:-8]))
		for img in img_npy:
			data = np.load(os.path.join(img_dir, img))
			self._imgs.append(data)
		
		self._labels = []
		label_files = os.listdir(label_dir)
		label_txts = [f for f in label_files if f.endswith(".txt")]
		label_txts = sorted(label_txts, key = lambda x: int(x[3:-4]))
		for label in label_txts:
			with open(os.path.join(label_dir, label), 'r') as f:
				data = []
				lines = list(f.readlines())[:2]
				for line in lines:
					num = line.strip('()\n').split(',')
					print(num)
					data.append([float(i) for i in num])
			data = np.array(data).reshape((6))
			self._labels.append(data)
		
		self._delta = [np.zeros((2, 3))]
		for i in range(1, len(self._labels)):
			self._delta.append(self._labels[i] - self._labels[i - 1])
		
		self.seq_len = seq_len
	
	def __len__(self):
		return len(self._imgs) - self.seq_len
	
	def __getitem__(self, idx):
		img = torch.from_numpy(self._imgs[idx]).float().unsqueeze(0)
		for i in range(1, self.seq_len):
			img = torch.cat((img, torch.from_numpy(self._imgs[idx + i]).float().unsqueeze(0)))
		his = torch.from_numpy(self._labels[idx]).float().unsqueeze(0)
		for i in range(1, self.seq_len):
			his = torch.cat((his, torch.from_numpy(self._labels[idx + i]).float().unsqueeze(0)))
		tar = torch.from_numpy(self._labels[idx + self.seq_len]).float()
		return img, his, tar