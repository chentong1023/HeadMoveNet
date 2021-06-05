import torch
import torch.nn as nn

class HeadPredictModel(nn.Module):
	def __init__(self, size_r, size_c):
		super(HeadPredictModel, self).__init__()
		self._size_r = size_r
		self._size_c = size_c
		self._inp_dim = size_r * size_c + 6
		self.lstm = nn.LSTM(input_size=self._inp_dim, hidden_size=512)
		self.linear = nn.Linear(in_features=512 * 4, out_features=6)
	
	def forward(self, img, his):
		x1 = img.reshape((-1, 4, self._size_r * self._size_c))
		x2 = torch.cat((x1, his), dim=-1).permute(1, 0, 2)
		x3 = self.lstm(x2)[0]
		x4 = torch.flatten(x3.permute(1, 0, 2), start_dim=1)
		x5 = self.linear(x4)
		return x5
		
		