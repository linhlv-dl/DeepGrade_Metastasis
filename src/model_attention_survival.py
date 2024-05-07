import torch
import torch.nn as nn
from MIL_attention import *
import torch.nn.functional as F

class NN_Model2a(nn.Module):
	def __init__(self, 
					in_features = 2048,
					fc_1= 200, 
					fc_2 = 100, 
					fc_output = 1):
		print('Attention model 2a')
		super(NN_Model2a, self).__init__()
		self.in_fc = in_features
		self.fc1 = fc_1
		self.fc2 = fc_2
		self.output = fc_output

		self.conv_extractor = nn.Sequential(nn.Conv1d(self.in_fc, 512, 1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.fc_extractor = nn.Sequential(nn.Linear(512, self.fc1),
											nn.PReLU(),
											nn.Dropout(p = 0.5))

		self.attention = Attention(self.fc1, self.fc2, self.output) # GateAttention

		self.classifier = nn.Sequential(nn.Dropout(p = 0.5),
										nn.Linear(self.fc1 * self.output, self.output),
										)

	def forward(self, x): 
		x = x.unsqueeze(3)
		x = x.squeeze(0)
		x = self.conv_extractor(x)
		x = x.view(x.size(0), -1)

		x = self.fc_extractor(x)
		x_attention = self.attention(x)

		x_2 = torch.mm(x_attention, x)
		y_prob = self.classifier(x_2)

		return y_prob, x_attention
