import torch
import os, warnings
from lifelines.utils import concordance_index
import numpy as np

class Regularization(object):
	def __init__(self, order, weight_decay):
		super(Regularization, self).__init__()
		self.order = order
		self.weight_decay = weight_decay

	def __call__(self, model):
		reg_loss = 0.
		for name, w in model.named_parameters():
			if 'weight' in name:
				reg_loss = reg_loss + torch.norm(w, p = self.order)
		reg_loss = self.weight_decay * reg_loss
		return reg_loss

class NegativeLogLikelihood(torch.nn.Module):
	# https://github.com/czifan/DeepSurv.pytorch/blob/f572a8a7d3ce5ad10609bd273ce3accaf7ea4b66/networks.py#L76
	def __init__(self):
		super(NegativeLogLikelihood, self).__init__()
		self.reg = Regularization(order = 2, weight_decay = 0)# 1e-5

	def forward(self, risk_pred, y, e, model):
		#print(y.shape)
		mask = torch.ones(y.shape[0], y.shape[0])
		mask[(y.T - y) > 0] = 0
		if torch.cuda.is_available():
			mask = mask.cuda()
		log_loss = torch.exp(risk_pred) * mask
		#log_loss = torch.sum(log_loss, dim = 0) / torch.sum(mask, dim = 0)
		log_loss = torch.sum(log_loss, dim = 0)
		log_loss = torch.log(log_loss).reshape(-1,1)
		neg_log_loss = -torch.sum((risk_pred - log_loss) * e)/torch.sum(e)

		l2_loss = self.reg(model)
		return neg_log_loss + l2_loss

def c_index(risk_pred, y, e):
	'''
		- y = survival time
		- e = event (status)
	'''
	if not isinstance(risk_pred, np.ndarray):
		risk_pred = risk_pred.detach().cpu().numpy()
	if not isinstance(y, np.ndarray):
		y = y.detach().cpu().numpy()
	if not isinstance(e, np.ndarray):
		e = e.detach().cpu().numpy()
	return concordance_index(y, risk_pred, e)
