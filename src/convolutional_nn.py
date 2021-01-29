import math
import torch
import pdb

class convolutional_nn(torch.nn.Sequential):
	#Convolutional Neural Network as an MLP
	def __init__(self):
		super().__init__()

		self.add_module('conv1', torch.nn.Conv1d(in_channels = 42, out_channels=64, kernel_size=3, padding=1 ) )
		self.add_module('ReLU1', torch.nn.ReLU() )
		self.add_module('conv2', torch.nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=3, padding=1 ) )
		self.add_module('ReLU2', torch.nn.ReLU() )
		self.add_module('conv3', torch.nn.Conv1d(in_channels = 64, out_channels=64, kernel_size=3, padding=1) )
		self.add_module('ReLU3', torch.nn.ReLU() )
		self.add_module('conv4', torch.nn.Conv1d(in_channels = 64, out_channels=8, kernel_size=3, padding=1) )
		self.add_module('Softmax', torch.nn.Softmax(dim=1) )
		
		'''
		self.add_module('Flatten', torch.nn.Flatten() )
		self.add_module('linear1', torch.nn.Linear(5600, 500) )
		self.add_module('ReLU1', torch.nn.ReLU() )
		self.add_module('linear2', torch.nn.Linear(500, 150) )
		self.add_module('ReLU2', torch.nn.ReLU() )
		self.add_module('linear3', torch.nn.Linear(150, 8) )
		self.add_module('Softmax', torch.nn.Softmax(dim=1) )
		'''