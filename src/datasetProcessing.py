import numpy as np
import torch

class datasetProcessing(torch.utils.data.Dataset):
	def __init__(self, path, n_prot, n_aa, n_features):
		dataset = torch.from_numpy(np.load(path)).float()
		dataset = dataset.reshape(n_prot,n_aa,n_features)
		features_universe = [i for i in range(0,n_features)]
		index_features = features_universe[0:21] + features_universe[35:56]
		index_labels = features_universe[22:30]
		self.n_samples = n_prot; self.n_aa = n_aa; self.n_features = n_features
		self.Training_Test_preparation(dataset,index_features,index_labels)


	def Training_Test_preparation(self, ts,index_features,index_labels):
		dataset_observations = ts[:,:,index_features]
		labels = ts[:,:,index_labels]
		training_DS = dataset_observations[0:5434,:,:]; training_labels = labels[0:5434,:,:]
		test_DS = dataset_observations[5435:5690,:,:]; test_labels = labels[5435:5690,:,:]
		validation_DS = dataset_observations[5690:5926,:,:]; validation_labels = labels[5690:5926,:,:]

		'''
			convert initial tensor dimension (N,L,C) into the ones that are
			require in convolutional class on pytorch (N,C,L), where:
				N = batch size (number of samples)
				C = channel size (number of features)
				L = sample size (number of aminoacids per protein in our case)
		'''
		
		self.tr = training_DS.permute([0,2,1]); self.tt = test_DS.permute([0,2,1])
		self.vc = validation_DS.permute([0,2,1]); self.trl = training_labels.permute([0,2,1])
		self.ttl = test_labels.permute([0,2,1]); self.vcl = validation_labels.permute([0,2,1])
        

	def __len__(self):
		return len(self.tr)  # required

	def __getitem__(self, idx):
		return self.tr[idx], self.trl[idx]
