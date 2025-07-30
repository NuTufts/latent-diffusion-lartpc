import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch 


class protons(Dataset):
	def __init__(self,
				 data_root,
				 num_batches,
				 events_per_batch,
				 only_one=False,
				 v1=False):
		self.data_root = data_root
		self.num_batches = num_batches
		self.events_per_batch = events_per_batch
		self._length = self.num_batches * self.events_per_batch
		self.only_one = only_one
		if self.only_one:
			self._length = 1000
		self.v1 = v1

	def __len__(self):
		return self._length

	def __getitem__(self, i):

		## Use same event every time (testing purposes) 
		if self.only_one:
			i = 0

		## Load appropriate batch file 
		batch_idx = i // self.events_per_batch
		if self.v1:
			batch_file = os.path.join(self.data_root, f'protons64_{batch_idx}.npy')
		else: 
			batch_file = os.path.join(self.data_root, f'batch_{batch_idx}.npy')
		batch_data = np.load(batch_file)
		
		## Load corresponding momentum
		if self.v1:
			mom_file = os.path.join(self.data_root, f'protons64_mom_{batch_idx}.npy')
		else:
			mom_file = os.path.join(self.data_root, f'batch_mom_{batch_idx}.npy')
		mom_data = np.load(mom_file)

		## Get specific event 
		event_idx = i % self.events_per_batch
		event = batch_data[event_idx]

		## Normalize momentum 
		mom = mom_data[event_idx] #/ 500.0 ## [-1000, 1000] -> [-2, 2]

		## Add single channel 
		event = np.expand_dims(event, -1)

		## Save as dictionary 
		example = {} 
		example["image"] = event.astype(np.float32)
		example["momentum"] = mom.astype(np.float32)

		return example 



class gaussianBlobTrain(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/gaussian_blobs_16x16/train", 
						 num_batches=64, events_per_batch=128, **kwargs)

class gaussianBlobValidation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/gaussian_blobs_16x16/val",
						 num_batches=64, events_per_batch=128, **kwargs)

class gaussianBlobxTrain(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/gaussian_blobs_x/train", 
						 num_batches=32, events_per_batch=128, **kwargs)

class gaussianBlobxValidation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/gaussian_blobs_x/val",
						 num_batches=32, events_per_batch=128, **kwargs)


if __name__ == "__main__": 
	train = gaussianBlobxValidation() 
	example = train[0]
	print(example["image"].shape)
	print(example["momentum"].shape)
