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
				 only_one=False):
		self.data_root = data_root
		self.num_batches = num_batches
		self.events_per_batch = events_per_batch
		self._length = self.num_batches * self.events_per_batch
		self.only_one = only_one
		if self.only_one:
			self._length = 1000

	def __len__(self):
		return self._length

	def __getitem__(self, i):

		## Use same event every time (testing purposes) 
		if self.only_one:
			i = 0

		## Load appropriate batch file 
		batch_idx = i // self.events_per_batch
		batch_file = os.path.join(self.data_root, f'protons64_{batch_idx}.npy')
		batch_data = np.load(batch_file)
		
		## Load corresponding momentum
		mom_file = os.path.join(self.data_root, f'protons64_mom_{batch_idx}.npy')
		mom_data = np.load(mom_file)

		## Get specific event 
		event_idx = i % self.events_per_batch
		event = batch_data[event_idx]

		## Normalize momentum 
		mom = mom_data[event_idx] / 500.0 ## [-1000, 1000] -> [-2, 2]

		## Add single channel 
		event = np.expand_dims(event, -1)

		## Save as dictionary 
		example = {} 
		example["image"] = event.astype(np.float32)
		example["momentum"] = mom.astype(np.float32)

		return example 


class protons64Train(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_sqrt/train", 
						 num_batches=1152, events_per_batch=128, **kwargs)


class protons64Validation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_sqrt/val",
						 num_batches=152, events_per_batch=128, **kwargs)


class protons64xTrain(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_sqrt/train",  
						 num_batches=155, events_per_batch=128, **kwargs)


class protons64xValidation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_sqrt/val",
						num_batches=21, events_per_batch=128, **kwargs)


# class protons512Train(protons):
# 	def __init__(self, **kwargs):
# 		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons512/train", 
# 						 num_batches=0, events_per_batch=64, **kwargs)
		
# class protons512Validation(protons):
# 	def __init__(self, **kwargs):
# 		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons512/val",
# 						 num_batches=893, events_per_batch=64, **kwargs)