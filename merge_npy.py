import os
import numpy as np
import matplotlib.pyplot as plt

## show sample 

my_dir = "/n/home11/zimani/datasets/protons64/test/"

data_file = my_dir + "protons64_100.npy"
mom_file = data = my_dir + "protons64_mom_100.npy"

data = np.load(data_file)
mom = np.load(mom_file)

idx = 4

plt.imshow(data[idx], cmap='gray')
plt.title(str(mom[idx]), fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("zzz.png")


exit() 

# Define the directory containing the .npy files
directory = 'protons64_sample/samples/'

# Initialize an empty list to store the arrays
combined_data = []


# Iterate through all files in the directory
for filename in sorted(os.listdir(directory)):
	if filename.endswith('.npy') and 'batch_' in filename:
		# Load the .npy file
		file_path = os.path.join(directory, filename)
		try: 
			data = np.load(file_path)
			
			# Append the data to the list
			combined_data.append(data)
		except: 
			continue 
		

# Combine all arrays into one
combined_data = np.concatenate(combined_data, axis=0)

# Save the combined array into a new .npy file
np.save(directory[:-1]+".npy", combined_data)

print("Saved:", directory[:-1]+".npy, shape =", combined_data.shape)
