import os
from glob import glob
import torch.utils.data as data
import numpy as np

# Path of the data set
path_noisy = r"..\CNCL-denoising-main\data\mayo_data_pp\train\quarter_1mm"
path_clean = r"..\CNCL-denoising-main\data\mayo_data_pp\train\full_1mm"

noisy_list = sorted(glob(os.path.join(path_noisy, "*")))
clean_list = sorted(glob(os.path.join(path_clean, "*")))

def load_npy(filepath):
    image = np.load(filepath)
    image = np.array(image).astype('float32')
    mean = np.mean(image)
    var = np.var(image)
    # return np.expand_dims(image, axis=0), mean, var
    return image, mean, var

def  get_Training_Set():
    return DatasetFromFolder(noisy_list, clean_list)

class DatasetFromFolder(data.Dataset):
    def __init__(self, noisy_list, clean_list):
        super(DatasetFromFolder, self).__init__()
        self.noisy_list = noisy_list
        self.clean_list = clean_list

    def __getitem__(self, index):
        noisy, mean_n, var_n = load_npy(self.noisy_list[index])
        clean, mean_c, var_c = load_npy(self.clean_list[index])
        noisy = (noisy - mean_n) / var_n
        clean = (clean - mean_n) / var_n
        noise = noisy - clean
        target = np.concatenate((clean, noise), axis=0)
        return {"A": target, "B": noisy, "C": mean_n, "D": var_n}
    
    def __len__(self):
        return len(self.noisy_list)