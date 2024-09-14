import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data1, is_train):
        self.data1 = data1
        self.is_train = is_train
        
        total_samples = len(data1)
        self.train_samples = int(0.7 * total_samples)
        self.test_samples = total_samples - self.train_samples

    def __len__(self):
        if self.is_train:
            return self.train_samples
        else:
            return self.test_samples

    def __getitem__(self, idx):
        if self.is_train:
            return torch.tensor(self.data1[idx], dtype=torch.float32)
        else:
            offset = self.train_samples
            return torch.tensor(self.data1[offset + idx], dtype=torch.float32)


def get_model_data(data_path,data_size,image_channel,image_height,image_weight,batch_size):
    real_data = np.memmap(data_path, dtype=np.uint8, mode='r', shape=(data_size,) + (image_channel, image_height, image_weight), offset=0)
    train_dataset = MyDataset(real_data, is_train=True)
    test_dataset = MyDataset(real_data, is_train=False)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=True)
    return train_dataloader,test_dataloader