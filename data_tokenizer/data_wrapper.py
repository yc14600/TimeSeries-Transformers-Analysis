from torch.utils.data import Dataset
import torch


class Dataset_Custom_Wrapper(Dataset):
    def __init__(self, args, dataset):
        # size [seq_len, label_len, pred_len]
        self.args = args
        self.dataset = dataset
        self.scale = dataset.scale
        self.data_x = self.dataset.data_x
        self.data_y = self.dataset.data_y
        if self.args.data != 'PEMS':
            self.data_stamp = self.dataset.data_stamp
        self.n_features = self.data_x.shape[1]
        

    def __getitem__(self, index):
        x = self.data_x[index:index+self.args.seq_len]
        target = self.data_y[index+self.args.seq_len:index+self.args.seq_len+self.args.pred_len]
        if self.args.data != 'PEMS':
            stamp = self.data_stamp[index]  
            return x,target, stamp
        else:
            return x,target,torch.zeros(1)

    def __len__(self):
        return len(self.data_x) - self.args.seq_len - self.args.pred_len + 1
    
    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)


    
    
