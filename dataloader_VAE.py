import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler


def get_h5ad_data(file_name='zebrafish_scNODE0_2000genes_3227cells_12tps.h5ad'):
    # dataset_dir = '/home/hanyuji/Workbench/Data/h5ad/'
    dataset_dir = '/home/hanyuji/Data/scNODE_data/h5ad/'
    
    loaded_adata = sc.read_h5ad(dataset_dir + file_name)

    timepoints = sorted(loaded_adata.obs['tp'].unique())

    data_list = []

    for tp in timepoints:
        # 选择对应时间点的细胞
        subset = loaded_adata[loaded_adata.obs['tp'] == tp]
        # 获取X矩阵
        X_matrix = subset.X.toarray() if hasattr(subset.X, "toarray") else subset.X
        # 添加到数组中
        data_list.append(X_matrix)

    return data_list


class scDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return sum(len(data) for data in self.data_list)

    def __getitem__(self, idx):
        # 查找idx所属的numpy数组及其对应的局部idx
        for label, data in enumerate(self.data_list):
            if idx < len(data):
                sample = data[idx]
                return sample, label
            idx -= len(data)


def get_dataloader(
    data_list,
    batch_size=64,
    shuffle=True,
):
    # data_list = get_h5ad_data()

    dataset = scDataset(data_list)
    dataloader = DataLoader(
        dataset, num_workers=4, batch_size=batch_size, shuffle=shuffle
    )

    return dataloader


def normalize(data_list):
    """
    归一化数据列表中的每个numpy数组
    """
    normalized_data_list = []
    scalers = []

    for data in data_list:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
        normalized_data_list.append(normalized_data)
        scalers.append(scaler)

    return normalized_data_list, scalers


def inverse_normalize(normalized_data_list, scalers):
    """
    对归一化后的数据列表中的每个numpy数组进行逆归一化
    """
    original_data_list = []

    for normalized_data, scaler in zip(normalized_data_list, scalers):
        original_data = scaler.inverse_transform(normalized_data)
        original_data_list.append(original_data)

    return original_data_list
