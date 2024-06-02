import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scanpy as sc


def get_h5ad_data(file_name='zebrafish_scNODE0_2000genes_3227cells_12tps.h5ad'):
    dataset_dir = '/home/hanyuji/Workbench/Data/h5ad/'
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
    batch_size=64,
    shuffle=True,
    file_name='zebrafish_scNODE0_2000genes_3227cells_12tps.h5ad',
):

    data_list = get_h5ad_data(file_name)

    dataset = scDataset(data_list)
    dataloader = DataLoader(
        dataset, num_workers=4, batch_size=batch_size, shuffle=shuffle
    )

    return dataloader
