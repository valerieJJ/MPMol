import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip)
from typing import Optional, Callable, List
import os.path as osp

class DatasetZINC(InMemoryDataset):
    def __init__(self,  root: Optional[str], split='train', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.split = split
        super(DatasetZINC,self).__init__(root, transform, pre_transform, pre_filter)
        loadpath = self.processed_paths[0]
        if split=="train":
            loadpath = self.processed_paths[0]
        if split== "val":
            loadpath = self.processed_paths[1]
        if split=="test":
            loadpath = self.processed_paths[2]

        self.data, self.slices = torch.load(loadpath)

    @property
    def raw_file_names(self) -> List[str]:
        return ["train_ZINC_mp.pkl",
                "val_ZINC_mp.pkl",
                "test_ZINC_mp.pkl"]

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self) -> str:
        return ['train_dataset-zinc_mp.pkl',
                'val_dataset_zinc_mp.pkl',
                'test_dataset_zinc_mp.pkl']

    def process(self):
        print("zinc-dataset processing")

        if self.split=="train":
            train_data_list = torch.load(self.raw_paths[0])
            torch.save(self.collate(train_data_list), self.processed_paths[0])
        if self.split=="val":
            val_data_list = torch.load(self.raw_paths[1])
            torch.save(self.collate(val_data_list), self.processed_paths[1])
        if self.split=="test":
            test_data_list = torch.load(self.raw_paths[2])
            torch.save(self.collate(test_data_list), self.processed_paths[2])
        return
