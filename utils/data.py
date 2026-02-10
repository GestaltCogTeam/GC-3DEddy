from torch.utils.data import Dataset
import os, json
import numpy as np
from typing import List


def build_sea_dataset(data_path: str, datasets_name: str, data_load_channel: int, data_load_reso: int, time_patch_num: int, vae_channels_list: List, train_val_test_ratio = [0.9, 0.05, 0.05]):
    # build dataset
    train_set = SeaDataset(data_path, datasets_name, train_val_test_ratio, 'train', data_load_channel, data_load_reso, time_patch_num, vae_channels_list)
    val_set = SeaDataset(data_path, datasets_name, train_val_test_ratio, 'valid', data_load_channel, data_load_reso, time_patch_num, vae_channels_list)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}')

    return train_set, val_set

def build_sea_test_dataset(data_path: str, datasets_name: str, data_load_channel: int, data_load_reso: int, time_patch_num: int, vae_channels_list: List, train_val_test_ratio = [0.9, 0.05, 0.05]):
    # build dataset
    test_set = SeaDataset(data_path, datasets_name, train_val_test_ratio, 'test', data_load_channel, data_load_reso, time_patch_num, vae_channels_list)
    print(f'[Dataset] {len(test_set)=}')

    return test_set

class SeaDataset(Dataset):

    def __init__(
            self,
            data_path: str,
            datasets_name: str,
            train_val_test_ratio: List[float],
            mode: str,
            data_load_channel: int,
            data_load_reso: int,
            time_patch_num: int,
            vae_channels_list: List[int],
        ) -> None:

        super().__init__()

        assert os.path.exists(data_path), f"Invalid data_path. Must be valid path."
        assert len(train_val_test_ratio) == 3, f"Invalid train_val_test_ratio. Must be [float, float, float]."
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."

        self.data_file_path = f'{data_path}/data' + (f'_{datasets_name}.dat' if datasets_name else '.dat')
        self.description_file_path = f'{data_path}/desc' + (f'_{datasets_name}.json' if datasets_name else '.json')
        self.train_val_test_ratio = train_val_test_ratio
        self.mode = mode
        self.time_patch_num = time_patch_num

        self.description = self._load_description(self.description_file_path)
        self.data = self._load_data(self.data_file_path, self.description)
        self.data_shape = self.data.shape
        self.data_load_channel = data_load_channel
        self.data_load_reso = data_load_reso
        self.vae_channels_list = [a + self.data_load_channel for a in vae_channels_list]

    def _load_description(self, description_file_path) -> dict:

        try:
            with open(description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {description_file_path}') from e

    def _load_data(self, data_file_path, description) -> np.ndarray:

        try:
            data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        if self.mode == 'train':
            return data[:train_len]
        elif self.mode == 'valid':
            return data[train_len:train_len+valid_len]
        else: # self.mode == 'test'
            return data[train_len+valid_len:]

    def __getitem__(self, index: int) -> dict:
        h, w = self.data_shape[-2:]
        cur_h = self.data_load_reso
        cur_w = self.data_load_reso
        start_lat = (h - cur_h) // 2
        start_lon = (w - cur_w) // 2
        if self.mode == 'train':
            # random crop
            start_lat = np.random.randint(0, h - cur_h)
            start_lon = np.random.randint(0, w - cur_w)
        return self.data[index, :self.time_patch_num, self.vae_channels_list, start_lat:start_lat+cur_h, start_lon:start_lon+cur_w].transpose(1,0,2,3), \
                self.data[index, 0, :self.data_load_channel, start_lat:start_lat+cur_h, start_lon:start_lon+cur_w]

    def __len__(self) -> int:

        return self.data_shape[0]