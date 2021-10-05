import os
import glob
import numpy as np
import pandas as pd

from config import Train_path, Train2_path, Test_csv_path

import torch
from torch.utils.data import Dataset

piece_to_idx = {'B121151': 0, 'B555894': 1, 'A79630': 2, 'B927695': 3,
                'A369919': 4, 'A65195': 5, 'A46872': 6, 'A511193': 7,
                'A314222': 8, 'A794147': 9, 'A714858': 10, 'A834782': 11,
                'A775826': 12, 'A393450': 13, 'A330530': 14, 'A35776': 15,
                'A9017': 16, 'A84462': 17, 'A37248': 18, 'A726846': 19,
                'A87907589': 20, 'A859833': 21, 'A913844': 22, 'A778989': 23,
                'A538130': 24, 'A620395': 25, 'A555894': 26, 'A121151': 27,
                'A927695': 28, 'A11887': 29, 'A85484': 30, 'A47143': 31}
group_to_idx = {'G13_715': 0, 'G49_1030': 1, 'G15_1230': 2,
                'G02_715': 3, 'G11-5_715': 4, 'G01_715': 5, 'G21_1230': 6}

idx_to_piece = {value: key for key, value in piece_to_idx.items()}
idx_to_group = {value: key for key, value in group_to_idx.items()}

piece_count = [5, 3, 7, 7, 4, 3, 3]
piece_shift = [0, 5, 8, 15, 22, 26, 29]

group = 0
piece_to_group = {}
for piece in range(len(piece_to_idx)):
    if group + 1 < len(piece_shift) and piece == piece_shift[group + 1]:
        group += 1
    piece_to_group[piece] = group


class GeneralDataset(Dataset):
    def __init__(self, paths):
        groups = []
        for path in paths:
            groups += list(glob.glob(path + '/*'))

        datas = []
        self.groups = []
        self.pieces = []
        self.files = []
        self.ids = []
        self.lens = []

        max_len = 0
        for group_path in groups:
            group = group_to_idx[os.path.basename(group_path)]
            files = glob.glob(group_path + '/*/*.csv')
            for file in files:
                df = pd.read_csv(file, header=None).fillna(value=0)
                self.ids.append(df.values[0])
                datas.append(df.values[2:].astype(np.float32).T)
                piece = piece_to_idx[
                    os.path.basename(file).replace('.csv', '')]
                self.groups.extend([group for _ in range(len(datas[-1]))])
                self.pieces.extend([piece for _ in range(len(datas[-1]))])
                self.files.extend([file for _ in range(len(datas[-1]))])
                self.lens.extend(
                    [datas[-1].shape[1] for _ in range(len(datas[-1]))])
                max_len = max(max_len, datas[-1].shape[1])
        self.datas = []
        for data in datas:
            now_len = data.shape[1]
            self.datas.append(np.pad(data, [(0, 0), (0, max_len - now_len)]))
        self.ids = np.concatenate(self.ids, axis=0)
        self.datas = np.concatenate(self.datas, axis=0)
        self.groups = np.array(self.groups)
        self.pieces = np.array(self.pieces)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return (
            self.datas[idx, None, :24],
            self.datas[idx, None, :],
            self.groups[idx],
            self.pieces[idx],
        )


class TrainDataset(GeneralDataset):
    def __init__(self, paths=[Train_path]):
        super().__init__(paths)


class FinalDataset(GeneralDataset):
    def __init__(self, paths=[Train_path, Train2_path]):
        super().__init__(paths)


class TestDataset(Dataset):
    def __init__(self, path=Test_csv_path):
        df = pd.read_csv(path, header=None).T
        self.ids = df.values[:, 0].astype(str)
        self.datas = df.values[:, 1:].astype(np.float32)
        self.datas = self.datas[:, :24]
        self.lens = 24 - np.isnan(self.datas).astype(np.int).sum(axis=1)
        self.datas = np.nan_to_num(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        front = self.datas[idx, None]
        return front, np.zeros((1, 300)).astype(np.float32), 0, 0


class PreprocessDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        front, other, group, piece = self.dataset[idx]
        if self.transform is not None:
            front = self.transform(front)
            other = self.transform(other)
        return front, other, group, piece


class RandomCut:
    def __init__(self, a=20, b=25):
        self.a = a
        self.b = b

    def __call__(self, data):
        if data.shape[1] == 24:
            length = np.random.randint(self.a, self.b)
            data = data.copy()
            data[:, length:] = 0
        return data


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std


class Difference:
    def __init__(self, degree=0):
        self.degree = degree

    def __call__(self, data):
        datas = [data]
        for _ in range(self.degree):
            diff = datas[-1][:, 1:] - datas[-1][:, :-1]
            diff = np.pad(diff, [(0, 0), (0, 1)])
            datas.append(diff)
        data = np.concatenate(datas, axis=0)
        return data


class MovAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def __call__(self, data):
        cum_sum = np.pad(
            data[0],
            (1, self.window_size - 1),
            constant_values=(0, data[0, -1]))
        cum_sum = np.cumsum(cum_sum, dtype=np.float32)
        mov_avg = cum_sum[self.window_size:] - cum_sum[:-self.window_size]
        data = np.concatenate([data, mov_avg[None]], axis=0)
        return data


class ExpAverage:
    def __init__(self, decay=0.3):
        self.decay = decay

    def __call__(self, data):
        now = data[0][0]
        ema = [now]
        for value in data[0][1:]:
            now = now * self.decay + value * (1 - self.decay)
            ema.append(now)
        ema = np.array(ema)[None].astype(np.float32)
        data = np.concatenate([data, ema], axis=0)
        return data


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def get_statistic(dataset):
    square_summ = 0
    summ = 0
    n_element = 0
    for _, other, _, _ in dataset:
        n_element += (other != 0).sum()
        square_summ += (other * other).sum()
        summ += other.sum()
    mean = summ / n_element
    std = np.sqrt(square_summ / n_element - mean * mean)
    return mean, std


if __name__ == '__main__':
    # Train all
    finaldataset = FinalDataset()
    print('Final dataset shape:', finaldataset.datas.shape)
    print('length min :', np.min(finaldataset.lens))
    print('length max :', np.max(finaldataset.lens))
    print('length mean:', np.mean(finaldataset.lens))
    print('length std :', np.std(finaldataset.lens))

    test_dataset = TestDataset()
    print('Test dataset shape :', test_dataset.datas.shape)
    print('length min :', np.min(test_dataset.lens))
    print('length max :', np.max(test_dataset.lens))
    print('length mean:', np.mean(test_dataset.lens))
    print('length std :', np.std(test_dataset.lens))
