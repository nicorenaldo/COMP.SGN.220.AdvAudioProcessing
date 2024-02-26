#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union, Dict
from pickle import load as pickle_load
from pathlib import Path

from torch.utils.data import Dataset
import numpy

from utils import get_files_from_dir_with_pathlib


__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self,
                 root_dir: Union[str, Path],
                 split: str = 'train',
                 ) \
            -> None:
        """An example of an object of class torch.utils.data.Dataset

        :param root_dir: Root directory of the dataset.
        :type root_dir: str or pathlib.Path
        :param split: Split to use (training or testing), defaults to 'training'.
        :type split: str
        :param key_features: Key of the features in the files, defaults to 'features'.
        :type key_features: str
        :param key_class: Key of the
        :type key_class: str
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split

        if split not in ['train', 'test', 'val']:
            raise ValueError(f"Split {split} not supported. Use 'train', 'test', or 'val'.")

        data_path = self.root_dir / self.split
        self.files = get_files_from_dir_with_pathlib(data_path)
        self.data = []
        for i, a_file in enumerate(self.files):
            self.data.append(self._load_file(a_file))

    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, numpy.ndarray]]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|numpy.ndarray]
        """
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        item: Dict[str, Union[int, numpy.ndarray]] = self.data[item]
        return item["features"], item["label"]


if __name__ == '__main__':
    dataset_dir = Path('dataset/features')
    dataset = MyDataset(dataset_dir, split='test')
    features, label = dataset[0]
    print(features.shape)
    print(label)
    print(len(dataset))
