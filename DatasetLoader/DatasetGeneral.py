import zipfile
import torch
import json
import scipy
import scipy.io
import numpy as np
from os import path
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, Data
from DatasetLoader.data_utils import dataset_drive_url, even_quantile_labels
from google_drive_downloader import GoogleDriveDownloader as gdd

##
# Install ogb
# !pip install ogb
##


class DatasetGeneral(InMemoryDataset):
    """
    name: "lastfm_asia", "deezer_europe"
    """
    def __init__(self, name, features, label, edges):
        self.dataset_name = name
        super(DatasetGeneral, self).__init__("saved_data", None, None)
        self.data = Data(x=features, y=label, edge_index=edges)
