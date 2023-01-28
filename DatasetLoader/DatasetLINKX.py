import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset, download_url


class DatasetLINKX(InMemoryDataset):
    r"""A variety of non-homophilous graph datasets from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

    .. note::
        Some of the datasets provided in :class:`LINKXDataset` are from other
        sources, but have been updated with new features and/or labels.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"penn94"`,
            :obj:`"reed98"`, :obj:`"amherst41"`, :obj:`"cornell5"`,
            :obj:`"johnshopkins55"`, :obj:`"genius"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    github_url = ('https://github.com/CUAI/Non-Homophily-Large-Scale/'
                  'raw/master/data')
    gdrive_url = 'https://drive.google.com/uc?confirm=t&'

    facebook_datasets = [
        'penn94', 'reed98', 'amherst41', 'cornell5', 'johnshopkins55'
    ]

    datasets = {
        'penn94': {
            'data.mat': f'{github_url}/facebook100/Penn94.mat'
        },
        'reed98': {
            'data.mat': f'{github_url}/facebook100/Reed98.mat'
        },
        'amherst41': {
            'data.mat': f'{github_url}/facebook100/Amherst41.mat',
        },
        'cornell5': {
            'data.mat': f'{github_url}/facebook100/Cornell5.mat'
        },
        'johnshopkins55': {
            'data.mat': f'{github_url}/facebook100/Johns%20Hopkins55.mat'
        },
        'genius': {
            'data.mat': f'{github_url}/genius.mat'
        },
        'wiki': {
            'wiki_views2M.pt':
            f'{gdrive_url}id=1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',
            'wiki_edges2M.pt':
            f'{gdrive_url}id=14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',
            'wiki_features2M.pt':
            f'{gdrive_url}id=1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
        }
    }

    splits = {
        'penn94': f'{github_url}/splits/fb100-Penn94-splits.npy',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = list(self.datasets[self.name].keys())
        if self.name in self.splits:
            names += [self.splits[self.name].split('/')[-1]]
        return names

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for filename, path in self.datasets[self.name].items():
            fpath = download_url(path, self.raw_dir)
            newfile = os.path.dirname(fpath) + os.sep + filename
            os.rename(fpath, newfile)
        if self.name in self.splits:
            download_url(self.splits[self.name], self.raw_dir)

    def _process_wiki(self):

        paths = {x.split('/')[-1]: x for x in self.raw_paths}
        x = torch.load(paths['wiki_features2M.pt'])
        edge_index = torch.load(paths['wiki_edges2M.pt']).t().contiguous()
        y = torch.load(paths['wiki_views2M.pt'])

        return Data(x=x, edge_index=edge_index, y=y)

    def _process_facebook(self):
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])

        metadata = torch.from_numpy(mat['local_info'].astype('int64'))
        convMap = []
        newIdx = 0
        for y in metadata[:, 1]:
            if y == 0:
                convMap.append(-1)
            else:
                convMap.append(newIdx)
                newIdx += 1

        A = mat['A'].tocsr().tocoo()

        A_row = [ convMap[i] for i in A.row ]
        A_col = [ convMap[i] for i in A.col ]

        A_row_mod = []
        A_col_mod = []
        for i in range(len(A_row)):
            if A_row[i] != -1 and A_col[i] != -1:
                A_row_mod.append(A_row[i])
                A_col_mod.append(A_col[i])


        row = torch.from_numpy(np.array(A_row_mod)).to(torch.long)
        col = torch.from_numpy(np.array(A_col_mod)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        metadata = metadata[metadata[:, 1] != 0]

        xs = []
        y = metadata[:, 1] - 1
        x = torch.cat([metadata[:, :1], metadata[:, 2:]], dim=-1)
        for i in range(x.size(1)):
            _, out = x[:, i].unique(return_inverse=True)
            xs.append(F.one_hot(out).to(torch.float))
        x = torch.cat(xs, dim=-1)

        data = Data(x=x, edge_index=edge_index, y=y)

        return data

    def _process_genius(self):
        from scipy.io import loadmat

        mat = loadmat(self.raw_paths[0])
        edge_index = torch.from_numpy(mat['edge_index']).to(torch.long)
        x = torch.from_numpy(mat['node_feat']).to(torch.float)
        y = torch.from_numpy(mat['label']).squeeze().to(torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def process(self):
        if self.name in self.facebook_datasets:
            data = self._process_facebook()
        elif self.name == 'genius':
            data = self._process_genius()
        elif self.name == 'wiki':
            data = self._process_wiki()
        else:
            raise NotImplementedError(
                f"chosen dataset '{self.name}' is not implemented")

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}({len(self)})'