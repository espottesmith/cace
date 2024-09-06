# Copyright (c) 2023 PyG Team <team@pyg.org>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import os.path as osp
import ssl
import urllib
import zipfile

import torch
from torch import Tensor


# Typing
WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT21 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 1
WITH_PT22 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 2
WITH_PT23 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 3
WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4
WITH_PT25 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 5
WITH_PT111 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 11
WITH_PT112 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 12
WITH_PT113 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 13


class SparseTensor:  # type: ignore
    def __init__(
        self,
        row: Optional[Tensor] = None,
        rowptr: Optional[Tensor] = None,
        col: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ):
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @classmethod
    def from_edge_index(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
        is_sorted: bool = False,
        trust_data: bool = False,
    ) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @property
    def storage(self) -> SparseStorage:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    @classmethod
    def from_dense(self, mat: Tensor,
                    has_value: bool = True) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def size(self, dim: int) -> int:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def nnz(self) -> int:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def is_cuda(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def has_value(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def set_value(self, value: Optional[Tensor],
                    layout: Optional[str] = None) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def fill_value(self, fill_value: float,
                    dtype: Optional[torch.dtype] = None) -> 'SparseTensor':
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def coo(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def csr(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def requires_grad(self) -> bool:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")

    def to_torch_sparse_csr_tensor(
        self,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        raise ImportError("'SparseTensor' requires 'torch-sparse'")


class torch_sparse:  # type: ignore
    @staticmethod
    def matmul(src: SparseTensor, other: Tensor,
                reduce: str = "sum") -> Tensor:
        raise ImportError("'matmul' requires 'torch-sparse'")

    @staticmethod
    def sum(src: SparseTensor, dim: Optional[int] = None) -> Tensor:
        raise ImportError("'sum' requires 'torch-sparse'")

    @staticmethod
    def mul(src: SparseTensor, other: Tensor) -> SparseTensor:
        raise ImportError("'mul' requires 'torch-sparse'")

    @staticmethod
    def set_diag(src: SparseTensor, values: Optional[Tensor] = None,
                    k: int = 0) -> SparseTensor:
        raise ImportError("'set_diag' requires 'torch-sparse'")

    @staticmethod
    def fill_diag(src: SparseTensor, fill_value: float,
                    k: int = 0) -> SparseTensor:
        raise ImportError("'fill_diag' requires 'torch-sparse'")

    @staticmethod
    def masked_select_nnz(src: SparseTensor, mask: Tensor,
                            layout: Optional[str] = None) -> SparseTensor:
        raise ImportError("'masked_select_nnz' requires 'torch-sparse'")


def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using exist file", filename)
        return path

    if log:
        print("Downloading", url)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)


def is_torch_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if (torch_geometric.typing.WITH_PT112
                and src.layout == torch.sparse_csc):
            return True
    return False


def is_sparse(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is of type
    :class:`torch.sparse.Tensor` (in any sparse layout) or of type
    :class:`torch_sparse.SparseTensor`.

    Args:
        src (Any): The input object to be checked.
    """
    return is_torch_sparse_tensor(src) or isinstance(src, SparseTensor)
