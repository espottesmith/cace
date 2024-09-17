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
import typing
import urllib
import zipfile

import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack


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

MAX_INT64 = torch.iinfo(torch.int64).max

OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
MISSING = '???'


class AttrType(Enum):
    NODE = 'NODE'
    EDGE = 'EDGE'
    OTHER = 'OTHER'


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


def coalesce(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    reduce: str = 'sum',
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (str, optional): The reduce operation to use for merging edge
            features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Example:
        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> coalesce(edge_index)
        tensor([[1, 2, 3],
                [3, 1, 2]])

        >>> # Sort `edge_index` column-wise
        >>> coalesce(edge_index, sort_by_row=False)
        tensor([[2, 3, 1],
                [1, 2, 3]])

        >>> coalesce(edge_index, edge_attr)
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([2., 1., 1.]))

        >>> # Use 'mean' operation to merge edge features
        >>> coalesce(edge_index, edge_attr, reduce='mean')
        (tensor([[1, 2, 3],
                [3, 1, 2]]),
        tensor([1., 1., 1.]))
    """
    num_edges = edge_index[0].size(0)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if num_nodes * num_nodes > MAX_INT64:
        raise ValueError("'coalesce' will result in an overflow")

    idx = edge_index[0].new_empty(num_edges + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = index_sort(idx[1:], max_value=num_nodes * num_nodes)
        if isinstance(edge_index, Tensor):
            edge_index = edge_index[:, perm]
        elif isinstance(edge_index, tuple):
            edge_index = (edge_index[0][perm], edge_index[1][perm])
        else:
            raise NotImplementedError
        if isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        if edge_attr is None or isinstance(edge_attr, Tensor):
            return edge_index, edge_attr
        if isinstance(edge_attr, (list, tuple)):
            return edge_index, edge_attr
        return edge_index

    if isinstance(edge_index, Tensor):
        edge_index = edge_index[:, mask]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][mask], edge_index[1][mask])
    else:
        raise NotImplementedError

    dim_size: Optional[int] = None
    if isinstance(edge_attr, (Tensor, list, tuple)) and len(edge_attr) > 0:
        dim_size = edge_index.size(1)
        idx = torch.arange(0, num_edges, device=edge_index.device)
        idx.sub_(mask.logical_not_().cumsum(dim=0))

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
        return edge_index, edge_attr
    if isinstance(edge_attr, (list, tuple)):
        if len(edge_attr) == 0:
            return edge_index, edge_attr
        edge_attr = [scatter(e, idx, 0, dim_size, reduce) for e in edge_attr]
        return edge_index, edge_attr

    return edge_index


def to_torch_coo_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_coo`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_coo)

    """
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    if not WITH_PT21:
        adj = torch.sparse_coo_tensor(
            indices=edge_index,
            values=edge_attr,
            size=tuple(size) + edge_attr.size()[1:],
            device=edge_index.device,
        )
        adj = adj._coalesced_(True)
        return adj

    return torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
        is_coalesced=True,
    )


def to_torch_csr_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    is_coalesced: bool = False,
) -> Tensor:
    r"""Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor` with layout
    `torch.sparse_csr`.
    See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
        is_coalesced (bool): If set to :obj:`True`, will assume that
            :obj:`edge_index` is already coalesced and thus avoids expensive
            computation. (default: :obj:`False`)

    :rtype: :class:`torch.sparse.Tensor`

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_csr_tensor(edge_index)
        tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
               col_indices=tensor([1, 0, 2, 1, 3, 2]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(4, 4), nnz=6, layout=torch.sparse_csr)

    """
    if size is None:
        size = int(edge_index.max()) + 1

    if isinstance(size, (tuple, list)):
        num_src_nodes, num_dst_nodes = size
        if num_src_nodes is None:
            num_src_nodes = int(edge_index[0].max()) + 1
        if num_dst_nodes is None:
            num_dst_nodes = int(edge_index[1].max()) + 1
        size = (num_src_nodes, num_dst_nodes)
    else:
        size = (size, size)

    if not is_coalesced:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, max(size))

    if edge_attr is None:
        # Expanded tensors are not yet supported in all PyTorch code paths :(
        # edge_attr = torch.ones(1, device=edge_index.device)
        # edge_attr = edge_attr.expand(edge_index.size(1))
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_csr_tensor(
        crow_indices=index2ptr(edge_index[0], size[0]),
        col_indices=edge_index[1],
        values=edge_attr,
        size=tuple(size) + edge_attr.size()[1:],
        device=edge_index.device,
    )

    return adj


def ptr2index(ptr: Tensor, output_size: Optional[int] = None) -> Tensor:
    index = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return index.repeat_interleave(ptr.diff(), output_size=output_size)


def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    if size is None:
        size = int(index.max()) + 1 if index.numel() > 0 else 0

    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype != torch.int64)


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


def contains_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> contains_isolated_nodes(edge_index)
        False

        >>> contains_isolated_nodes(edge_index, num_nodes=3)
        True
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, _ = remove_self_loops(edge_index)
    return torch.unique(edge_index.view(-1)).numel() < num_nodes


def maybe_num_nodes(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor], SparseTensor],
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if torch_geometric.utils.is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))

        if torch.jit.is_tracing():
            # Avoid non-traceable if-check for empty `edge_index` tensor:
            tmp = torch.concat([
                edge_index.view(-1),
                edge_index.new_full((1, ), fill_value=-1)
            ])
            return tmp.max() + 1  # type: ignore

        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    elif isinstance(edge_index, tuple):
        return max(
            int(edge_index[0].max()) + 1 if edge_index[0].numel() > 0 else 0,
            int(edge_index[1].max()) + 1 if edge_index[1].numel() > 0 else 0,
        )
    elif isinstance(edge_index, SparseTensor):
        return max(edge_index.size(0), edge_index.size(1))
    raise NotImplementedError


def to_edge_index(adj: Union[Tensor, SparseTensor]) -> Tuple[Tensor, Tensor]:
    r"""Converts a :class:`torch.sparse.Tensor` or a
    :class:`torch_sparse.SparseTensor` to edge indices and edge attributes.

    Args:
        adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> adj = to_torch_coo_tensor(edge_index)
        >>> to_edge_index(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    if isinstance(adj, SparseTensor):
        row, col, value = adj.coo()
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.stack([row, col], dim=0).long(), value

    if adj.layout == torch.sparse_coo:
        adj = adj._coalesced_(True)
        return adj.indices().detach().long(), adj.values()

    if adj.layout == torch.sparse_csr:
        row = ptr2index(adj.crow_indices().detach())
        col = adj.col_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    if WITH_PT112 and adj.layout == torch.sparse_csc:
        col = ptr2index(adj.ccol_indices().detach())
        row = adj.row_indices().detach()
        return torch.stack([row, col], dim=0).long(), adj.values()

    raise ValueError(f"Unexpected sparse tensor layout (got '{adj.layout}')")


def remove_self_loops(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Example:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_attr = [[1, 2], [3, 4], [5, 6]]
        >>> edge_attr = torch.tensor(edge_attr)
        >>> remove_self_loops(edge_index, edge_attr)
        (tensor([[0, 1],
                [1, 0]]),
        tensor([[1, 2],
                [3, 4]]))
    """
    size: Optional[Tuple[int, int]] = None
    if not typing.TYPE_CHECKING and torch.jit.is_scripting():
        layout: Optional[int] = None
    else:
        layout: Optional[torch.layout] = None

    value: Optional[Tensor] = None
    if is_torch_sparse_tensor(edge_index):
        layout = edge_index.layout
        size = (edge_index.size(0), edge_index.size(1))
        edge_index, value = to_edge_index(edge_index)

    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if layout is not None:
        assert edge_attr is None
        assert value is not None
        value = value[mask]
        if str(layout) == 'torch.sparse_coo':  # str(...) for TorchScript :(
            return to_torch_coo_tensor(edge_index, value, size, True), None
        elif str(layout) == 'torch.sparse_csr':
            return to_torch_csr_tensor(edge_index, value, size, True), None
        raise ValueError(f"Unexpected sparse tensor layout (got '{layout}')")

    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    if not WITH_PT113:
        keys = [k.neg() for k in keys] if descending else keys
        out = np.lexsort([k.detach().cpu().numpy() for k in keys], axis=dim)
        return torch.from_numpy(out).to(keys[0].device)

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)

    return out


def sort_edge_index(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[OptTensor, List[Tensor], str] = MISSING,
    num_nodes: Optional[int] = None,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, OptTensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index`.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
            or multi-dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise/by destination node.
            (default: :obj:`True`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
        (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

    .. warning::

        From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
        tuple whenever :obj:`edge_attr` is passed as an argument (even in case
        it is set to :obj:`None`).

    Examples:
        >>> edge_index = torch.tensor([[2, 1, 1, 0],
                                [1, 2, 0, 1]])
        >>> edge_attr = torch.tensor([[1], [2], [3], [4]])
        >>> sort_edge_index(edge_index)
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]])

        >>> sort_edge_index(edge_index, edge_attr)
        (tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]),
        tensor([[4],
                [3],
                [2],
                [1]]))
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if num_nodes * num_nodes > MAX_INT64:
        if not WITH_PT113:
            raise ValueError("'sort_edge_index' will result in an overflow")
        perm = lexsort(keys=[
            edge_index[int(sort_by_row)],
            edge_index[1 - int(sort_by_row)],
        ])
    else:
        idx = edge_index[1 - int(sort_by_row)] * num_nodes
        idx += edge_index[int(sort_by_row)]
        _, perm = index_sort(idx, max_value=num_nodes * num_nodes)

    if isinstance(edge_index, Tensor):
        edge_index = edge_index[:, perm]
    elif isinstance(edge_index, tuple):
        edge_index = (edge_index[0][perm], edge_index[1][perm])
    else:
        raise NotImplementedError

    if edge_attr is None:
        return edge_index, None
    if isinstance(edge_attr, Tensor):
        return edge_index, edge_attr[perm]
    if isinstance(edge_attr, (list, tuple)):
        return edge_index, [e[perm] for e in edge_attr]

    return edge_index


def is_undirected(  # noqa: F811
    edge_index: Tensor,
    edge_attr: Union[Optional[Tensor], List[Tensor]] = None,
    num_nodes: Optional[int] = None,
) -> bool:
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will check for equivalence in all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)

    :rtype: bool

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                         [1, 0, 0]])
        >>> weight = torch.tensor([0, 0, 1])
        >>> is_undirected(edge_index, weight)
        True

        >>> weight = torch.tensor([0, 1, 1])
        >>> is_undirected(edge_index, weight)
        False

    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_attrs: List[Tensor] = []
    if isinstance(edge_attr, Tensor):
        edge_attrs.append(edge_attr)
    elif isinstance(edge_attr, (list, tuple)):
        edge_attrs = edge_attr

    edge_index1, edge_attrs1 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=True,
    )
    edge_index2, edge_attrs2 = sort_edge_index(
        edge_index,
        edge_attrs,
        num_nodes=num_nodes,
        sort_by_row=False,
    )

    if not torch.equal(edge_index1[0], edge_index2[1]):
        return False

    if not torch.equal(edge_index1[1], edge_index2[0]):
        return False

    assert isinstance(edge_attrs1, list) and isinstance(edge_attrs2, list)
    for edge_attr1, edge_attr2 in zip(edge_attrs1, edge_attrs2):
        if not torch.equal(edge_attr1, edge_attr2):
            return False

    return True


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        index (Tensor): The indices.
        size (int, optional): The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:
        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


# TODO: this
def map_index(
    src: Tensor,
    index: Tensor,
    max_index: Optional[Union[int, Tensor]] = None,
    inclusive: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Maps indices in :obj:`src` to the positional value of their
    corresponding occurrence in :obj:`index`.
    Indices must be strictly positive.

    Args:
        src (torch.Tensor): The source tensor to map.
        index (torch.Tensor): The index tensor that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to :obj:`True`, it is assumed that
            every entry in :obj:`src` has a valid entry in :obj:`index`.
            Can speed-up computation. (default: :obj:`False`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.BoolTensor`)

    Examples:
        >>> src = torch.tensor([2, 0, 1, 0, 3])
        >>> index = torch.tensor([3, 2, 0, 1])

        >>> map_index(src, index)
        (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))

        >>> src = torch.tensor([2, 0, 1, 0, 3])
        >>> index = torch.tensor([3, 2, 0])

        >>> map_index(src, index)
        (tensor([1, 2, 2, 0]), tensor([True, True, False, True, True]))

    .. note::

        If inputs are on GPU and :obj:`cudf` is available, consider using RMM
        for significant speed boosts.
        Proceed with caution as RMM may conflict with other allocators or
        fragments.

        .. code-block:: python

            import rmm
            rmm.reinitialize(pool_allocator=True)
            torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)
    """
    if src.is_floating_point():
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.is_floating_point():
        raise ValueError(f"Expected 'index' to be an index (got "
                         f"'{index.dtype}')")
    if src.device != index.device:
        raise ValueError(f"Both 'src' and 'index' must be on the same device "
                         f"(got '{src.device}' and '{index.device}')")

    if max_index is None:
        max_index = torch.maximum(src.max(), index.max())

    # If the `max_index` is in a reasonable range, we can accelerate this
    # operation by creating a helper vector to perform the mapping.
    # NOTE This will potentially consumes a large chunk of memory
    # (max_index=10 million => ~75MB), so we cap it at a reasonable size:
    THRESHOLD = 40_000_000 if src.is_cuda else 10_000_000
    if max_index <= THRESHOLD:
        if inclusive:
            assoc = src.new_empty(max_index + 1)  # type: ignore
        else:
            assoc = src.new_full((max_index + 1, ), -1)  # type: ignore
        assoc[index] = torch.arange(index.numel(), dtype=src.dtype,
                                    device=src.device)
        out = assoc[src]

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    WITH_CUDF = False
    if src.is_cuda:
        try:
            import cudf
            WITH_CUDF = True
        except ImportError:
            import pandas as pd
            warnings.warn("Using CPU-based processing within 'map_index' "
                          "which may cause slowdowns and device "
                          "synchronization. Consider installing 'cudf' to "
                          "accelerate computation")
    else:
        import pandas as pd

    if not WITH_CUDF:
        left_ser = pd.Series(src.cpu().numpy(), name='left_ser')
        right_ser = pd.Series(
            index=index.cpu().numpy(),
            data=pd.RangeIndex(0, index.size(0)),
            name='right_ser',
        )

        result = pd.merge(left_ser, right_ser, how='left', left_on='left_ser',
                          right_index=True)

        out_numpy = result['right_ser'].values
        if (index.device.type == 'mps'  # MPS does not support `float64`
                and issubclass(out_numpy.dtype.type, np.floating)):
            out_numpy = out_numpy.astype(np.float32)

        out = torch.from_numpy(out_numpy).to(index.device)

        if out.is_floating_point() and inclusive:
            raise ValueError("Found invalid entries in 'src' that do not have "
                             "a corresponding entry in 'index'. Set "
                             "`inclusive=False` to ignore these entries.")

        if out.is_floating_point():
            mask = torch.isnan(out).logical_not_()
            out = out[mask].to(index.dtype)
            return out, mask

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    else:
        left_ser = cudf.Series(src, name='left_ser')
        right_ser = cudf.Series(
            index=index,
            data=cudf.RangeIndex(0, index.size(0)),
            name='right_ser',
        )

        result = cudf.merge(left_ser, right_ser, how='left',
                            left_on='left_ser', right_index=True, sort=True)

        if inclusive:
            try:
                out = from_dlpack(result['right_ser'].to_dlpack())
            except ValueError:
                raise ValueError("Found invalid entries in 'src' that do not "
                                 "have a corresponding entry in 'index'. Set "
                                 "`inclusive=False` to ignore these entries.")
        else:
            out = from_dlpack(result['right_ser'].fillna(-1).to_dlpack())

        out = out[src.argsort().argsort()]  # Restore original order.

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    size: Optional[Tuple[int, int]] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
        ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]))

        >>> bipartite_subgraph(subset, edge_index, edge_attr,
        ...                    return_edge_mask=True)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]),
        tensor([False, False,  True,  True, False, False, False, False,
                True,  True,  False]))
    """
    device = edge_index.device

    src_subset, dst_subset = subset
    if not isinstance(src_subset, Tensor):
        src_subset = torch.tensor(src_subset, dtype=torch.long, device=device)
    if not isinstance(dst_subset, Tensor):
        dst_subset = torch.tensor(dst_subset, dtype=torch.long, device=device)

    if src_subset.dtype != torch.bool:
        src_size = int(edge_index[0].max()) + 1 if size is None else size[0]
        src_node_mask = index_to_mask(src_subset, size=src_size)
    else:
        src_size = src_subset.size(0)
        src_node_mask = src_subset
        src_subset = src_subset.nonzero().view(-1)

    if dst_subset.dtype != torch.bool:
        dst_size = int(edge_index[1].max()) + 1 if size is None else size[1]
        dst_node_mask = index_to_mask(dst_subset, size=dst_size)
    else:
        dst_size = dst_subset.size(0)
        dst_node_mask = dst_subset
        dst_subset = dst_subset.nonzero().view(-1)

    edge_mask = src_node_mask[edge_index[0]] & dst_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        src_index, _ = map_index(edge_index[0], src_subset, max_index=src_size,
                                 inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_subset, max_index=dst_size,
                                 inclusive=True)
        edge_index = torch.stack([src_index, dst_index], dim=0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr
